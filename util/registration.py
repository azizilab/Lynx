import os
import cv2
import tifffile
import numpy as np

from typing import List, Dict
from skimage.filters import gaussian as gaussian_blur
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2gray
from valis import registration
from valis.non_rigid_registrars import OpticalFlowWarper
from valis.warp_tools import warp_img
from __init__ import LOGGER


def get_affine_matrix(
    source: np.ndarray, 
    target: np.ndarray,
    pts_source: List[tuple] = None,  
    pts_target: List[tuple] = None,
    sigma: float = 5,
    n_matches: int = 50
) -> np.ndarray:
    """
    Compute 2x3 Affine transformation matrix by registering 
    `source` image against `target` (SIFT)
    """
    img_src = source.copy()
    img_dst = target.copy()

    # AHE & gaussian filter
    img_src = gaussian_blur(equalize_adapthist(img_src), sigma=sigma)
    img_dst = gaussian_blur(equalize_adapthist(img_dst), sigma=sigma)

    if img_src.max() <= 1:
        img_src = np.round(img_src*255).astype(np.uint8)
    if img_dst.max() <= 1:
        img_dst = np.round(img_dst*255).astype(np.uint8)

    if pts_source is not None and pts_target is not None:
        assert len(pts_source) == len(pts_target), \
            "Anchor pts btw source & target should have equal number"
        pts_source = np.float32(pts_source).reshape(-1, 1, 2)
        pts_target = np.float32(pts_target).reshape(-1, 1, 2)

    else:  # Finding anchor pts w/ SIFT
        sift = cv2.SIFT_create()
        pts_src, des_src = sift.detectAndCompute(img_src, None)
        pts_dst, des_dst = sift.detectAndCompute(img_dst, None)
        
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des_src, des_dst, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        # If insufficient anchor points (likely causing misalignment)
        # expand the searcing space, & choose the top 3 anchors
        # (min. requirement for computing affine transformation)
        sort_matches = False
        if len(good_matches) < n_matches:
            sort_matches = True
            good_matches = []
            for m, n in matches:
                if m.distance < 0.9*n.distance:
                    good_matches.append(m)
                    
        pts_source, pts_target = [], []
        for m in good_matches:
            pt1, pt2 = pts_src[m.queryIdx].pt, pts_dst[m.trainIdx].pt
            if pt1 not in pts_source and pt2 not in pts_target:
                pts_source.append(pt1)
                pts_target.append(pt2)

        pts_source = np.float32(pts_source).reshape(-1, 1, 2)
        pts_target = np.float32(pts_target).reshape(-1, 1, 2)

        if sort_matches:
            pts_source, pts_target = _reorder_points(pts_source, pts_target)
            pts_source, pts_target = pts_source[:5], pts_target[:5]
            LOGGER.warning('Re-calculated pts w/ larger search space')

    M, _ = cv2.estimateAffinePartial2D(pts_source, pts_target, cv2.RANSAC) 
    return M


def affine_warp(
    img_src: np.ndarray,
    dst_shape,
    M: np.ndarray = np.array([[1,0,0], [0,1,0]], dtype=np.float32)
):
    """Compute Warped image given precomputed transformation matrix"""
    return cv2.warpAffine(img_src, M, (dst_shape[1], dst_shape[0])) 
        

def _reorder_points(pts1, pts2):
    # Calculate distances
    size = min(len(pts1), len(pts2))
    dists = [(i, np.linalg.norm(pts1[i]-pts2[i])) 
            for i in range(size)]

    # Sort the distances by ascending order
    sorted_dists = sorted(dists, key=lambda x: x[1])
    sorted_pts1 = np.array([pts1[i] for i, _ in sorted_dists])
    sorted_pts2 = np.array([pts2[i] for i, _ in sorted_dists])
    return sorted_pts1, sorted_pts2


def non_rigid_warp(
    source: np.ndarray,
    target: np.ndarray,
    bk_dxdy: np.ndarray = None
):
    """
    Non-rigid Alignmeng / Warping w/ Optical Flow backbone
    """
    assert source.ndim <= 3, \
        "Only support 2D / 3D images"    
    shape = source.shape[:2]

    if bk_dxdy is None:
        assert source.ndim == target.ndim, \
            "Source and target ndim must be equal"
        
        img_src = source.copy()
        img_dst = target.copy()
        
        if source.ndim == 3:
            img_src = rgb2gray(img_src)
            img_dst = rgb2gray(img_dst)
    
        # Calc. optical flow
        registrar = OpticalFlowWarper()
        bk_dxdy = registrar.calc(moving_img=img_src, fixed_img=img_dst)

    # Warping original images
    if source.ndim == 3: # Warping RGB image, dim: (Y, X, C)
        img_src_warped = np.zeros_like(source, dtype=np.uint8)
        for i, chan in enumerate(source.transpose(2,0,1)):  
            chan_warped = warp_img(chan.astype(np.float32)/255.0, bk_dxdy=bk_dxdy, out_shape_rc=shape)
            img_src_warped[:,:,i] = np.round(chan_warped*255).astype(np.uint8)

    else:  # Warping grayscale image, dim: (Y, X)
        img_src_warped = warp_img(source, bk_dxdy=bk_dxdy, out_shape_rc=shape)

    return img_src_warped, bk_dxdy


def run_valis_multi(
    src_dir: str,
    res_dir: str,
    ref_prefix: str=None,
    file_prefixes: Dict[str, str]=None,
    mdata_dict: Dict=None,
    kill_jvm=False
):
    """
    End-to-End multimodal registration with VALIS

    file structure should be:
    src_dir/
    ├─ slice_01/
    │  ├─ he_image.tif
    │  ├─ cyif_image.ome.tif
    │  ├─ ...
    ├─ slice_02/
    ├─ slice_03/
    ├─ ...

    Optional Args:
        ref_prefix: prefix of reference image type for every slice, e.g. "HE_"
        file_prefixes: dictionary of prefixes to image types, with (key, value) pairs
                       as (prefix, image_type). Used for saving with metadata.
        mdata_dict: dictionary of metadata. (key, value) pairs as
                    (image_type, metadata). Used for saving with metadata.
    """
    for dir in os.listdir(src_dir):
        print("warping", dir, "...")
        
        slide_src_dir = os.path.join(src_dir, dir)
        reference_slide = None
        reg_slides = []
    
        for file in os.listdir(slide_src_dir):
            if file.startswith(ref_prefix):
                reference_slide = file
            else:
                reg_slides.append(file)
    
        # make VALIS object
        registrar = registration.Valis(slide_src_dir, res_dir, 
                                       image_type="multi", 
                                       reference_img_f=reference_slide, 
                                       align_to_reference=True,
                                       crop="reference")
        
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        
        # warp images
        for reg_slide in reg_slides:
            slide = registrar.get_slide(reg_slide)
            slide_im = slide.slide2image(level=0)
            warped_im = np.moveaxis(slide.warp_img(img=slide_im, crop="reference"), -1, 0)
    
            # save warped image, with metadata
            if file_prefixes is not None:
                for prefix in file_prefixes:
                    if reg_slide.startswith(prefix):
                        mdata = mdata_dict[file_prefixes[prefix]]
                        break
            else:
                mdata = None
            
            tifffile.imwrite(os.path.join(res_dir, reg_slide), warped_im, metadata=mdata)
        
        del registrar, rigid_registrar, non_rigid_registrar, error_df, slide, slide_im, warped_im

    if kill_jvm:
        registration.kill_jvm()
    else:
        print("NOTE: JVM HAS NOT BEEN KILLED. Make sure to run kill_jvm() at the end of your script.")
    

def run_valis(
    src_dir: str,
    res_dir: str, 
    ref_slide: str=None,
    micro=False,
    kill_jvm=False,
    **kwargs
):    
    """
    End-to-End registration pipeline w/ VALIS
    Reference: https://www.nature.com/articles/s41467-023-40218-9
    """
    # Additional argument settings
    args = {
        # 'img_list': None,                            # Specify for aligning subset of imgs.
        # 'series': 0,                                 # Resolution series # for pyramid formatted imgs
        'align_to_ref': True,                        # Aligning `to` vs. `towards` the ref. image
        'image_type': 'brightfield',                 # Registration image type BF / Fluorescence
        'micro_res': 2000,                           # Resolution for valis micro-registration
        'warped_fname': 'valis_stacked.ome.tif'      # Warped stacked output filename
    }

    for k, v in kwargs.items():
        args[k] = v

    if ref_slide is not None:
        registrar = registration.Valis(src_dir,
                                       res_dir, 
                                       # series=args['series'],
                                       # img_list=args['img_list'],
                                       reference_img_f=ref_slide, 
                                       align_to_reference=args['align_to_ref'], 
                                       imgs_ordered=True,
                                       image_type=args['image_type'])
        
    else:
        registrar = registration.Valis(src_dir, 
                                       res_dir, 
                                       imgs_ordered=True)
        
    rigid_registrar, non_rigid_registrar, _ = registrar.register()

    if micro:
        registrar.register_micro(max_non_rigid_registration_dim_px=args['micro_res'], align_to_reference=True)

    # save results
    save_dir = os.path.join(res_dir, "registered_slides")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    registrar.warp_and_save_slides(save_dir, crop="reference")

    if kill_jvm:
        registration.kill_jvm()
    else:
        print("NOTE: JVM HAS NOT BEEN KILLED. Make sure to run kill_jvm() at the end of your script.")

    # aligned_imgs = [tifffile.imread(os.path.join(save_dir, f))
    #                 for f in sorted(os.listdir(save_dir))
    #                 if f[-8:] == 'ome.tiff']

    # aligned_imgs = np.array(aligned_imgs)
    # aligned_imgs = aligned_imgs.transpose((3,0,1,2))
    # tifffile.imwrite(os.path.join(save_dir, 'valis_stacked.ome.tif'), aligned_imgs, metadata={'axes': 'CZYX'})

    # print("Aligned stacked image saved to:", os.path.join(save_dir, args['warped_fname']))

    return registrar, rigid_registrar, non_rigid_registrar


def kill_jvm():
    registration.kill_jvm()
