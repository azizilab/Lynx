import os
import cv2
import numpy as np

from skimage.filters import gaussian as gaussian_blur
from skimage.exposure import equalize_adapthist
from valis import registration

from __init__ import LOGGER


def get_affine_matrix(
    source: np.ndarray,
    target: np.ndarray,
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
                
    pts1, pts2 = [], []
    for m in good_matches:
        pt1, pt2 = pts_src[m.queryIdx].pt, pts_dst[m.trainIdx].pt
        if pt1 not in pts1 and pt2 not in pts2:
            pts1.append(pt1)
            pts2.append(pt2)
    pts1 = np.float32(pts1).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)

    if sort_matches:
        pts1, pts2 = _reorder_points(pts1, pts2)
        pts1, pts2 = pts1[:5], pts2[:5]
        LOGGER.warning('Re-calculated pts w/ larger search space')

    # Only allow translation, scaling & rotation
    M, _ = cv2.estimateAffinePartial2D(pts1, pts2, cv2.RANSAC) 
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


def run_valis(
    src_dir: str,
    res_dir: str, 
    ref_slide: str=None,
    micro=False,
    kill_jvm=False,
    **kwargs
):    
    """
    Registration w/ VALIS
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
