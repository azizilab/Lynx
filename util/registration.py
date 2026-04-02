import os
import sys
import cv2
import tifffile
import numpy as np
import pandas as pd
import anndata as ad

from typing import List, Dict
from scipy.ndimage import map_coordinates
from skimage.filters import gaussian as gaussian_blur
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2gray
from valis import registration
from valis.non_rigid_registrars import OpticalFlowWarper
from valis.warp_tools import warp_img

import geopandas as gpd
import spatialdata as sd
import spatialdata_io as sdio
import spatialdata_plot as sdpl
from napari_spatialdata import Interactive

from spatialdata import SpatialData
from spatialdata.transformations import (
    BaseTransformation,
    Sequence,
    get_transformation,
    set_transformation,
)
from shapely.geometry import Point
from spatialdata.models import ShapesModel, TableModel, Image2DModel
from spatialdata.transformations import remove_transformation
from spatialdata.transformations import align_elements_using_landmarks, get_transformation


sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from __init__ import LOGGER

# Helper functions for static image registration (tiff images)

def get_affine_matrix(
    source: np.ndarray, 
    target: np.ndarray,
    pts_source: List[tuple] = None,  
    pts_target: List[tuple] = None,
    sigma: float = 5,
    n_matches: int = 50
) -> np.ndarray:
    r"""
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
        LOGGER.info('Affine Registration w/ pre-defined anchor points...')
        assert len(pts_source) == len(pts_target), \
            "Anchor pts btw source & target should have equal number"
        pts_source = np.float32(pts_source).reshape(-1, 1, 2)
        pts_target = np.float32(pts_target).reshape(-1, 1, 2)

    else:  
        LOGGER.info('Affine Registration w/ SIFT..')
        sift = cv2.SIFT_create()
        pts_src, des_src = sift.detectAndCompute(img_src, None)
        pts_dst, des_dst = sift.detectAndCompute(img_dst, None)
        
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des_src, des_dst, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.9*n.distance:
                good_matches.append(m)

        # If insufficient anchor points (likely causing misalignment)
        # expand the searcing space, & choose the top 3 anchors
        # (min. requirement for computing affine transformation)
        sort_matches = False
        if len(good_matches) < n_matches:
            LOGGER.warning(
                'Insufficient # anchor points from SIFT,'
                'choose the top-aligned anchors to avoid misalignment'
            )
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
    dst_shape: np.ndarray,
    M: np.ndarray = np.array([[1,0,0], [0,1,0]], dtype=np.float32)
) -> np.ndarray:
    """Compute Warped image given precomputed transformation matrix"""
    img_warped = cv2.warpAffine(img_src, M, (dst_shape[1], dst_shape[0])) 
    img_warped = (img_warped-img_warped.min()) / (img_warped.max()-img_warped.min())
    return img_warped
        

def affine_transform_coords(
    M: np.ndarray,
    coords: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    r"""Transform coordinates from `source` image to `destination` image"""
    # Convert coordinates to a format required by cv2.transform
    src_coords = np.array(coords, dtype=np.float32).reshape(-1, 1, 2)
    transformed_coords = cv2.transform(src_coords, M)
    dst_coords = np.array([coord[0] for coord in transformed_coords])
    return dst_coords


def inverse_affine_transform_coords(
    M: np.ndarray,
    coords: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    r"""Transform xy-coordinates from `destination` image back to `source` image"""
    # Convert M (2x3) to a 3x3 matrix by appending [0, 0, 1]
    M_inv = np.vstack([M, [0, 0, 1]])  # Convert to 3x3
    M_inv = np.linalg.inv(M_inv)[:2]  # Invert and take first two rows

    # Convert coordinates to the format required by cv2.transform
    dst_coords = np.array(coords, dtype=np.float32).reshape(-1, 1, 2)
    transformed_coords = cv2.transform(dst_coords, M_inv)

    src_coords = transformed_coords.reshape(-1, 2)  # (N, 1, 2) -> (N, 2)
    return src_coords


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


def nonrigid_warp(
    source: np.ndarray,
    target: np.ndarray,
    bk_dxdy: np.ndarray = None
):
    r"""Non-rigid Alignmeng / Warping w/ Optical Flow backbone"""
    assert source.ndim <= 3, \
        "Only support 2D / 3D images"    
    shape = target.shape[:2]

    if bk_dxdy is None:
        assert source.ndim == target.ndim, "Source and target ndim must be equal"
        
        img_src = source.copy()
        img_dst = target.copy()
        
        if source.ndim == 3:
            img_src = rgb2gray(img_src)
            img_dst = rgb2gray(img_dst)
    
        # Calculate optical flow & displacement (2, Y, X)
        registrar = OpticalFlowWarper(n_grid_pts=100, sigma_ratio=.5, smoothing_method="gauss")
        bk_dxdy = registrar.calc(moving_img=img_src, fixed_img=img_dst)

    # Warping original images
    if source.ndim == 3: # Warping RGB image, dim: (Y, X, C)
        img_warped = np.zeros_like(source, dtype=np.uint8)
        for i, chan in enumerate(source.transpose(2,0,1)):  
            chan_warped = warp_img(chan.astype(np.float32)/255.0, bk_dxdy=bk_dxdy, out_shape_rc=shape)
            img_warped[:,:,i] = np.round(chan_warped*255).astype(np.uint8)

    else:  # Warping grayscale image, dim: (Y, X)
        img_warped = warp_img(source, bk_dxdy=bk_dxdy, out_shape_rc=shape)
        img_warped = (img_warped-img_warped.min()) / (img_warped.max()-img_warped.min())

    return img_warped, bk_dxdy


def nonrigid_transform_coords(
    coords: list[tuple[float, float]],
    bk_dxdy: np.ndarray,
):
    r"""Transform xy-coordinates based on non-rigid displacement field"""
    x, y = coords[:, 0], coords[:, 1]  

    # Interpolate dx and dy at subpixel positions
    dx = map_coordinates(bk_dxdy[0], [x, y], mode='nearest')
    dy = map_coordinates(bk_dxdy[1], [x, y], mode='nearest')

    x_warped = x - dx
    y_warped = y - dy

    return np.vstack([x_warped, y_warped]).T 


# Helper functions for spatialdata-napari registration

def delete_coordinate_system(sdata, cs_name):
    """Removes a coordinate system by deleting associated transformations from all elements."""
    for element_type, element_name, element in sdata._gen_elements():
        try:
            remove_transformation(element, cs_name)
        except KeyError:
            # This element was not linked to the coordinate system
            pass


def transform_coords(sdata, coords, image_key, target_coordinate_system="aligned"):
    """
    Apply affine transformation matrix M to input coordinates
    """
    transform = get_transformation(
        sdata[image_key],
        to_coordinate_system=target_coordinate_system
    ).transformations[-1]
    M = transform.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
    M = M[:-1]
    transformed_coords = registration.affine_transform_coords(M, coords)
    return transformed_coords


def postpone_transformation(
    sdata: SpatialData,
    transformation: BaseTransformation,
    source_coordinate_system: str,
    target_coordinate_system: str,
):
    """Align all elements in `sdata` by appending `transformation` to existing transformations"""
    for element_type, element_name, element in sdata._gen_elements():
        old_transformations = get_transformation(element, get_all=True)
        if source_coordinate_system in old_transformations:
            old_transformation = old_transformations[source_coordinate_system]
            sequence = Sequence([old_transformation, transformation])
            set_transformation(element, sequence, target_coordinate_system)


def update_adata_coords(
    sdata: sd.SpatialData, 
    matrix: np.ndarray,
    obsm_key: str = "spatial", 
    pre_scalefactor: float = 0.2125,
    post_scalefactor: float = 0.2125,
    size_limit: tuple = None,
):
    """
    Updates `sdata.table.obsm[obsm_key]` to match a specific coordinate system 
    defined on the associated spatial element.
    """
    moving_coords = np.array(
        sdata.table.obsm[obsm_key] / pre_scalefactor, dtype=np.float32
    ).reshape(-1, 1, 2)

    transformed_coords = cv2.transform(moving_coords, matrix[:-1])
    transformed_coords =  np.array([
        list(map(int, np.round(coord[0]))) 
        for coord in transformed_coords
    ]) * post_scalefactor

    if size_limit is not None and len(size_limit) == 2:
        xlim, ylim = size_limit
        coord_to_keep = (
            (transformed_coords[:, 0] >= 0) & (transformed_coords[:, 0] < xlim) & 
            (transformed_coords[:, 1] >= 0) & (transformed_coords[:, 1] < ylim)
        )
        transformed_coords = transformed_coords[coord_to_keep]
        adata_to_keep = sdata.table[coord_to_keep].copy()
        print(adata_to_keep)
        del sdata.table
        sdata.table = adata_to_keep
    
    sdata.table.obsm[obsm_key] = transformed_coords

    return None


def compile_spatialdata(
    adata: ad.AnnData,
    img: np.ndarray,
    image_key: str,
):
    """Compile AnnData and image into SpatialData object"""
    img_obj = Image2DModel.parse(img, scale_factors=(2, 2, 2))
    gdf = gpd.GeoDataFrame(
        pd.DataFrame([1]*adata.n_obs, columns=['radius']),
        geometry=[Point(x, y) for x, y in adata.obsm['spatial']],
    )
    shape_obj = ShapesModel.parse(gdf)

    adata_obj = TableModel.parse(adata)
    adata_obj.uns["spatialdata_attrs"] = {
        "region": "spots",  
        "region_key": "region",  
        "instance_key": "spot_id", 
    }
    adata_obj.obs["region"] = pd.Categorical(["spots"] * len(adata_obj))
    adata_obj.obs["spot_id"] = shape_obj.index

    sdata = sd.SpatialData(
        images={image_key: img_obj},
        shapes={"spots": shape_obj},
        table={"table": adata_obj}
    )

    return sdata

