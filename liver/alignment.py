# %%
# ---------------------------------------------------------------------
# Compute pairwise mapping across modalities w/ different resolutions
# --------------------------------------------------------------------
# e.g. Xenium (cells) - DESI (pixels)

# %%
import os
import sys
import gc
import json
import tifffile

import numpy as np
import scanpy as sc
import squidpy as sq
import spatialdata as sd
import matplotlib.pyplot as plt

from skimage import morphology
from skimage.transform import rescale

sys.path.append('..')
from util import IO, utils, registration

# %%
%reload_ext autoreload
%autoreload 2


# %%
from matplotlib.gridspec import GridSpec
from IPython.display import display, clear_output
from ipywidgets import Button

def interactive_anchor_annots(
    src_img, dst_img, 
    src_channel=None,
    dst_channel=None,
    inverse_src=False,
    inverse_dst=False,
    sample_id=None
):
    """
    Annotate anchor points on "source" (moving) & "destination" (target) image
    """
    plt.close('all')

    src_points = []
    dst_points = []

    if src_img.ndim == 3:
        src_img = src_img.mean(0) if src_channel is None else src_img[max(0, src_channel)]
    if inverse_src:
        src_img = 1 - src_img

    if dst_img.ndim == 3:
        dst_img = dst_img.mean(0) if dst_channel is None else dst_img[max(0, dst_channel)]
    if inverse_dst:
        dst_img = 1 - dst_img

    def onclick_src(event):
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            src_points.append((ix, iy))
            print(f"Clicked source pixel: x={ix:.2f}, y={iy:.2f}")

    def onclick_dst(event):
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            dst_points.append((ix, iy))
            print(f"Clicked destination pixel: x={ix:.2f}, y={iy:.2f}")

    fig = plt.figure(figsize=(3, 3))
    gs = GridSpec(1, 2, width_ratios=[1, 1])  

    # Display the first image
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(src_img)
    ax0.set_title('Source Image ' + sample_id)
    ax0.axis('off')

    # Display the second image
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(dst_img)
    ax1.set_title('Target Image ' + sample_id)
    ax1.axis('off') 

    plt.tight_layout()
    plt.show()
    
    ax0.figure.canvas.mpl_connect('button_press_event', lambda event: onclick_src(event) if event.inaxes == ax0 else None)
    ax1.figure.canvas.mpl_connect('button_press_event', lambda event: onclick_dst(event) if event.inaxes == ax1 else None)

    return src_points, dst_points


# %%
# Load sample DESI image & preprocess boundary
def load_desi_img(
    filename, 
    sigma=5,
    erode_pixel=3,
    min_area=500,
):
    img = utils.norm_by_channel(tifffile.imread(filename))
    roi_mask = utils.get_roi_mask(img, sigma=sigma, min_area=min_area, erode_pixel=erode_pixel)
    return img, roi_mask

def load_xenium_img(dir, sample_id, scale=0.1):
    adata = IO.load_xenium(os.path.join(dir, sample_id), load_metadata=True, load_img=True)
    img = np.squeeze(adata.uns['spatial'][sample_id]['images']['hires'])
    img = rescale(img, scale=scale).astype(np.float32)
    img = (img-img.min()) / (img.max()-img.min())

    with open(os.path.join(dir, sample_id, 'experiment.xenium'), 'r') as ifile:
        scalefactor = json.load(ifile)['pixel_size'] 
    if 'x_centroid' in adata.obs.columns and 'y_centroid' in adata.obs.columns:
        coords = adata.obs[['x_centroid', 'y_centroid']] / scalefactor
        coords = np.round(coords*scale).astype(np.int16).values
    else:
        coords = adata.obsm['spatial'] / scalefactor
        coords = np.round(coords*scale).astype(np.int16)
    return img, coords, scalefactor


# %%
# TODO: sample run on proseg output
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
# sample_ids = sorted([
#     f for f in os.listdir(xenium_path) 
#     if os.path.isdir(os.path.join(xenium_path, f))
# ])

sample_ids = ['NIH_F5']


# %%
%matplotlib widget
#-----------------------------
#  Alignment across modality                
#-----------------------------
desi_anchor_points = []
xenium_anchor_points = []

# %%
sample_id = sample_ids[0]  # Iterate through all `sample_ids`
print('Loading {}...'.format(sample_id))
desi_img, mask = load_desi_img(os.path.join(desi_path, sample_id+'.ome.tif'))
xenium_img, coords = load_xenium_img(xenium_path, sample_id)

print('Aligning images...')
desi_points, xenium_points = interactive_anchor_annots(
    src_img=desi_img,
    dst_img=xenium_img,
    src_channel=1,
    inverse_dst=True,
    sample_id=sample_id
)

# %%
print('Appending anchors of {}...'.format(sample_id))
desi_anchor_points.append(desi_points)
xenium_anchor_points.append(xenium_points)

# Save anchors
anchor_dir = '../data/integrated/anchors'
for i, sample_id in enumerate(sample_ids):
    np.save(os.path.join(anchor_dir, 'DESI_'+sample_id), desi_anchor_points[i])
    np.save(os.path.join(anchor_dir, 'Xenium_'+sample_id), xenium_anchor_points[i])


# %%
%matplotlib inline

# %%
# ---------------------------------------------
# Compute cross-domain projections 
# with Rigid alignment (source <==> target)
# ---------------------------------------------
# To get mappable Xenium cell -> DESI pixel
# we need to set src (Xenium) & dst (DESI)

# Load anchors
anchor_dir = '../data/integrated/anchors/'
desi_anchor_points = []
xenium_anchor_points = []

for sample_id in sample_ids:
    desi_anchors = np.load(os.path.join(anchor_dir, 'DESI_'+sample_id+'.npy'))
    xenium_anchors = np.load(os.path.join(anchor_dir, 'Xenium_'+sample_id+'.npy'))

    desi_anchor_points.append(desi_anchors)
    xenium_anchor_points.append(xenium_anchors)

del desi_anchors, xenium_anchors


# %%
# Compute mapped coordinates
scale = 0.1
channel = 2  # selected DESI channel
xenium_mapped_coords = []  # Xenium coords onto DESI space
desi_mapped_coords = []  # DESI coords onto Xenium space

for i, sample_id in enumerate(sample_ids):
    print('Coordinate mapping - Xenium -> DESI for {}'.format(sample_id))
    print('\t Loading images...')
    desi_img, mask = load_desi_img(os.path.join(desi_path, sample_id+'.ome.tif'), erode_pixel=1)
    desi_coords = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5')).obsm['spatial']
    xenium_img, _, scalefactor = load_xenium_img(xenium_path, sample_id, scale=scale)
    xenium_coords = sd.read_zarr(os.path.join(xenium_path, 'NIH_F5_proseg', 'output_annotated.zarr'))['table'].obsm['spatial'] / scalefactor
    xenium_coords = np.round(xenium_coords*scale).astype(np.int16)

    # DESI foreground pixels (upon erosion)
    print('\t Getting affine tranform matrix...')
    roi_pixels = set(
        tuple(coord)
        for coord in np.array(np.nonzero(mask)).T[:, [1, 0]]
    )

    M = registration.get_affine_matrix(
        source=xenium_img,
        target=desi_img[channel],
        pts_source=xenium_anchor_points[i],
        pts_target=desi_anchor_points[i]
    )   

    xenium_warped = registration.affine_warp(
        img_src=xenium_img,
        dst_shape=desi_img[channel].shape,
        M=M
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes[0, 0].imshow(desi_img[channel], cmap='magma')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('DESI', fontsize=30)

    axes[0, 1].imshow(xenium_warped, cmap='magma')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Xenium Warped', fontsize=30)
 
    axes[1, 0].imshow(desi_img.mean(0), cmap='magma')
    axes[1, 0].imshow(mask, alpha=.3)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('DESI ROI', fontsize=30)
 
    axes[1, 1].imshow(desi_img[channel], cmap='magma', alpha=.5)
    axes[1, 1].imshow(xenium_warped, cmap='magma', alpha=.5)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Overlapped', fontsize=30)
    
    plt.tight_layout()
    plt.show()

    print('\t Mapping coords...')
    # Xenium -> DESI
    proj_coords = registration.affine_transform_coords(M=M, coords=xenium_coords)
    xenium_to_desi_coords = np.array([
        coord if tuple(coord) in roi_pixels else [-1, -1]
        for coord in proj_coords 
    ])

    # DESI -> Xenium
    desi_to_xenium_coords = registration.inverse_affine_transform_coords(M=M, coords=desi_coords) / scale * scalefactor

    xenium_mapped_coords.append(xenium_to_desi_coords)
    desi_mapped_coords.append(desi_to_xenium_coords)

    # Count mapping percentage
    offmap_count = 0
    for coord in xenium_to_desi_coords:
        if np.array_equal(coord, [-1, -1]):
            offmap_count += 1
    map_count = len(xenium_to_desi_coords)-offmap_count
    print('\t{0}/{1} ({2}%) mapped to DESI'.format(
        map_count, len(xenium_to_desi_coords), np.round((map_count/len(xenium_to_desi_coords))*100, 2)
    ))
    del coord, map_count, offmap_count

    print('=====================\n\n')
    del desi_img, xenium_img, roi_pixels, mask, M
    del proj_coords, xenium_to_desi_coords, desi_to_xenium_coords
    gc.collect()


# %%
# Project coordinates to each other's space
for sample_id, xenium_to_desi_coords, desi_to_xenium_coords in zip(sample_ids, xenium_mapped_coords, desi_mapped_coords):
    print("Saving cell <-> pixel mapping of {}...".format(sample_id))

    # adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), raw_count=True, load_metadata=True)
    adata_xenium = sd.read_zarr(os.path.join(xenium_path, 'NIH_F5_proseg', 'output_annotated.zarr'))['table']
    adata_xenium.obsm['desi_map'] = xenium_to_desi_coords
    cells_to_keep = adata_xenium.obs_names[
        np.logical_not((adata_xenium.obsm['desi_map'] == -1).any(1))
    ]

    # # Save anndatas w/ aligned coords
    # adata_xenium.write_h5ad(os.path.join(xenium_path, sample_id, 'cell_feature_matrix.h5'))
    # adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
    # adata_desi.obsm['xenium_map'] = desi_to_xenium_coords
    # adata_desi.write_h5ad(os.path.join(desi_path, sample_id+'_proseg.h5'))


