# %%
# ---------------------------------------------------------
# Cell-type annotations (default segmentation vs. Proseg)
# ---------------------------------------------------------

# %%
import os
import sys
import gc
import zarr
import json
import tifffile

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from IPython.display import display

sys.path.append('..')
from util import IO, utils

# %%
# --------------------
#  Proseg evaluation
# --------------------

import cv2
from skimage.color import label2rgb
from skimage.transform import rescale
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation

# %%
def mask_to_rgb(mask):
    n_lbls = len(np.unique(mask)[1:])
    colors = np.random.random((n_lbls, 3))
    rgb = label2rgb(mask, colors=colors, bg_label=0)
    return rgb


def mask_to_boundary(mask, roi=None):
    np.random.seed(0)
    mask_roi = mask.copy() if roi is None else mask[roi]
    unique_labels = np.unique(mask_roi)[1:]
    mask_roi -= unique_labels[0]
    unique_labels -= unique_labels[0]-1
    boundary_canvas = np.zeros_like(mask_roi, dtype=np.uint8)

    for label in unique_labels:
        rand_int = np.random.randint(1, 256)
        contours = binary_dilation(
            find_boundaries(mask_roi == label).astype(np.uint8),
            footprint=np.ones((3, 3))
        )
        boundary_canvas[contours == 1] = rand_int

    return boundary_canvas


def polygon_to_boundary(img, polygon_df, roi=None):
    np.random.seed(0)
    mask = np.zeros_like(img, dtype=np.uint8)
    unique_labels = np.unique(polygon_df.iloc[:, -1])
    for label in unique_labels:
        rand_int = np.random.randint(1, 256)
        coords = polygon_df.loc[polygon_df.iloc[:, -1] == label].iloc[:, :-1].to_numpy().astype(np.int32)
        cv2.polylines(mask, [coords], isClosed=True, color=rand_int, thickness=3)

    return mask if roi is None else mask[roi]

# %%
# Load XOA (default) segmentation 
data_path = '../data/xenium/NIH_F4/'
proseg_path = '../data/xenium_reseg/NIH_F4/outs/'

# %%
with open(os.path.join(data_path, 'experiment.xenium'), 'r') as ifile:
    scalefactor = json.load(ifile)['pixel_size']

ymin = 6000
xmin = 30000
patch_size = 2048

roi = tuple([slice(ymin, ymin+patch_size), slice(xmin, xmin+patch_size)])
img = tifffile.imread(os.path.join(data_path, 'morphology_focus.ome.tif'))[roi].astype(np.float32)
img = (img-img.min()) / (img.max()-img.min())

# %%
z = zarr.open(os.path.join(data_path, 'cells.zarr.zip'), mode='r')
xoa_mask = z['masks']['1'][:][roi]
xoa_mask = mask_to_boundary(xoa_mask)

del z
gc.collect()

# %%
# Load proseg segmentation
proseg_path = '../data/xenium_reseg/NIH_F4/outs/'
proseg_boundary_df = pd.read_csv(
    os.path.join(proseg_path, 'cell_boundaries.csv.gz'), compression='gzip', index_col=[0]
)
proseg_boundary_df.loc[:, 'vertex_x'] = (proseg_boundary_df.vertex_x / scalefactor).astype(np.uint32)
proseg_boundary_df.loc[:, 'vertex_y'] = (proseg_boundary_df.vertex_y / scalefactor).astype(np.uint32)

# Subset polygons within ROI
isin_roi = np.logical_and(
    (proseg_boundary_df.vertex_x >= xmin) & (proseg_boundary_df.vertex_x < xmin+patch_size).to_numpy(),
    (proseg_boundary_df.vertex_y >= ymin) & (proseg_boundary_df.vertex_y < ymin+patch_size).to_numpy()
)
proseg_boundary_df = proseg_boundary_df.loc[isin_roi]
proseg_boundary_df.iloc[:, 0] = proseg_boundary_df.vertex_x.copy() - xmin
proseg_boundary_df.iloc[:, 1] = proseg_boundary_df.vertex_y.copy() - ymin
# proseg_boundary_df.head()

proseg_mask = polygon_to_boundary(img, proseg_boundary_df)
gc.collect()


# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=500)
ax1.imshow(img, cmap='magma')
ax1.axis('off')
ax1.set_title('Image patch', fontsize=20)

# ax2.imshow(img, cmap='magma')
# ax2.imshow(xoa_mask > 0, alpha=.3)
ax2.imshow(mask_to_rgb(xoa_mask))
ax2.axis('off')
ax2.set_title('XOA (default) segmentation', fontsize=20)

# ax3.imshow(img, cmap='magma')
# ax3.imshow(proseg_mask > 0, alpha=.3)
ax3.imshow(mask_to_rgb(proseg_mask))
ax3.axis('off')
ax3.set_title('Proseg segmentation', fontsize=20)

plt.tight_layout()
plt.show()

# %%
fig.savefig('../sketch/segmentation_boundary.png', bbox_inches='tight', format='png', dpi=300)

# %%
# ----------------------------
#  Cluster-based annotations
# ----------------------------

# %%
import scanpy as sc
import squidpy as sq

xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_M5'

# %%
adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id))
adata_xenium_norm = adata_xenium.copy()

sc.pp.normalize_total(adata_xenium_norm)
sc.pp.log1p(adata_xenium_norm)
sc.pp.pca(adata_xenium_norm)
sc.pp.neighbors(adata_xenium_norm)
sc.tl.umap(adata_xenium_norm)


# %%
sc.pl.umap(
    adata_xenium_norm, 
    color=['CYP3A4', 'CYP2A7', 'ADH1C', 'HAMP'],  # Hepatocytes
    cmap='magma', ncols=2, s=10
)

# Cholangiocytes
sc.pl.umap(
    adata_xenium_norm, 
    color=['KRT7', 'CFTR', 'TM4SF4', 'EHF'],
    cmap='magma', ncols=2, s=10
)

# Fibroblasts
sc.pl.umap(
    adata_xenium_norm, 
    color=['FBN1', 'PDGFRA', 'THY1', 'ASPN'],  # Fibroblasts
    cmap='magma', ncols=2, s=10
)

# Smooth Muscle cells
sc.pl.umap(
    adata_xenium_norm, 
    color=['MYH11', 'ACTA2', 'CNN1', 'RERGL'],
    cmap='magma', ncols=2, s=10
)

# Endothelial
sc.pl.umap(
    adata_xenium_norm,
    color=['SNCG', 'CD34', 'PECAM1'], 
    cmap='magma', ncols=2, s=10
)

# Pan sinusoidal
sc.pl.umap(
    adata_xenium_norm, 
    color=['LYVE1', 'FCGR1A', 'CD14'], 
    cmap='magma', ncols=2, s=10
)

# Kupffer
sc.pl.umap(
    adata_xenium_norm, 
    color=['CD68', 'FCGR1A', 'VSIG4', 'CD86'],
    cmap='magma', ncols=2, s=10
)

# M2
sc.pl.umap(
    adata_xenium_norm, 
    color=['CD163', 'MARCO', 'FCGR1A'], 
    cmap='magma', ncols=2, s=10
)

# B-cell
sc.pl.umap(
    adata_xenium_norm, 
    color=['CD19', 'PTPRC'], 
    cmap='magma', ncols=2, s=10
)

# T-cell,
sc.pl.umap(
    adata_xenium_norm, 
    color=['CD3E', 'CD4', 'CD8A'], 
    cmap='magma', ncols=2, s=10
)

# %%
sc.tl.leiden(adata_xenium_norm, resolution=1.5, flavor='igraph', n_iterations=2)
sc.pl.umap(adata_xenium_norm, color='leiden', s=5)

# %%
sc.pl.umap(
    adata_xenium_norm, color='leiden', 
    groups=[
        '6'
    ],
    s=10
)

(adata_xenium_norm.obs['leiden'] == '12').sum()


# %%
cell_types = [
    'Sinusoidal',
    'Hepatocytes',
    'T-cells',
    'Fibroblasts',
    'Smooth Muscle cells',
    'Endothelial',
    'Kupffer',
    'M2',
    'Cholangiocytes + Progenitor'
]

cell_type_assignments = {
    '0':    'M2',
    '1':    'Hepatocytes',
    '2':    'Hepatocytes',
    '3':    'Hepatocytes',
    '4':    'Hepatocytes',
    '5':    'Hepatocytes',
    '6':    'Fibroblasts',
    '7':    'T-cells',
    '8':    'Kupffer',
    '9':    'Hepatocytes',
    '10':   'Cholangiocytes + Progenitor',
    '11':   'Endothelial',
    '12':   'T-cells',
    '13':   'Sinusoidal',
    '14':   'Smooth Muscle cells',
}


# %%
adata_xenium_norm.obs['cell_type'] = adata_xenium_norm.obs['leiden'].apply(
    lambda x: cell_type_assignments[x]
)
adata_xenium_norm.obs['cell_type'] = adata_xenium_norm.obs['cell_type'].astype('category')

sc.pl.umap(adata_xenium_norm, color='cell_type')
adata_xenium_norm.obs['cell_type'].value_counts()

# %%
# Save annotations w/ raw-count matrix
adata_xenium.obs['cell_type'] = adata_xenium_norm.obs['cell_type'].values.copy()
# adata_xenium.obs['leiden'] = adata_xenium_norm.obs['leiden'].values.copy()
adata_xenium.write_h5ad(os.path.join(xenium_path, sample_id, 'cell_feature_matrix.h5'))

# %%
