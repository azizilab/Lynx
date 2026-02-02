# %%
import os
import gc
import sys
import json
import tifffile

import numpy as np
import scanpy as sc
import pandas as pd
import squidpy as sq
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


%matplotlib inline
%load_ext autoreload
%autoreload 2


# %%
# ----------------------------------------------------------
#  Preprocessing: load Xenium patch & cell-type annotations
# ----------------------------------------------------------

# %%
data_path = '../data/breast/'
sample_id = 'sample1'
adata_xenium = IO.load_xenium(
    os.path.join(data_path, sample_id),
    min_counts=0,
    min_cells=0,
    load_metadata=True
)
adata_xenium.obsm['spatial'].max(0)

# Load scalefactor
with open(os.path.join(data_path, sample_id, 'experiment.xenium'), 'r') as ifile:
    scalefactor = json.load(ifile)['pixel_size'] 

# %%
# Open the TIFF file and check pyramid layers
he_filename = 'Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image_registered.ome.tif'
he_img = tifffile.imread(os.path.join(data_path, sample_id, he_filename)).astype(np.float32)
gc.collect()

# Normalize per channel to [0, 1]
he_img = he_img.astype(np.float32)
for i, chan in enumerate(he_img):
    he_img[i] = (chan - chan.min()) / (chan.max() - chan.min())


# %%
# Attach cell-type annotations
cell_type = pd.read_csv(os.path.join(data_path, sample_id, 'Xenium_annot.csv'), index_col=[0])['Cluster'].to_list()
adata_xenium.obs['cell_type'] = cell_type
adata_xenium.obs['cell_type'] = adata_xenium.obs['cell_type'].astype('category')

# %%
# Save DCIS patch reported in Janesick & Chitra

# Save HE image patch
xmin, xmax = 1500, 3250
ymin, ymax = 2000, 4000
outdir = os.path.join(data_path, 'dcis_fov')
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

# Save HE image patch
he_patch = he_img[:, ymin:ymax, xmin:xmax]
tifffile.imwrite(os.path.join(outdir, 'he.tif'), he_patch)


# Save full-res HE image patch
xmin, xmax = int(1500/scalefactor), int(3250/scalefactor)
ymin, ymax = int(2000/scalefactor), int(4000/scalefactor)
outdir = os.path.join(data_path, 'dcis_fov')
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

he_patch = he_img[:, ymin:ymax, xmin:xmax]
tifffile.imwrite(os.path.join(outdir, 'he_hires.tif'), he_patch)


# Save expression patch
adata_patch = adata_xenium[
    (adata_xenium.obsm['spatial'][:, 0] >= xmin) & (adata_xenium.obsm['spatial'][:, 0] <= xmax) &
    (adata_xenium.obsm['spatial'][:, 1] >= ymin) & (adata_xenium.obsm['spatial'][:, 1] <= ymax)
].copy()
adata_patch.obsm['spatial'] -= np.array([xmin, ymin])
adata_patch.write_h5ad(os.path.join(outdir, 'cell_feature_matrix.h5ad'))

# %%
sq.pl.spatial_scatter(
    adata_patch, 
    color='FASN',
    size=20, img=False, edgecolor='none',
    cmap='Reds',   
)

plt.figure()
plt.imshow(he_patch.transpose(1, 2, 0))
plt.title('Breast Cancer H&E (DCIS patch)', fontsize=15)
plt.show()


# %%
# ----------------------------------------------------
#  Generate paired H&E image patches per Xenium cell
# ----------------------------------------------------

# %%
scalefactor = 0.2125
data_path = '../data/breast/dcis_fov/'
adata_patch = sc.read_h5ad(os.path.join(data_path, 'cell_feature_matrix.h5'))
he_img = tifffile.imread(os.path.join(data_path, 'he.tif')).astype(np.float32)
coords = np.round(adata_patch.obsm['spatial']).astype(np.uint16)

# %%
# from histomicstk.preprocessing.color_normalization import reinhard_color_normalization


# %%
# --- Patch extraction utility ---
def extract_patches(img_raw, coords, P=64):
    """
    img: np.ndarray, shape (3, H, W), values in [0, 1] or [0, 255]
    coords: np.ndarray, shape (N, 2), (x, y)
    w: int, patch size
    Returns: np.ndarray, shape (N, 3*P*P)
    """
    mean = img_raw.mean((1, 2))[:, None, None]
    std = img_raw.std((1, 2))[:, None, None]
    img = (img_raw - mean) / std

    pad = P // 2
    img_padded = np.pad(img, ((0,0), (pad,pad), (pad,pad)), mode='reflect')
    patches = []
    for x, y in coords:
        x, y = int(x), int(y)
        x_pad, y_pad = x + pad, y + pad
        patch = img_padded[:, y_pad-pad:y_pad+pad, x_pad-pad:x_pad+pad]
        patches.append(patch.flatten()) 
    return np.stack(patches)    # Shape: (N, 3*P*P)

he_patches = extract_patches(he_img, coords, P=32)
adata_he = sc.AnnData(
    X=he_patches,
    obs=adata_patch.obs.copy(),
    obsm={'spatial': adata_patch.obsm['spatial'].copy()}
)
# adata_he.write_h5ad(os.path.join(data_path, 'he_patches.h5ad'))
adata_he.write_h5ad(os.path.join(data_path, 'he_patches_norm.h5ad'))

