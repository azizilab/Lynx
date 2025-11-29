# Infer spatial gradients on Xenium triple-positive breast cancer 
# Histology + Xenium

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

import pyro
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import rcParams

sns.set_context('paper')
rcParams.update({'font.family': 'Arial'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')

import IO, plot, utils, trajectory
import vgae, configs, dataset

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
# Save DCIS patch suggested by Janesick & Chitra

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



# %%
# ---------------
#   LYNX runs
# ---------------

# %%
# Dataset specs
n_subgraphs = 8
k = 8
r = 50

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500
lr = 1e-3
patience = 500

data_path = '../data/breast/dcis_fov/'
outdir = '../figures/'
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'cell_feature_matrix.h5'))
adata_he = sc.read_h5ad(os.path.join(data_path, 'he_patches_norm.h5ad'))
cluster_key = 'cell_type'

# Filter out 'Unlabeled' cells & cells with extremely rare cell-types (DCIS2)
rare_labels = adata_xenium.obs[cluster_key].value_counts()[
    adata_xenium.obs[cluster_key].value_counts() < 10
].index.to_list()

labeled_mask = np.logical_and(
    adata_xenium.obs[cluster_key] != 'Unlabeled',
    ~adata_xenium.obs[cluster_key].isin(rare_labels)
)
adata_xenium = adata_xenium[labeled_mask].copy()
adata_xenium.obs.index = adata_xenium.obs.index.astype(int)
adata_he = adata_he[labeled_mask].copy()
patch_size = np.sqrt(adata_he.var.shape[0] // 3).astype(int)

del rare_labels, labeled_mask
gc.collect()

# Model setup
graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_he,
    n_subgraphs=n_subgraphs, 
    k=k, r=r, is_weighted=True,
    cluster_key=cluster_key,
    alpha=1.0,

    # Update modality labels
    query='HE', query_proj_key='spatial',
    ref='Xenium', ref_proj_key='spatial' 
)

train_data, val_data = random_split(graph_data, [0.7, 0.3])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

# Training & Inference
train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience, 
    device=torch.device('cuda'),
)

model_configs = configs.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden, 
    c_latent=n_latent,
    patch_size=patch_size,
    act=nn.SiLU(),
    infer_cell_interaction=True
) 

# %%
pyro.clear_param_store()
torch.cuda.empty_cache()

model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(graph_data, train_configs, DEBUG=True)
res = model.evaluate(
    adata_xenium, adata_he,
    graph_data=graph_data,
    device=torch.device('cpu'),
)

# %%
# Evaluation
plot.disp_kde_scatter(
    adata_xenium.X.A.flatten().copy(),
    res.px.flatten().copy(),
    xlabel=r"Observation $log(x+1)$",
    ylabel=r"Reconstruction $log(x+1)$",
    title='Feature reconstruction (human breast cancer)'
)
gc.collect()

# %%
# Save LYNX inference results
outdir = '../results/breast/'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
np.save(os.path.join(outdir, 'LYNX_latent_6.npy'), adata_xenium.obsm['X_z'])
np.save(os.path.join(outdir, 'LYNX_pseudotime_6.npy'), adata_xenium.obs['t'])
adata_xenium.write_h5ad(os.path.join(outdir, 'LYNX_xenium.h5ad'))

# %%
