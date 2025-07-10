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
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from skimage.transform import rescale
from sklearn.neighbors import NearestNeighbors

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import rcParams

sns.set_context('paper')
rcParams.update({'font.family': 'Liberation Sans'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# %%
sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')

import IO, plot, utils, trajectory
import vgae, configs, dataset
from importlib import reload

# %%
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

he_img = rescale(he_img, scalefactor, preserve_range=True, channel_axis=0)
gc.collect()


# %%
# Attach cell-type annotations
cell_type = pd.read_csv(os.path.join(data_path, sample_id, 'Xenium_annot.csv'), index_col=[0])['Cluster'].to_list()
adata_xenium.obs['cell_type'] = cell_type
adata_xenium.obs['cell_type'] = adata_xenium.obs['cell_type'].astype('category')

# %%
# Save DCIS patch suggested by Janesick & Chitra
xmin, xmax = 1500, 3250
ymin, ymax = 2000, 4000
outdir = os.path.join(data_path, 'dcis_fov')
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

# Save HE image patch
# he_patch = he_img[:, ymin:ymax, xmin:xmax]
# tifffile.imwrite(os.path.join(outdir, 'he.tif'), he_patch)

# Save expression patch
adata_patch = adata_xenium[
    (adata_xenium.obsm['spatial'][:, 0] >= xmin) & (adata_xenium.obsm['spatial'][:, 0] <= xmax) &
    (adata_xenium.obsm['spatial'][:, 1] >= ymin) & (adata_xenium.obsm['spatial'][:, 1] <= ymax)
].copy()
adata_patch.obsm['spatial'] -= np.array([xmin, ymin])
adata_patch.write_h5ad(os.path.join(outdir, 'cell_feature_matrix.h5ad'))

# %%
# sq.pl.spatial_scatter(
#     adata_patch, 
#     color='FASN',
#     size=20, img=False, edgecolor='none',
#     cmap='Reds',   
# )

# plt.figure()
# plt.imshow(he_patch.transpose(1, 2, 0))
# plt.title('Breast Cancer H&E (DCIS patch)', fontsize=15)
# plt.show()

# %%
# ----------------------------------------------------------
# Generate flattened H&E adata by taking mini-patch around cell centroids
# (1). Simplest approach: taking only centroids??
data_path = '../data/breast/dcis_fov/'
adata_patch = sc.read_h5ad(os.path.join(data_path, 'cell_feature_matrix.h5ad'))
he_patch = tifffile.imread(os.path.join(data_path, 'he.tif')).astype(np.float32)
coords = np.round(adata_patch.obsm['spatial']).astype(np.uint8)

# %%
he_intensity = np.zeros((len(coords), 3), dtype=np.float32)

# %%
# (2). Take patch avg
# %%
# If we take smoothing patch, we need to take `r` < nearest neighbor distance
sq.gr.spatial_neighbors(adata_patch, coord_type='generic', n_neighs=10)

def avg_knn_distances(adata, n=10):
    """
    Compute the average distance from each point to its n nearest neighbors.

    Parameters:
        adata: AnnData object with spatial coordinates in adata.obsm['spatial']
        n: number of nearest neighbors

    Returns:
        avg_dists: numpy array of average distances for each point
    """
    coords = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=n+1).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    # Exclude the first column (distance to self = 0)
    avg_dists = dists[:, 1:n+1].mean(axis=1)
    return avg_dists

avg_dists = avg_knn_distances(adata_patch, n=20)
print("Mean average kNN distance:", avg_dists.mean())

# %%
def extract_mean_rgb(he_patch, coords, spot_size=1):
    """
    Extract mean RGB values from he_patch around each coordinate.

    Parameters
    ----------
        he_patch : np.ndarray, shape (3, H, W)
        coords : np.ndarray, shape (n, 2), integer coordinates (y, x)
        spot_size : int, must be odd, size of the square patch

    Returns
    -------
        he_intensity : np.ndarray, shape (n, 3)
    """
    assert spot_size % 2 == 1, "spot_size must be odd"
    pad = spot_size // 2
    n = coords.shape[0]
    H, W = he_patch.shape[1:]
    # Pad image to handle border cases
    he_patch_padded = np.pad(he_patch, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
    # Shift coords for padding
    coords_pad = coords + pad

    # Prepare indices for patch extraction
    y_idx = coords_pad[:, 0][:, None] + np.arange(-pad, pad+1)
    x_idx = coords_pad[:, 1][:, None] + np.arange(-pad, pad+1)
    # Broadcast to get all combinations
    y_idx = y_idx[:, :, None]
    x_idx = x_idx[:, None, :]

    # Gather patches for all channels
    he_intensity = np.zeros((n, 3), dtype=np.float32)
    for c in range(3):
        # Extract patches: shape (n, spot_size, spot_size)
        patches = he_patch_padded[c][y_idx, x_idx]
        # Mean over patch
        he_intensity[:, c] = patches.mean(axis=(1, 2))
    return he_intensity

# Example usage:
# he_intensity = extract_mean_rgb(he_patch, coords, spot_size=3)
he_intensity = extract_mean_rgb(he_patch, coords, spot_size=3)

# %%
adata_he = sc.AnnData(
    X=he_intensity, 
    obs=adata_patch.obs.copy(),
    obsm={'spatial': adata_patch.obsm['spatial'].copy()}
)
adata_he.var_names = ['R', 'G', 'B']

# %%
adata_he.write_h5ad(os.path.join(data_path, 'he.h5ad'))

# %%
# adata_he = sc.read_h5ad(os.path.join(data_path, 'he.h5ad'))
sc.pl.embedding(
    adata_he,  
    basis='spatial',
    color='G',
    size=20, 
    cmap='magma'
)

# %%
sc.pp.pca(adata_he)
sc.pl.pca(adata_he, color='R', cmap='RdBu_r')
# Maybe the best, best solution is to take out the full patch of the exact segmentation!

# %%
# ----------------------------------------------------------

# %%
%reload_ext autoreload

# %%
# Dataset specs
n_subgraphs = 8
k = 20
r = 25

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500
lr = 1e-3
patience = 50

# Real data
data_path = '../data/breast/dcis_fov/'
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'cell_feature_matrix.h5'))
adata_he = sc.read_h5ad(os.path.join(data_path, 'he.h5'))

# Preprocess, add cell-type labels in integers
if 'cell_type' in adata_xenium.obs.keys():
    adata_xenium.obs['leiden'] = adata_xenium.obs.cell_type.factorize()[0]
else:
    adata_norm = adata_xenium.copy()
    sc.pp.normalize_total(adata_norm)
    sc.pp.log1p(adata_norm)

    sc.pp.pca(adata_norm)
    sc.pp.neighbors(adata_norm)
    sc.tl.leiden(adata_norm, random_state=42)

    adata_xenium.obs['leiden'] = adata_norm.obs['leiden'].copy()
    del adata_norm   

# %%
graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_he,
    n_subgraphs=n_subgraphs, 
    k=k, r=r,is_weighted=True,
    ref='Xenium', query='HE', ref_proj_key='spatial', query_proj_key='spatial'
)

train_data, val_data = random_split(graph_data, [0.8, 0.2])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

# Training & Inference
train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience, 
    device=torch.device('cuda'),
    anneal=True,
    verbose=True
)

model_configs = configs.set_model_configs(
    c_in=adata_xenium.shape[1],   # ref-dim 
    c_aux=adata_he.shape[1],  # query-dim
    c_hidden=n_hidden, 
    c_latent=n_latent,
    act=nn.SiLU(),
    ref=graph_data.ref, 
    query=graph_data.query,
    k_hop=1,
    num_heads=1,
    num_clusters=graph_data.num_clusters,
    verbose=True
) 

# %%
# del model
pyro.clear_param_store()
torch.cuda.empty_cache()
reload(vgae)

# %%
model = vgae.HeteroVGAE(model_configs, device=torch.device('cuda'))
model.fit(train_configs, train_dl=train_dl, val_dl=val_dl, DEBUG=True)
res = model.evaluate(
    adata_xenium, adata_he,
    graph_data=graph_data,
    device=torch.device('cpu')
)

# %%
# Evaluation
# (1). Reconstruction
rand_indices = np.random.choice(
    np.arange(adata_xenium.shape[0]*adata_xenium.shape[1]), 10000, replace=False
)
plot.disp_kde_scatter(
    adata_xenium.X.A.flatten()[rand_indices],
    res.px.flatten()[rand_indices],
    xlabel=r"Ground-truth observation",
    ylabel=r"Reconstructed observation",
    title='Xenium feature reconstruction'
)
del rand_indices
gc.collect()

# %%
gc.collect()

# %%
adata_xenium.obsm['X_z'] = res.qzx
sc.pp.neighbors(adata_xenium, use_rep='X_z')
sc.tl.umap(adata_xenium)
sc.pl.umap(adata_xenium, color='cell_type')


# %%
sq.pl.spatial_scatter(
    adata_xenium, 
    color='CD8A',
    size=20, img=False, edgecolor='none', cmap='Reds'
)

# %%
# (2). Trajectory Inference
# High-dim gradients
adata_xenium.obsm['X_z'] = res.qzx
trajectory.compute_trajectory(
    adata_xenium, 
    use_rep='X_z',
    ppt_lambda=1000,
    ppt_sigma=.1,
    ppt_niter=200,
    # root_marker='FASN',  # known invasive marker
    root_marker='CD8A',  # known immune marker
)

sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu_r', size=20, img=False,
    title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (Xenium)'
)

# # Low-dim gradients
# adata_he.obsm['X_z'] = res.qzu
# trajectory.compute_trajectory(
#     adata_he, 
#     use_rep='X_z',
#     # dist_metric='knn',
#     # root_marker='Taurine '
# )

# sq.pl.spatial_scatter(
#     adata_he, color='t', 
#     cmap='RdBu_r', size=1, img=False,
#     title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (H&E)'
# )

# %%
plot.disp_trajectory(
    adata_xenium, 
    cmap='RdBu_r',
    title='Spatial Gradients\n LYNX (Xenium)'
)

# %%
# Ablation: what's the manifold directly applying SimplePPT?
import scFates as scf
scf.tl.tree(
    adata_xenium,
    use_rep='X_z',
    Nodes=int(adata_xenium.shape[0] * 0.1),
    ppt_lambda=1e6,
    ppt_sigma=.1,
    ppt_nsteps=200,
    seed=42,
    device=torch.device('cuda')
)

# %%
scf.pl.graph(adata_xenium)


# %%
# Validation: whether we captured the DCIS - Invasive trajectory well!
sc.pl.embedding(
    adata_xenium,
    basis='spatial',
    color='cell_type',
    groups=['Invasive_Tumor', 'DCIS_1'],
    size=20, frameon=False
)


# %%
# Validation: whether baseline UMAP tells us such trajectory?
adata_norm = adata_xenium.copy()
sc.pp.normalize_total(adata_norm)
sc.pp.log1p(adata_norm)
sc.pp.pca(adata_norm)
sc.pp.neighbors(adata_norm)
sc.tl.umap(adata_norm)

# %%
sc.pl.umap(adata_norm, 
    color='cell_type', 
    groups=['Invasive_Tumor', 'DCIS_1'],
    size=20, frameon=False,
    title='Baseline UMAP (Xenium)'
)

# %%
sc.pl.pca(adata_norm, 
    color='cell_type', 
    groups=['Invasive_Tumor', 'DCIS_1'],
    size=20, frameon=False,
    title='Baseline PCA (Xenium)'
)

# %%
sc.tl.diffmap(adata_norm)

# %%
sc.pl.embedding(
    adata_norm, 
    basis='diffmap',
    color='cell_type',
    groups=['Invasive_Tumor', 'DCIS_1'],
    size=20, frameon=False,
    title='Baseline Diffusion Map (Xenium)'
)


# %%
plot.disp_trajectory(
    adata_xenium, 
    cmap='RdBu_r',
    title='Spatial Gradients\n LYNX (Xenium)'
)

plot.disp_trajectory(
    adata_he, 
    cmap='RdBu_r',
    title='Spatial Gradients\n LYNX (DESI)'
)

# %%
# Visualize latent (z) & spatial clustering
plot.disp_factor_corr(res.qzx)
plot.disp_spatial_latents(adata_xenium, res.qzx, ncols=3)

sc.pp.normalize_total(adata_xenium)
sc.pp.log1p(adata_xenium)
utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=5, sample_id=sample_id,
    show=True
)
