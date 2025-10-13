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
rcParams.update({'font.family': 'Arial'})
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

# Optional: rescale the image based on scalefactor
# he_img = rescale(he_img, scalefactor, preserve_range=True, channel_axis=0)
# gc.collect()

# %%
# Attach cell-type annotations
# cell_type = pd.read_csv(os.path.join(data_path, sample_id, 'Xenium_annot.csv'), index_col=[0])['Cluster'].to_list()
# adata_xenium.obs['cell_type'] = cell_type
# adata_xenium.obs['cell_type'] = adata_xenium.obs['cell_type'].astype('category')

# %%
# Save DCIS patch suggested by Janesick & Chitra

# Save HE image patch
# xmin, xmax = 1500, 3250
# ymin, ymax = 2000, 4000
# outdir = os.path.join(data_path, 'dcis_fov')
# if not os.path.exists(outdir):
#     os.makedirs(outdir, exist_ok=True)

# # Save HE image patch
# he_patch = he_img[:, ymin:ymax, xmin:xmax]
# tifffile.imwrite(os.path.join(outdir, 'he.tif'), he_patch)


# Save full-res HE image patch
xmin, xmax = int(1500/scalefactor), int(3250/scalefactor)
ymin, ymax = int(2000/scalefactor), int(4000/scalefactor)
outdir = os.path.join(data_path, 'dcis_fov')
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

he_patch = he_img[:, ymin:ymax, xmin:xmax]
tifffile.imwrite(os.path.join(outdir, 'he_hires.tif'), he_patch)


# Save expression patch
# adata_patch = adata_xenium[
#     (adata_xenium.obsm['spatial'][:, 0] >= xmin) & (adata_xenium.obsm['spatial'][:, 0] <= xmax) &
#     (adata_xenium.obsm['spatial'][:, 1] >= ymin) & (adata_xenium.obsm['spatial'][:, 1] <= ymax)
# ].copy()
# adata_patch.obsm['spatial'] -= np.array([xmin, ymin])
# adata_patch.write_h5ad(os.path.join(outdir, 'cell_feature_matrix.h5ad'))

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
# --- Patch extraction utility ---
def extract_patches(img, coords, P=64):
    """
    img: np.ndarray, shape (3, H, W), values in [0, 1] or [0, 255]
    coords: np.ndarray, shape (N, 2), (x, y)
    w: int, patch size
    Returns: np.ndarray, shape (N, 3*P*P)
    """
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
adata_he.write_h5ad(os.path.join(data_path, 'he_patches.h5ad'))


# %%
# ---------------
#   LYNX runs
# ---------------

# %%
# Dataset specs
n_subgraphs = 8
k = 50
r = 50

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 50
lr = 1e-2
patience = 50

# Real data; TODO: try condition on H&E embedding
data_path = '../data/breast/dcis_fov/'
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'cell_feature_matrix.h5'))
adata_he = sc.read_h5ad(os.path.join(data_path, 'he_patches.h5ad'))
cluster_key = 'cell_type'


# %%
# DEBUG color normalization!!!!!
plt.hist(adata_he.X[:100].flatten(), bins=50, edgecolor='k')
plt.title('H&E patch pixel intensity distribution')
plt.show()

# %%
# Is H&E image contaimnated???
indices = np.random.choice(adata_he.shape[0], size=5, replace=False)

for idx in indices:
    patch = adata_he[idx].X.toarray().reshape(3, 32, 32)
    patch = patch.transpose(1, 2, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(patch)
    ax2.hist(patch.flatten(), bins=50, edgecolor='k')
    plt.show()

del idx, indices, patch
gc.collect()



# %%
# TODO: Perturb u: condition on N(0, 1) gaussian noise
# adata_he.X = np.random.rand(adata_xenium.shape[0], 3072).astype(np.float32)

graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_he,
    n_subgraphs=n_subgraphs, 
    k=k, 
    r=r, 
    is_weighted=True,
    cluster_key=cluster_key,
    num_clusters=len(adata_xenium.obs[cluster_key].cat.categories),

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
    verbose=False
)

# %%
if 'model' in globals():
    del model
pyro.clear_param_store()
torch.cuda.empty_cache()

if 'model' in globals():
    del model
pyro.clear_param_store()
torch.cuda.empty_cache()

model_configs = configs.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden, 
    c_latent=n_latent,
    patch_size=32,
    act=nn.SiLU(),
    abundance_penalization=k
) 

# TODO: debug why sparse attention doesn't fit well
model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(train_configs, train_dl=train_dl, val_dl=val_dl, DEBUG=True)
res = model.evaluate(
    adata_xenium, adata_he,
    graph_data=graph_data,
    device=torch.device('cpu')
)

# %%
# Evaluation
# (i). Reconstruction
plot.disp_kde_scatter(
    adata_xenium.X.A.flatten().copy(),
    res.px.flatten().copy(),
    subset_ratio=0.005,
    xlabel=r"Observation (log1p)",
    ylabel=r"Reconstruction (log1p)",
    title='Xenium feature reconstruction\n u: paired H&E patch\n v: attention'
    # title='Xenium feature reconstruction (V: attention\n u: paired H&E patch)'
)
gc.collect()


# %%
# plt.figure(figsize=(10, 4))
# plt.hist(np.log1p(adata_xenium.X.A.flatten()), bins=50, edgecolor='k', alpha=0.5, label='Observed')
# plt.hist(np.log1p(res.px.flatten()), bins=50, edgecolor='k', alpha=0.5, label='Reconstructed')
# plt.legend()
# plt.show()


# %%
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
# (ii). Trajectory Inference
# TODO: re-implement analysis for  branching trajectories, find a common fork point

# High-dim gradients
# adata_xenium.obsm['X_z'] = res.qzx
# trajectory.compute_trajectory(
#     adata_xenium, 
#     use_rep='X_z',
#     ppt_lambda=1000,
#     ppt_sigma=.1,
#     ppt_niter=200,
#     root_marker='FASN',  # known invasive marker
#     # root_marker='CD8A',  # known immune marker
# )

# sq.pl.spatial_scatter(
#     adata_xenium, color='t', 
#     cmap='RdBu_r', size=20, img=False,
#     title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (Xenium)'
# )

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
# plot.disp_trajectory(
#     adata_xenium, 
#     cmap='RdBu_r',
#     title='Spatial Gradients\n LYNX (Xenium)'
# )

# %%
# Save low-dim embedding
# np.save(os.path.join('../results/breast/', 'LYNX_xenium_6.npy'), res.qzx)


# %%
# # Testing evidence of bifurcation
# precursor_markers = [
#     'ALDH1A3', 'KIT', 'EPCAM', 'RUNX1', 'LIF', 'FOXA1', 'FOXC2'
# ]

# EMT_markers = [
#     'ACTA2', 'CD44', 'CLDN5', 'SNAL1', 'ZEB1', 'ZEB2',
#     'MMP1', 'MMP2', 'MMP12', 'DSP', 'DSC2', 'CTTN', 
#     'ENAH', 'FOXC2', 'EGFR', 'ERBB2'
# ]

# precursur_markers = [g for g in precursor_markers if g in adata_xenium.var_names]
# EMT_markers = [g for g in EMT_markers if g in adata_xenium.var_names]

# # adata_norm = adata_xenium.copy()
# # sc.pp.normalize_total(adata_norm)
# # sc.pp.log1p(adata_norm)

# adata_xenium.obs['precursor_score'] = adata_norm[:, precursur_markers].X.mean(1)
# adata_xenium.obs['EMT_score'] = adata_norm[:, EMT_markers].X.mean(1)

# sc.pl.umap(adata_xenium, color='precursor_score', cmap='seismic')
# sc.pl.umap(adata_xenium, color='EMT_score', cmap='seismic')
# sc.pl.umap(adata_xenium, color=['CEACAM6', 'FASN'], cmap='seismic')


# %%
# Ablation: modifying `sigma` regularization strength
# branching vs. curve
import scFates as scf
scf.tl.tree(
    adata_xenium,
    use_rep='X_z',
    Nodes=int(adata_xenium.shape[0] * 0.1),
    ppt_lambda=1e5,
    seed=42,
    device=torch.device('cuda')
)

scf.pl.graph(adata_xenium)

# %%
# Branching trajectory from "fork" to "tips"
if 'milestones_colors' in adata_xenium.uns.keys():
    adata_xenium.uns.pop('milestones_colors')
scf.tl.root(adata_xenium, 931)
scf.tl.pseudotime(adata_xenium,n_jobs=20,n_map=100,seed=42)
t = adata_xenium.obs['t'].values
adata_xenium.obs['t'] = (t - t.min()) / (t.max() - t.min())


# %%
sc.pl.umap(
    adata_xenium, color='t', cmap='RdBu_r'
)

# %%
# (iii). Cell-cell interaction analysis

import holoviews as hv
hv.extension('bokeh')

def summarize_edge_weights(
    adata,  
    ccc_rep='omega', 
    cluster_key='cell_type', 
    cluster_labels=None,
    title='', 
    vmax=None,
    show_fig=False
):
    """Compute cluster-wise summary of cell-cell interactions"""
    if cluster_labels is None:
        cluster_labels = adata.obs[cluster_key].cat.categories
    per_idx_labels = adata.obs['cell_type'].values
    n_clusters = len(cluster_labels)
    mat = np.zeros((n_clusters, n_clusters), dtype=np.float32)

    # Aggregate: for each receiver type, average over its cells
    for i, rtype in enumerate(cluster_labels):
        mask = (per_idx_labels == rtype)
        if mask.sum() > 0:
            mat[i] = adata.obsm[ccc_rep][mask].mean(axis=0)   # sender cell types

    # add omega as an extra sender column
    df = pd.DataFrame(
        mat,
        index=cluster_labels, 
        columns=list(cluster_labels)
    )

    # plot heatmap
    if show_fig:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, cmap="magma", vmax=vmax, linecolor='gray', linewidth=0.5)
        plt.xlabel("Sender", fontsize=10)
        plt.ylabel("Receiver", fontsize=10)
        plt.title(title, fontsize=20)
        plt.show()

    return df


def plot_celltype_interaction(attn_df, amplitude=1):
    assert np.array_equal(attn_df.index, attn_df.columns)
    attn_score = attn_df.values
    cell_types = attn_df.columns

    graph = hv.Graph([
        (cell_types[i], cell_types[j], attn_score[i, j])
        for i in range(len(cell_types)-1) for j in range(i+1, len(cell_types))
    ], vdims=['weight'])
    labels = hv.Labels(graph.nodes, ['x', 'y'], 'index')

    graph = graph.opts(
        node_color='index', edge_color=hv.dim('weight')*amplitude, cmap='Category10',
        edge_cmap='Reds', edge_line_width=hv.dim('weight')*amplitude,
    )
    graph = (graph * labels.opts(text_font_size='10pt', text_color='black'))

    return graph

# %%
categories = adata_xenium.obs[cluster_key].cat.categories
abundance = graph_data[0]['Xenium'].abundance
abundance_dict = {cat: abundance[i] for i, cat in enumerate(categories)}

# %%
# Overall cell-cell interaction
_ = summarize_edge_weights(
    adata_xenium, 
    cluster_key=cluster_key, 
    title='Overall Interaction',
    vmax=.5,
    show_fig=True
)

# %%
# TODO: redo stromal cluster subtypes!!!!


