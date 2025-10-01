# Debug runs on simulation & real-data for 
# DESI (y) -> Latent (z) -> Xenium (x) generative paths

# %%
import os
import gc
import sys

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
from importlib import reload

# %%
# Dataset specs
n_subgraphs = 16
k = 30
r = 50

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500
lr = 1e-2
patience = 50

# Real data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id+'_proseg'), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'_reseg.h5'))

# Preprocess, add cell-type labels in integers
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
cluster_key = 'cell_type' if 'cell_type' in adata_xenium.obs.keys() else None
# cluster_key = 'subtype'


# %%
graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs, 
    k=k,
    r=r,
    cluster_key=cluster_key,
    is_weighted=True
)

train_data, val_data = random_split(graph_data, [0.7, 0.3])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

# Training & Inference
train_configs = configs.set_train_configs(
    n_epochs=n_epochs,
    lr=lr, patience=patience, 
    device=torch.device('cuda'),
    anneal=False,
    verbose=True
)

model_configs = configs.set_model_configs(
    c_in=adata_xenium.shape[1],   # ref-dim 
    c_aux=adata_desi.shape[1],  # query-dim
    c_hidden=n_hidden, 
    c_latent=n_latent,
    act=nn.SiLU(),
    ref=graph_data.ref, 
    query=graph_data.query,
    num_clusters=graph_data.num_clusters,
    verbose=True
) 

# %%
if 'model' in globals():
    del model
pyro.clear_param_store()
torch.cuda.empty_cache()

model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(train_configs, train_dl=train_dl, val_dl=val_dl, DEBUG=True)

# Full inference with best model params
res = model.evaluate(
    adata_xenium, adata_desi,
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
# (2). Trajectory Inference
# Quick correctness check via UMAP visualization
sc.pp.neighbors(adata_xenium, use_rep='X_z')
sc.tl.umap(adata_xenium)
sc.pl.umap(adata_xenium)


# %%
# High-dim gradients
adata_xenium.obsm['X_z'] = res.qzx
trajectory.compute_trajectory(
    adata_xenium, 
    use_rep='X_z',
    # dist_metric='knn',
    root_marker='DPT',
)

sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu_r', size=20, img=False,
    title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (Xenium)'
)

# Low-dim gradients
adata_desi.obsm['X_z'] = res.qzu
trajectory.compute_trajectory(
    adata_desi, 
    use_rep='X_z',
    # dist_metric='knn',
    root_marker='Taurine '
)

sq.pl.spatial_scatter(
    adata_desi, color='t', 
    cmap='RdBu_r', size=1, img=False,
    title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (DESI)'
)

# %%
plot.disp_trajectory(
    adata_xenium, 
    cmap='RdBu_r',
    title='Spatial Gradients\n LYNX (Xenium)'
)

plot.disp_trajectory(
    adata_desi, 
    cmap='RdBu_r',
    title='Spatial Gradients\n LYNX (DESI)'
)

# %%
# Visualize latent (z) & spatial clustering
# plot.disp_factor_corr(res.qzx)
# plot.disp_spatial_latents(adata_xenium, res.qzx, ncols=3)

# Computing discrete zones & zone-specific features (need log-normalized data)
sc.pp.normalize_total(adata_xenium)
sc.pp.log1p(adata_xenium)
utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=5, sample_id=sample_id,
    show=True
)

# %%
# (3). Evaluate cell-cell interaction represented by cell-to-cell edge features

# %%
# (3.1) Retrieve inferred edge weights (check sparsity?)
def disp_edge_weights(adata, ccc_rep='omega', cluster_key='cell_type'):
    """Display summary of cell-cell interactions"""
    cluster_labels = adata.obs[cluster_key].cat.categories
    per_idx_labels = adata_xenium.obs['cell_type'].values
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, cmap="magma")
    plt.xlabel("Sender cell type / omega")
    plt.ylabel("Receiver cell type")
    plt.title("Cell type × Cell type + Omega attention scores")
    plt.show()

cluster_coeff = adata_xenium.obsm['omega']
disp_edge_weights(adata_xenium)

# %%
# TODO: debugging manifold learning with ODE methods??
# Save the latent embedding
# np.save('../results/liver/LYNX_xenium_6_debug.npy', adata_xenium.obsm['X_z'])


# %%
# (3.2) Retrieve inferred edge weights per "zone"
import holoviews as hv
hv.extension('bokeh')

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


def build_celltype_attention(
    adata_subset,
    categories,
    attn_key='omega',
    celltype_key='cell_type',
    agg='mean',
    normalize=False
):
    """
    Parameters
    ----------
    adata_subset : AnnData
        The subset of the full data.
    attn_key : str
        Key in adata_subset.obsm storing the attention matrix (n_cells x n_clusters).
    celltype_key : str
        The obs field containing the raw cell type strings for each cell.
    agg : str or callable
        Aggregation function for grouping by cell type. E.g., 'mean', 'sum', etc.
    normalize : boolean
        Whether to use Laplacian normalization

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (#full_celltypes, #full_celltypes) with the cell types
        as both row and column labels.
    """

    row_labels = adata_subset.obs[celltype_key].values

    # Retrieve the attention matrix.
    attn_matrix = adata_subset.obsm[attn_key]

    # Build a DataFrame: rows are the "target" cell types (raw strings)
    # and columns correspond to the factorized cell type labels.
    df = pd.DataFrame(
        data=attn_matrix,
        index=row_labels,
        columns=categories,
    )

    # Aggregate attention values by the target cell type.
    df_agg = df.groupby(level=0).agg(agg)
    intersection = df_agg.index.intersection(categories)
    df_agg = df_agg.loc[intersection]
    df_agg = df_agg[df_agg.index]

    # Symmetrize the matrix by adding its transpose.
    df_agg = df_agg.add(df_agg.T, fill_value=0)

    # Normalize rows by their sums.
    row_sums = df_agg.sum(axis=1).replace(0, np.nan)
    M = df_agg.values
    d_inv_sqrt = 1.0 / np.sqrt(row_sums.values)  # shape: (K,)

    # Outer product scaling: M_norm[i,j] = M[i,j] / sqrt(rowSum[i]*rowSum[j])
    M_norm = d_inv_sqrt[:, None] * M * d_inv_sqrt[None, :]

    # Return the final DataFrame with full_categories as both index and columns.
    df = pd.DataFrame(M_norm if normalize else M, index=df_agg.index, columns=df_agg.columns)
    df.loc[(df!=0).any(axis=1)]
    return df

# %%
sq.pl.spatial_scatter(
    adata_xenium, color='zone', 
    size=20, img=False,
)


# %%
attn_dfs = []
attn_graphs = []
categories = adata_xenium.obs[cluster_key].cat.categories

for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_attn_df = build_celltype_attention(
        adata_subset=adata_sub,
        attn_key='omega',
        celltype_key='cell_type',
        categories=categories,
        agg='mean',
        normalize=True
    )
    attn_dfs.append(zone_attn_df)

    plt.figure(figsize=(6,5))
    sns.heatmap(zone_attn_df, cmap='magma')
    plt.title(f"Cell-type attention for z_cluster={cluster_id}")
    plt.xlabel("Cell type")
    plt.ylabel("Cell type")
    plt.tight_layout()
    plt.show()

    attn_graph = plot_celltype_interaction(zone_attn_df, amplitude=10)
    attn_graphs.append(attn_graph)

# %%
# Whether learnt edge weights are sparse??
plt.figure(figsize=(5, 2))
plt.hist(adata_xenium.obsm['omega'].flatten(), bins=50, edgecolor='black')
plt.show()

# %%
