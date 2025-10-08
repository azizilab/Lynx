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
k = 50
r = 50

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500 # debug abundance penalization
lr = 1e-2
patience = 50

# Real data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))


# Preprocess, add cell-type labels in integers
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
# cluster_key = 'cell_type' if 'cell_type' in adata_xenium.obs.keys() else None
cluster_key = 'subtype'

# %%
graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs, 
    k=k,
    r=r,
    cluster_key=cluster_key,
    num_clusters=len(adata_xenium.obs[cluster_key].cat.categories),
    is_weighted=True
)

train_data, val_data = random_split(graph_data, [0.7, 0.3])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

# Training & Inference
train_configs = configs.set_train_configs(
    n_epochs=n_epochs,
    lr=lr, patience=patience, 
    device=torch.device('cuda'),
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
# sc.pp.neighbors(adata_xenium, use_rep='X_z')
# sc.tl.umap(adata_xenium)
# sc.pl.umap(adata_xenium)

# %%
# High-dim gradients
trajectory.compute_trajectory(
    adata_xenium, 
    use_rep='X_z',
    root_marker='DPT'
)

sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu_r', size=20, img=False,
    title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (Xenium)'
)

# Low-dim gradients
trajectory.compute_trajectory(
    adata_desi, 
    use_rep='X_z',
    root_marker='Taurine ',
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
# Computing discrete zones & zone-specific features (need log-normalized data)
sc.pp.normalize_total(adata_xenium)
sc.pp.log1p(adata_xenium)
utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=5, sample_id=sample_id,
    show=False
)

sq.pl.spatial_scatter(
    adata_xenium, color='zone', 
    size=20, img=False,
    title='Discrete Zonation\nLYNX (Xenium)'
)


# %%
# Save the latent embedding
# np.save('../results/liver/LYNX_xenium_6_debug1.npy', adata_xenium.obsm['X_z'])
# np.save('../results/liver/LYNX_desi_6_debug1.npy', adata_desi.obsm['X_z'])
# np.save('../results/liver/LYNX_t_debug.npy', adata_xenium.obs['t'].values)


# %%
# (3). Evaluate cell-cell interaction represented by cell-to-cell edge features

# %%
import holoviews as hv
hv.extension('bokeh')

def summarize_edge_weights(
    adata,  
    ccc_rep='omega', 
    cluster_key='cell_type', 
    cluster_labels=None,
    title='', 
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
        sns.heatmap(df, cmap="magma", linecolor='gray', linewidth=0.5)
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
# (3.1) Retrieve inferred edge weights (check sparsity?)
_ = summarize_edge_weights(
    adata_xenium, 
    cluster_key=cluster_key, 
    title='Overall Interaction',
    show_fig=True
)

# %%
# (3.2) Retrieve inferred edge weights per "zone"
attn_dfs = []
attn_graphs = []

categories = adata_xenium.obs[cluster_key].cat.categories

for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_attn_df = summarize_edge_weights(
        adata_sub, 
        cluster_labels=categories, 
        title='Interaction (Zone {})'.format(cluster_id),
        show_fig=True
    )
    attn_dfs.append(zone_attn_df)
    attn_graph = plot_celltype_interaction(zone_attn_df, amplitude=10)
    attn_graphs.append(attn_graph)

holomap = hv.HoloMap({i: graph for i, graph in enumerate(attn_graphs)},  kdims='{}\nBin (PV->CV)'.format(sample_id))
holomap = holomap.opts(
    xaxis=None, yaxis=None, axiswise=True,
    width=500, height=500
) 
holomap

  # %%
# Whether learnt edge weights are sparse??
plt.figure(figsize=(5, 2))
plt.hist(adata_xenium.obsm['omega'].flatten(), bins=50, edgecolor='black')
plt.show()

# %%
