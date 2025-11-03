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
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

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
# Dataset specs
n_subgraphs = 16
k = 50
r = 50

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500
lr = 1e-2
patience = 50

xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
cluster_key = 'cell_type' if 'cell_type' in adata_xenium.obs.keys() else None

graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs, 
    k=k, r=r, is_weighted=True,
    alpha=0.5, 
    cluster_key=cluster_key
)

train_data, val_data = random_split(graph_data, [0.7, 0.3])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)
train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience, 
    device=torch.device('cuda')
)


# %%
if 'model' in globals():
    del model
pyro.clear_param_store()
torch.cuda.empty_cache()

model_configs = configs.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden, 
    c_latent=n_latent,
    act=nn.SiLU(),
    infer_cell_interaction=True,
    temperature=0.3
) 
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
# (i). Reconstruction
plot.disp_kde_scatter(
    adata_xenium.X.A.flatten(),
    res.px.flatten(),
    subset_ratio=0.001,
    xlabel=r"Observation $log(1+x)$",
    ylabel=r"Reconstruction $log(1+x)$",
    title='Feature reconstruction (Human liver)'
)
gc.collect()

# %%
# (ii). Trajectory Inference
# High-dim gradients
curve = trajectory.get_curve(
    adata_xenium, 
    epg_mu=5.0,
    epg_lambda=1.0,
)
trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu_r', size=20, img=False,
    title=r'Spatial Gradient $(t)$'+'\nLYNX (Xenium)'
)

# Low-dim gradients
curve = trajectory.get_curve(
    adata_desi, 
    epg_mu=5.0,
    epg_lambda=1.0,
)
trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine ')

sq.pl.spatial_scatter(
    adata_desi, color='t', 
    cmap='RdBu_r', size=1, img=False,
    title=r'Spatial Gradient $(t)$'+'\nLYNX (DESI)'
)

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable

def disp_trajectory(
    adata, 
    use_rep=None,
    figsize=(5, 4),
    cmap='RdBu_r',
    title=None
):
    if use_rep is None:
        use_rep = 'X_z'
    else:
        assert use_rep in adata.obsm.keys()

    principal_repr = adata.uns['graph']['F'][
        adata.uns['graph']['pnode_indices']
    ]
    n_nodes = principal_repr.shape[0]
    adata_repr = sc.AnnData(
        np.vstack([adata.obsm[use_rep], principal_repr])
    )
    sc.pp.neighbors(adata_repr)
    sc.pp.pca(adata_repr, n_comps=adata_repr.shape[1]-1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.scatter(
        adata_repr.obsm['X_pca'][:-n_nodes, 0],
        adata_repr.obsm['X_pca'][:-n_nodes, 1],
        c=adata.obs['t'], s=0.1, edgecolors=None, cmap=cmap
    )
    ax.plot(
        adata_repr.obsm['X_pca'][-n_nodes:, 0],
        adata_repr.obsm['X_pca'][-n_nodes:, 1],
        '.-', color='gray', lw=.5, ms=2, mfc='yellow'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    cb = plt.gcf().axes[-1]
    cb.set_ylabel(r'Pseudotime $(t)$', fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.show()

disp_trajectory(
    adata_xenium, 
    cmap='RdBu_r',
    title=r'Spatial gradients $(t)$ - LYNX'
)

# plot.disp_trajectory(
#     adata_desi, 
#     cmap='RdBu_r',
#     title='Spatial Gradients\n LYNX (DESI)'
# )

# %%
adata_xenium.obs.keys()


# %%
if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
    sc.pp.normalize_total(adata_xenium)
    sc.pp.log1p(adata_xenium)

# utils.get_zonation_features(    
#     adata_xenium, adata_desi,
#     n_zones=3, sample_id=sample_id,
#     abundance_test=True,
#     show=True
# )

utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=5, sample_id=sample_id,
    abundance_test=True,
    show=True
)

# %%
# Save the latent embedding
# np.save('../results/liver/LYNX_xenium_6_debug.npy', adata_xenium.obsm['X_z'])
# np.save('../results/liver/LYNX_desi_6_debug.npy', adata_desi.obsm['X_z'])
# np.save('../results/liver/LYNX_t_debug.npy', adata_xenium.obs['t'].values)

# outdir = '../results/liver/downstream/gradient'
# np.save(os.path.join(outdir, f'LYNX_{sample_id}_xenium_latent.npy'), adata_xenium.obsm['X_z'])
# np.save(os.path.join(outdir, f'LYNX_{sample_id}_desi_latent.npy'), adata_desi.obsm['X_z'])
# np.save(os.path.join(outdir, f'LYNX_{sample_id}_xenium_gradient.npy'), adata_xenium.obs['t'].values)
# np.save(os.path.join(outdir, f'LYNX_{sample_id}_desi_gradient.npy'), adata_desi.obs['t'].values)


# %%
# (iii). Evaluate cell-cell interaction represented by cell-to-cell edge features
# (3.1) Retrieve inferred edge weights (check sparsity?)
_ = plot.summarize_cell_interaction(
    adata_xenium, 
    cluster_key=cluster_key, 
    title='Overall Interaction',
    show_fig=True
)

# %%
# (3.2) Retrieve inferred edge weights per "zone"
attn_dfs = []
categories = adata_xenium.obs[cluster_key].cat.categories

for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_attn_df = plot.summarize_cell_interaction(
        adata_sub, 
        cluster_labels=categories, 
        title=f'Interaction (Zone {int(cluster_id)})',
        show_fig=True
    )
    attn_dfs.append(zone_attn_df)

# %%
# Interactive plot
import holoviews as hv
hv.extension('bokeh')

attn_graphs = []
categories = adata_xenium.obs[cluster_key].cat.categories

for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_attn_df = plot.summarize_cell_interaction(
        adata_sub, 
        cluster_labels=categories, 
    )
    attn_graph = plot.interactive_cell_interaction(zone_attn_df, amplitude=10)
    attn_graphs.append(attn_graph)

holomap = hv.HoloMap({i+1: graph for i, graph in enumerate(attn_graphs)},  kdims='{}\nBin (PV->CV)'.format(sample_id))
holomap = holomap.opts(
    xaxis=None, yaxis=None, axiswise=True,
    width=500, height=500
) 
holomap

# %%
# (3.3) Visualize spatial interaction within a local niche
subgraph_dict = plot.disp_spatial_interaction(
    adata_xenium, 
    target_idx=300,
    cluster_key=cluster_key, 
    return_subgraph=True,
    figsize=(8, 6)
)


# %%
for _ in range(20):
    _ = plot.disp_spatial_interaction(
        adata_xenium, 
        cluster_key=cluster_key, 
        figsize=(8, 6)
    )

gc.collect()

# %%
# (3.4) Statistical test to get post-hoc interaction values


# %%
# ---------------------------
#   multi-sample running
# ---------------------------
n_subgraphs = 16
k = 30
r = 50

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500
lr = 1e-2
patience = 20

# %%
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_ids = [
    'NIH_F1', 'NIH_F2', 'NIH_F3', 'NIH_F4', 'NIH_F5',
    'NIH_M1', 'NIH_M2', 'NIH_M3', 'NIH_M4', 'NIH_M5'
]
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'

for sample_id in sample_ids:
    adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
    adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))

    # Preprocess, add cell-type labels in integers
    adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
    cluster_key = 'cell_type' if 'cell_type' in adata_xenium.obs.keys() else None

    graph_data = dataset.HeteroDataset(
        adatas_ref=adata_xenium, 
        adatas_query=adata_desi,
        n_subgraphs=n_subgraphs, 
        k=k, r=r, is_weighted=True,
        # alpha=0.1,
        cluster_key=cluster_key
    )

    train_data, val_data = random_split(graph_data, [0.7, 0.3])
    train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)
    train_configs = configs.set_train_configs(
        n_epochs=n_epochs,
        lr=lr, patience=patience, 
        device=torch.device('cuda')
    )

    if 'model' in globals():
        del model
    pyro.clear_param_store()
    torch.cuda.empty_cache()

    model_configs = configs.set_model_configs(
        graph_data=graph_data,
        c_hidden=n_hidden, 
        c_latent=n_latent,
        act=nn.SiLU(),
        infer_cell_interaction=False,
        # temperature=0.3
    ) 
    model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
    model.fit(train_configs, train_dl=train_dl, val_dl=val_dl, DEBUG=True)

    # Full inference with best model params
    res = model.evaluate(
        adata_xenium, adata_desi,
        graph_data=graph_data,
        device=torch.device('cpu')
    )

    curve = trajectory.get_curve(
        adata_xenium, 
        epg_mu=5.0,
        epg_lambda=1.0,
    )
    trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

    # # Low-dim gradients
    curve = trajectory.get_curve(
        adata_desi, 
        epg_mu=5.0,
        epg_lambda=1.0,
    )
    trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine ')

    if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
        sc.pp.normalize_total(adata_xenium)
        sc.pp.log1p(adata_xenium)

    utils.get_zonation_features(    
        adata_xenium, adata_desi,
        n_zones=3, sample_id=sample_id,
        abundance_test=True,
        show=True
    )

    utils.get_zonation_features(    
        adata_xenium, adata_desi,
        n_zones=5, sample_id=sample_id,
        abundance_test=True,
        show=True
    )

    outdir = '../results/liver/downstream/gradient'
    np.save(os.path.join(outdir, f'LYNX_{sample_id}_xenium_latent.npy'), adata_xenium.obsm['X_z'])
    np.save(os.path.join(outdir, f'LYNX_{sample_id}_desi_latent.npy'), adata_desi.obsm['X_z'])
    np.save(os.path.join(outdir, f'LYNX_{sample_id}_xenium_gradient.npy'), adata_xenium.obs['t'].values)
    np.save(os.path.join(outdir, f'LYNX_{sample_id}_desi_gradient.npy'), adata_desi.obs['t'].values)

    del model, adata_xenium, adata_desi, graph_data
    gc.collect()
    torch.cuda.empty_cache()

# %%
