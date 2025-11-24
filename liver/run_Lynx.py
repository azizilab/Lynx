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

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import rcParams

sns.set_context('paper')
rcParams.update({'font.family': 'Arial'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')
import IO, plot, utils, test_assoc, trajectory
import vgae, configs, dataset

from importlib import reload
%matplotlib inline
%load_ext autoreload
%autoreload 2

# %%
# Hyperparameters
n_subgraphs = 16
n_hidden = 32
n_latent = 6
n_epochs = 500
lr = 1e-2
patience = 20

# Try cleanup xenium data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
# sample_id = 'NIH_F5'
sample_id = 'NIH_F5_proseg'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
# cluster_key = 'cell_type' if 'cell_type' in adata_xenium.obs.keys() else None
cluster_key = 'subtype'

graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs, 
    r=50, is_weighted=True, alpha=1.0,
    cluster_key=cluster_key
)

train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience, 
    device=torch.device('cuda')
)

model_configs = configs.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden, 
    c_latent=n_latent,
    act=nn.SiLU(),
    infer_cell_interaction=True,
    temperature=0.3
)

# %%
model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(graph_data, train_configs, DEBUG=True)
res = model.evaluate(
    adata_xenium, adata_desi,
    graph_data=graph_data,
    device=torch.device('cpu')
)

# %%
# # Save the latent embedding
# np.save('../results/liver/LYNX_xenium_6_debug.npy', adata_xenium.obsm['X_z'])
# np.save('../results/liver/LYNX_desi_6_debug.npy', adata_desi.obsm['X_z'])
# np.save('../results/liver/LYNX_t_debug.npy', adata_xenium.obs['t'].values)
# adata_xenium.write_h5ad('../results/liver/LYNX_xenium_6_debug.h5ad')

# outdir = '../results/liver/downstream/gradient'
# np.save(os.path.join(outdir, f'LYNX_{sample_id}_xenium_latent.npy'), adata_xenium.obsm['X_z'])
# np.save(os.path.join(outdir, f'LYNX_{sample_id}_desi_latent.npy'), adata_desi.obsm['X_z'])
# np.save(os.path.join(outdir, f'LYNX_{sample_id}_xenium_gradient.npy'), adata_xenium.obs['t'].values)
# np.save(os.path.join(outdir, f'LYNX_{sample_id}_desi_gradient.npy'), adata_desi.obs['t'].values)


# %%
# TMP: load saved adata w/ all parameters
# adata_xenium = sc.read_h5ad('../results/liver/LYNX_xenium_6_debug.h5ad')

# %%
# TODO: separate evaluation scripts
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
# Xenium gradient 
curve = trajectory.get_curve(adata_xenium, epg_lambda=0.01)
trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu_r', size=25, img=False,
    title=r'Inferred spatial Gradient $(t)$'+'\nLYNX'
)

plot.disp_trajectory(
    adata_xenium, 
    cmap='RdBu_r',
    title='Spatial Gradient \n LYNX (Xenium)'
)

# DESI gradient
curve = trajectory.get_curve(adata_desi, epg_lambda=0.01)
trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine ')

# sq.pl.spatial_scatter(
#     adata_desi, color='t', 
#     cmap='RdBu_r', size=1, img=False,
#     title=r'Spatial Gradient $(t)$'+'\nLYNX (DESI)'
# )

# plot.disp_trajectory(
#     adata_desi, 
#     cmap='RdBu_r',
#     title='Spatial Gradients\n LYNX (DESI)'
# )

# %%
if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
    sc.pp.normalize_total(adata_xenium)
    sc.pp.log1p(adata_xenium)

utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=3, sample_id=sample_id,
    abundance_test=True,
    show=True
)
sq.pl.spatial_scatter(
    adata_xenium, color='zone',
    size=25, img=False,
)

# %%
# (iii). Evaluate cell-cell interaction represented by cell-to-cell edge features
# (3.1) Retrieve overview summary of cell-cell interaction (apriori)
adata_xenium.obs[cluster_key] = adata_xenium.obs[cluster_key].astype('category')
cluster_labels=adata_xenium.obs[cluster_key].cat.categories
cci_df = plot.summarize_cell_interaction(
    adata_xenium, 
    cluster_key=cluster_key, 
    cluster_labels=cluster_labels,
    title='Overall Interaction',
    show_fig=True
)

# %%
# TMP: spatial cell-type distribution
sq.pl.spatial_scatter(
    adata_xenium, color='subtype',
    groups=['Progenitor+Cholangiocytes', 'PC-Hep', 'PP-Hep'],
    size=25, img=False,
)

# %%
# (3.2) Visualize spatial interaction within a local niche
# E.g. Visualize T-cell interaction patterns along the gradient
adata_subset = adata_xenium.copy()
adata_subset.obs.reset_index(inplace=True, drop=True)
cell_boundaries_filename = os.path.join(xenium_path, sample_id, 'cell_boundaries.parquet')
for idx in adata_subset.obs[adata_subset.obs[cluster_key] == 'SMCs'].sort_values('t').index[:5]:
    plot.disp_spatial_interaction(
        adata_xenium,
        target_idx=idx,
        cell_boundaries_parquet=cell_boundaries_filename,
        cluster_key=cluster_key,
    )
del idx, adata_subset


# %% 
cell_boundaries_filename = os.path.join(xenium_path, sample_id, 'cell_boundaries.parquet')
rand_indices= np.random.choice(adata_xenium.n_obs, size=5, replace=False)
for idx in rand_indices:
    subgraph_dict = plot.disp_spatial_interaction(
        adata_xenium,
        target_idx=idx,
        cell_boundaries_parquet=cell_boundaries_filename,
        cluster_key=cluster_key,
        return_subgraph=True
    )
    print(subgraph_dict['omega'])
    print(subgraph_dict['omega'].sum())
del idx


# %%
# (3.4) Statistical test vs. abundance
cluster_labels = adata_xenium.obs[cluster_key].cat.categories

sig_mask = test_assoc.test_cci(adata_xenium, cluster_labels, cluster_key=cluster_key)
cci_df = cci_df * sig_mask
plot.disp_heatmap(
    cci_df, 
    title='Significant cell-cell interaction (Overall)',
)

cci_dfs = []
for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_cci_df = plot.summarize_cell_interaction(
        adata_sub, 
        cluster_key=cluster_key,
        cluster_labels=cluster_labels,
        show_fig=False
    )
    
    sig_mask = test_assoc.test_cci(adata_sub, cluster_labels, cluster_key=cluster_key)
    zone_cci_df = zone_cci_df * sig_mask
    cci_dfs.append(zone_cci_df)
    plot.disp_heatmap(
        zone_cci_df, 
        title=f'Significant cell-cell interaction (Zone {int(cluster_id)})',
    )

    plot.netVisual_circle(
        zone_cci_df, min_threshold=0.05, vertex_size_max=20, figsize=(15, 15),
        title=f'Summary of cell-cell interaction\n (Zone {int(cluster_id)})' 
    )

del zone_cci_df
gc.collect()

# %%
fig, ax = plot.netVisual_circle(
    cci_dfs[1], min_threshold=0.05, vertex_size_max=20, figsize=(15, 15),
    title=f'Summary of cell-cell interaction\n (Zone 2)'
)
fig.savefig('../figures/LYNX_fig2_cci_zone2.pdf', bbox_inches='tight')



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
        infer_cell_interaction=True,
        # temperature=0.3
    ) 
    model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
    model.fit(graph_data, train_configs, DEBUG=True)

    # Full inference with best model params
    res = model.evaluate(
        adata_xenium, adata_desi,
        graph_data=graph_data,
        device=torch.device('cpu')
    )

    curve = trajectory.get_curve(adata_xenium)
    trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

    # # Low-dim gradients
    curve = trajectory.get_curve(adata_desi)
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
