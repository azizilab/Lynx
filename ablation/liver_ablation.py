# %%
# -------------------------------------------------------------------
# Ablation study — Human Liver (NIH_F5)
# - Ablation 1: SpatialVGAE   — LYNX without auxiliary u modality
# - Ablation 2: randomized auxiliary u modality
# -------------------------------------------------------------------
import os
import gc
import sys
import time

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

sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../util')
import IO, plot, utils, trajectory
import vgae, configs, dataset
from base_model import SpatialVGAE
from dataset import HeteroDataset

from importlib import reload
%matplotlib inline
%load_ext autoreload
%autoreload 2

# %%
#  Hyperparameters
n_subgraphs = 16
n_hidden     = 32
n_latent     = 6
n_epochs     = 500
lr           = 1e-2
patience     = 20
DEVICE       = torch.device('cuda')
OUTDIR       = '../results/liver/ablation/'
os.makedirs(OUTDIR, exist_ok=True)

# %%
# Load data
xenium_path = '../data/xenium/'
desi_path   = '../data/desi/'
sample_id   = 'NIH_F5_proseg'
cluster_key = 'subtype'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi   = sc.read_h5ad(os.path.join(desi_path, sample_id + '.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')

# %%
# Data construction
graph_data = HeteroDataset(
    adatas_ref=adata_xenium,
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs,
    r=50, is_weighted=True, alpha=1.0,
    cluster_key=cluster_key
)

train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience,
    device=DEVICE
)

# %%
# ======================================================
# ABLATION 1 — SpatialVGAE (LYNX without auxiliary u)
# ======================================================

model_configs_abl1 = configs.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden,
    c_latent=n_latent,
    act=nn.SiLU(),
    infer_cell_interaction=True,   
    temperature=0.3
)

# %%
model_abl1 = SpatialVGAE(model_configs_abl1, device=DEVICE)
model_abl1.fit(graph_data, train_configs, DEBUG=True)

res_abl1 = model_abl1.evaluate(
    adata_xenium, adata_desi,
    graph_data=graph_data,
    device=torch.device('cpu')
)

# %%
# Reconstruction quality
plot.disp_kde_scatter(
    adata_xenium.X.A.flatten(),
    res_abl1.px.flatten(),
    subset_ratio=0.001,
    xlabel=r'Observation $\log(1+x)$',
    ylabel=r'Reconstruction $\log(1+x)$',
    title='Feature reconstruction — SpatialVGAE (w/o auxiliary)'
)
gc.collect()

# %%
# Principal curve + pseudotime — Xenium (SpatialVGAE)
curve_abl1 = trajectory.get_curve(adata_xenium, epg_lambda=0.01, trim_radius_ratio=0.25)
trajectory.compute_pseudotime(adata_xenium, curve_abl1, root_marker='DPT')

sq.pl.spatial_scatter(
    adata_xenium, color='t',
    cmap='RdBu_r', size=25, img=False,
    title='Inferred Spatial Gradient\nSpatialVGAE (w/o auxiliary)'
)

plot.disp_trajectory(
    adata_xenium,
    cmap='RdBu_r',
    title='Inferred Spatial Gradient\nSpatialVGAE (w/o auxiliary) embedding'
)

# Stash pseudotime before ablation 2 overwrites X_z / t
adata_xenium.obs['t_abl1'] = adata_xenium.obs['t'].copy()
adata_xenium.obsm['X_z_abl1'] = adata_xenium.obsm['X_z'].copy()

# %%
# Save embeddings
np.save(os.path.join(OUTDIR, 'SpatialVGAE_xenium_z.npy'), adata_xenium.obsm['X_z_abl1'])

# %%
# ======================================================================
# ABLATION 2 — LYNX with randomized u  (randomized query coordinates)
# ======================================================================

rng = np.random.default_rng(42)
adata_desi_shuffled = adata_desi.copy()

rand_map_coords = np.random.uniform(
    low=adata_desi.obsm['xenium_map'].min(axis=0),
    high=adata_desi.obsm['xenium_map'].max(axis=0),
    size=adata_desi.obsm['xenium_map'].shape
)
adata_desi_shuffled.obsm['xenium_map'] = rand_map_coords

# %%
graph_data_shuffled = HeteroDataset(
    adatas_ref=adata_xenium,
    adatas_query=adata_desi_shuffled,
    n_subgraphs=n_subgraphs,
    r=50, is_weighted=True, alpha=1.0,
    cluster_key=cluster_key
)

model_configs_abl2 = configs.set_model_configs(
    graph_data=graph_data_shuffled,
    c_hidden=n_hidden,
    c_latent=n_latent,
    act=nn.SiLU(),
    infer_cell_interaction=True,
    temperature=0.3
)

# %%
model_abl2 = vgae.HeteroAttnVGAE(model_configs_abl2, device=DEVICE)
model_abl2.fit(graph_data_shuffled, train_configs, DEBUG=True)

res_abl2 = model_abl2.evaluate(
    adata_xenium, adata_desi_shuffled,
    graph_data=graph_data_shuffled,
    device=torch.device('cpu')
)

# %%
# Reconstruction quality
plot.disp_kde_scatter(
    adata_xenium.X.A.flatten(),
    res_abl2.px.flatten(),
    subset_ratio=0.001,
    xlabel=r'Observation $\log(1+x)$',
    ylabel=r'Reconstruction $\log(1+x)$',
    title='Feature reconstruction — LYNX (off-aligned coordinates)'
)
gc.collect()

# %%
curve_abl2_xenium = trajectory.get_curve(adata_xenium, epg_lambda=0.01, trim_radius_ratio=0.25)
trajectory.compute_pseudotime(adata_xenium, curve_abl2_xenium, root_marker='DPT')

sq.pl.spatial_scatter(
    adata_xenium, color='t',
    cmap='RdBu_r', size=25, img=False,
    title='Inferred Spatial Gradient\nLYNX (off-aligned auxiliary)'
)

plot.disp_trajectory(
    adata_xenium,
    cmap='RdBu_r',
    title='Inferred Spatial Gradient\nLYNX (off-aligned auxiliary) embedding'
)

adata_xenium.obs['t_abl2'] = adata_xenium.obs['t'].copy()
adata_xenium.obsm['X_z_abl2'] = adata_xenium.obsm['X_z'].copy()

# %%
# Save embeddings
np.save(os.path.join(OUTDIR, 'RandomU_xenium_z.npy'), adata_xenium.obsm['X_z_abl2'])

# %%
# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, t_key, title in zip(
    axes,
    ['t_abl1', 't_abl2'],
    ['SpatialVGAE (no u)', 'LYNX (random u)']
):
    adata_xenium.obs['t_plot'] = adata_xenium.obs[t_key]
    sq.pl.spatial_scatter(
        adata_xenium, color='t_plot',
        cmap='RdBu_r', size=25, img=False,
        title=f'Spatial Gradient\n{title}',
        ax=ax
    )

plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots()
ax = sq.pl.spatial_scatter(
    adata_xenium, color='t_abl1', size=20, 
    cmap='RdBu_r', img=False, colorbar=False, return_ax=True, 
    fig=fig, ax=ax, title=r'Inferred spatial gradient - LYNX (no auxiliary)'
)
sm = ax.collections[0] 
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
cbar.set_label(r'Pseudotime $(t)$', fontsize=8)
plt.show()

fig, ax = plt.subplots()
ax = sq.pl.spatial_scatter(
    adata_xenium, color='t_abl2', size=20, 
    cmap='RdBu_r', img=False, colorbar=False, return_ax=True, 
    fig=fig, ax=ax, title=r'Inferred spatial gradient - LYNX (off-aligned auxiliary)'
)
sm = ax.collections[0] 
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
cbar.set_label(r'Pseudotime $(t)$', fontsize=8)
plt.show()

# %%
# Benchmark against the ground-truth & full LYNX results
adata_ab = sc.read_h5ad('../results/liver/ab_validation.h5ad')
adata_xenium.obs['t_lynx'] = np.load(f'../results/liver/LYNX_t_new.npy').astype(np.float32)

# %%
# (1). scatter plot & spearman correlation
rand_indices = np.random.choice(adata_xenium.shape[0], size=int(0.1*adata_xenium.shape[0]), replace=False)
fig, ax = plot.disp_kde_scatter(
    adata_ab.obs['t_porto_central'].values, adata_xenium.obs['t_abl1'].values, 
    size=.3, indices=rand_indices, logscale=False, show_plot=False,
    xlabel=r"Antibody-annotated $(t)$",
    ylabel=r"LYNX (w/o U) prediction $(t)$",
    title="Spatial gradient\n LYNX (w/o auxiliary) vs. Ground-truth"
)
ax.plot([0, 1], [0, 1], ':', lw=0.75, color='k', alpha=0.8)
plt.show()

fig, ax = plot.disp_kde_scatter(
    adata_ab.obs['t_porto_central'].values, adata_xenium.obs['t_abl2'].values, 
    size=.3, indices=rand_indices, logscale=False, show_plot=False,
    xlabel=r"Antibody-annotated $(t)$",
    ylabel=r"LYNX (shuffled u) prediction $(t)$",
    title="Spatial gradient\n LYNX (off-aligned U) vs. Ground-truth"
)
ax.plot([0, 1], [0, 1], ':', lw=0.75, color='k', alpha=0.8)
plt.show()

# %%
# (2). RMSE
from util import metrics
from statannotations.Annotator import Annotator

n_repeats = 100
ts = ['t_abl1', 't_abl2', 't_lynx']
methods = ['LYNX\n(w/o auxiliary)', 'LYNX\n(off-aligned auxiliary)', 'LYNX\n(full)']

custom_palette = [
    "#ffe6fc",
    "#d8b5fd",
    '#d400ff'
]

rmses = metrics.compute_rmse(
    adata_xenium, 
    y_true=adata_ab.obs['t_porto_central'].values,
    n_repeats=n_repeats,
    use_rep=['t_abl1', 't_abl2', 't_lynx']
)

plot_df = pd.DataFrame({
    'RMSE': rmses.flatten(),
    'Methods': np.repeat(methods, n_repeats),
})

fig, ax = plt.subplots(figsize=(5, 8), dpi=300)
sns.boxplot(plot_df, x='Methods', y='RMSE', palette=custom_palette, ax=ax)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('Methods\n'+r"($lower\ is\ better$)", fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_title("Root Mean Squared Error\n vs. ground-truth", fontsize=16)

pairs = [
    ('LYNX\n(full)', 'LYNX\n(off-aligned auxiliary)'),
    ('LYNX\n(full)', 'LYNX\n(w/o auxiliary)'),
]
annotator = Annotator(ax, pairs, data=plot_df, x="Methods", y="RMSE")
annotator.configure(test='t-test_ind', text_format='star')
annotator.apply_and_annotate()

plt.show()

# %%

