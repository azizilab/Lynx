# %%
# -------------------------------------------------------------------
# Ablation study — Mouse Thymus (Mouse_Thymus1)
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

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util/')

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
n_subgraphs  = 16
k            = 8
n_hidden     = 32
n_latent     = 6
n_epochs     = 500
lr           = 1e-2
patience     = 20
DEVICE       = torch.device('cuda')
OUTDIR       = '../results/thymus/ablation/'
os.makedirs(OUTDIR, exist_ok=True)

# %%
# Load data
data_path = '../data/thymus/'
sample_id = 'Mouse_Thymus1'
cluster_key = 'cell_type'

adata_rna     = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))
adata_protein.var_names_make_unique()
cluster_key = cluster_key if cluster_key in adata_rna.obs.keys() else None

# %%
# Data construction
graph_data = HeteroDataset(
    adatas_ref=adata_rna,
    adatas_query=adata_protein,
    n_subgraphs=n_subgraphs,
    k=k, is_weighted=True,
    cluster_key=cluster_key,
    is_query_grid=True,
    is_ref_grid=True,
    query='protein', query_proj_key='spatial',
    ref='rna',       ref_proj_key='spatial'
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
    act=nn.SiLU()
)

model_abl1 = SpatialVGAE(model_configs_abl1, device=DEVICE)
model_abl1.fit(graph_data, train_configs, DEBUG=True)

res_abl1 = model_abl1.evaluate(
    adata_rna, adata_protein,
    graph_data=graph_data,
    device=torch.device('cpu')
)

# %%
# Reconstruction quality
plot.disp_kde_scatter(
    adata_rna.X.flatten(),
    res_abl1.px.flatten(),
    subset_ratio=0.001,
    xlabel=r'Observation $\log(1+x)$',
    ylabel=r'Reconstruction $\log(1+x)$',
    title='Feature reconstruction — SpatialVGAE (w/o auxiliary)'
)
gc.collect()

# %%
# Principal curve + pseudotime — RNA (SpatialVGAE)
curve_abl1 = trajectory.get_curve(adata_rna, trim_radius_ratio=0.25)
trajectory.compute_pseudotime(adata_rna, curve_abl1, root_marker='Dcn')

sq.pl.spatial_scatter(
    adata_rna, color='t',
    cmap='RdBu_r', size=100, img=False,
    title='Inferred Spatial Gradient\nSpatialVGAE (w/o auxiliary)'
)

plot.disp_trajectory(
    adata_rna,
    cmap='RdBu_r',
    title='Inferred Spatial Gradient\nSpatialVGAE (w/o auxiliary) embedding'
)

# Stash pseudotime before ablation 2 overwrites X_z / t
adata_rna.obs['t_abl1'] = adata_rna.obs['t'].copy()
adata_rna.obsm['X_z_abl1'] = adata_rna.obsm['X_z'].copy()

# %%
# Save embeddings
# np.save(os.path.join(OUTDIR, 'SpatialVGAE_rna_z.npy'), adata_rna.obsm['X_z_abl1'])

# %%
# ======================================================================
# ABLATION 2 — LYNX with randomized u  (randomized query coordinates)
# ======================================================================
rng = np.random.default_rng(42)
adata_protein_shuffled = adata_protein.copy()
rand_map_coords = rng.permutation(adata_protein.obsm['spatial'], axis=0)
adata_protein_shuffled.obsm['spatial'] = rand_map_coords

# %%
graph_data_shuffled = HeteroDataset(
    adatas_ref=adata_rna,
    adatas_query=adata_protein_shuffled,
    n_subgraphs=n_subgraphs,
    k=k, is_weighted=True,
    cluster_key=cluster_key,
    is_query_grid=True,
    is_ref_grid=True,
    query='protein', query_proj_key='spatial',
    ref='rna',       ref_proj_key='spatial'
)

model_configs_abl2 = configs.set_model_configs(
    graph_data=graph_data_shuffled,
    c_hidden=n_hidden,
    c_latent=n_latent,
    act=nn.SiLU()
)
model_abl2 = vgae.HeteroAttnVGAE(model_configs_abl2, device=DEVICE)
model_abl2.fit(graph_data_shuffled, train_configs, DEBUG=True)

res_abl2 = model_abl2.evaluate(
    adata_rna, adata_protein_shuffled,
    graph_data=graph_data_shuffled,
    device=torch.device('cpu')
)

# %%
# Reconstruction quality
plot.disp_kde_scatter(
    adata_rna.X.flatten(),
    res_abl2.px.flatten(),
    subset_ratio=0.001,
    xlabel=r'Observation $\log(1+x)$',
    ylabel=r'Reconstruction $\log(1+x)$',
    title='Feature reconstruction — LYNX (off-aligned coordinates)'
)
gc.collect()

# %%
curve_abl2_rna = trajectory.get_curve(adata_rna, trim_radius_ratio=0.25)
trajectory.compute_pseudotime(adata_rna, curve_abl2_rna, root_marker='Dcn')

sq.pl.spatial_scatter(
    adata_rna, color='t',
    cmap='RdBu_r', size=100, img=False,
    title='Inferred Spatial Gradient\nLYNX (off-aligned auxiliary)'
)

plot.disp_trajectory(
    adata_rna,
    cmap='RdBu_r',
    title='Inferred Spatial Gradient\nLYNX (off-aligned auxiliary) embedding'
)

adata_rna.obs['t_abl2'] = adata_rna.obs['t'].copy()
adata_rna.obsm['X_z_abl2'] = adata_rna.obsm['X_z'].copy()

# %%
# Save embeddings
# np.save(os.path.join(OUTDIR, 'RandomU_rna_z.npy'), adata_rna.obsm['X_z_abl2'])

# %%
# Visualizations
fig, ax = plt.subplots()
ax = sq.pl.spatial_scatter(
    adata_rna, color='t_abl1', size=100,
    cmap='RdBu_r', img=False, colorbar=False, return_ax=True,
    fig=fig, ax=ax, title=r'Inferred spatial gradient - LYNX (no auxiliary)'
)
sm = ax.collections[0]
cbar = plt.colorbar(sm, ax=ax, aspect=20)
cbar.set_label(r'Pseudotime $(t)$', fontsize=8)
plt.show()

fig, ax = plt.subplots()
ax = sq.pl.spatial_scatter(
    adata_rna, color='t_abl2', size=100,
    cmap='RdBu_r', img=False, colorbar=False, return_ax=True,
    fig=fig, ax=ax, title=r'Inferred spatial gradient - LYNX (off-aligned auxiliary)'
)
sm = ax.collections[0]
cbar = plt.colorbar(sm, ax=ax, aspect=20)
cbar.set_label(r'Pseudotime $(t)$', fontsize=8)
plt.show()

# %%
# Benchmark against the ground-truth & full LYNX results
t_cma = adata_rna.obs['CMA'].copy()
adata_rna.obs['CMA'] = (t_cma - t_cma.min()) / (t_cma.max() - t_cma.min())

n_latent = 6
adata_rna.obsm['X_z'] = sc.read_h5ad(f'../results/thymus/lynx_rna_{n_latent}_{sample_id}.h5ad').obsm['X_z'].copy()
curve = trajectory.get_curve(adata_rna, trim_radius_ratio=0.25)
trajectory.compute_pseudotime(adata_rna, curve, root_marker='Dcn')
adata_rna.obs['t_lynx'] = adata_rna.obs['t'].copy()

# %%
# (1). scatter plot & spearman correlation
rand_indices = np.random.choice(adata_rna.shape[0], size=int(0.1*adata_rna.shape[0]), replace=False)
fig, ax = plot.disp_kde_scatter(
    adata_rna.obs['CMA'].values, adata_rna.obs['t_abl1'].values,  
    size=.3, indices=rand_indices, logscale=False, show_plot=False,
    xlabel=r"Ground-truth $(t)$",
    ylabel=r"LYNX (w/o auxiliary) prediction $(t)$",
    title="Spatial gradient\n LYNX (w/o auxiliary) vs. Ground-truth"
)
ax.plot([0, 1], [0, 1], ':', lw=0.75, color='k', alpha=0.8)
plt.show()

fig, ax = plot.disp_kde_scatter(
    adata_rna.obs['CMA'].values, adata_rna.obs['t_abl2'].values,  
    size=.3, indices=rand_indices, logscale=False, show_plot=False,
    xlabel=r"Ground-truth $(t)$",
    ylabel=r"LYNX (off-aligned auxiliary) prediction $(t)$",
    title="Spatial gradient\n LYNX (off-aligned auxiliary) vs. Ground-truth"
)
ax.plot([0, 1], [0, 1], ':', lw=0.75, color='k', alpha=0.8)
plt.show()

# %%
# (2). RMSE
import metrics
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
    adata_rna,
    y_true=adata_rna.obs['CMA'].values,
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
