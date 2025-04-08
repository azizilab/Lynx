# %%
# Running LYNX on simulation data

import os
import gc
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

import pyro
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from IPython.display import display

sns.set_context('paper')
rcParams.update({'font.family': 'Liberation Sans'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 100})
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
%reload_ext autoreload

# %%
# -------------
#  Load data
# -------------

# Dataset specs
n_subgraphs = 16
k = 20
r = 50
sigma = 20

# Simulation
data_path = '../data/simulation'
adata_desi = sc.read_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'xenium_refit_feature_matrix.h5'))
adata_xenium.obs['leiden'], categories = adata_xenium.obs.cell_type.factorize()
categories = categories.values

graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs, 
    k=k,
    r=r,
    is_weighted=True,
    sigma=sigma,
    verbose=True
)

train_data, val_data = random_split(graph_data, [0.7, 0.3])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

# %%
# -----------------------------
#  Model training & inference
# -----------------------------

pyro.clear_param_store()
torch.cuda.empty_cache()

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 400
lr = 1e-2
patience = 20

# Training & Inference
train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience, anneal=True,
    device=torch.device('cuda'),
)

model_configs = configs.set_model_configs(
    c_in=adata_xenium.shape[1],   # ref-dim 
    c_aux=adata_desi.shape[1],   # query-dim
    c_hidden=n_hidden, 
    c_latent=n_latent,
    act=nn.SiLU(),
    ref=graph_data.ref, 
    query=graph_data.query,
    num_clusters=graph_data.num_clusters,
    gene_symbols = adata_xenium.var_names
) 

model = vgae.HeteroVGAE(model_configs, device=torch.device('cuda'))
model.fit(train_configs, train_dl=train_dl, val_dl=val_dl, DEBUG=True)

res = model.evaluate(
    adata_xenium, adata_desi, 
    graph_data=graph_data,
    device=torch.device('cpu')
)


# %%
# np.save('../results/simulation/lynx_6_desi.npy', res.qzu)
# np.save('../results/simulation/lynx_6_xenium.npy', res.qzx)
# adata_desi.obs['t'].to_csv('../results/simulation/lynx_desi_pseudotime.csv', index=True)
# adata_xenium.obs['t'].to_csv('../results/simulation/lynx_xenium_pseudotime.csv', index=True)

# %%
# -------------
#  Evaluation
# -------------

from scipy.special import comb
def _convert_gradients(gradients):
    """TMP: convert ground-truth gradients to 0-1"""
    v = gradients + gradients.min()
    return (v-v.min()) / (v.max()-v.min())

def plot_factor_corr(z):
    z_corr = np.corrcoef(z.T)
    z_score = np.abs(np.tril(z_corr, k=-1)).sum() / comb(z_corr.shape[0], 2)

    g = sns.clustermap(z_corr, cmap='RdBu_r')
    g.figure.suptitle(
        'q(z)\n Correlation score: {}'.format(np.round(z_score, 3)), 
        fontsize=30, y=1.05
    )
    plt.show()

# %%
plot_factor_corr(res.qzu)

# %%
# (1). Observation
rand_indices = np.random.choice(
    np.arange(adata_xenium.shape[0]*adata_xenium.shape[1]), 10000, replace=False
)
plot.disp_kde_scatter(
    adata_xenium.X.flatten()[rand_indices],
    res.px.flatten()[rand_indices],
    xlabel=r"Ground-truth observation",
    ylabel=r"Reconstructed observation",
    title='Xenium feature reconstruction'
)
del rand_indices
gc.collect()

# %%
# (2). Trajectory inference
# Low-dim gradients (u)
adata_desi.obsm['X_z_lynx'] = res.qzu
trajectory.compute_trajectory(
    adata_desi, 
    use_rep='X_z_lynx',
    n_neighbors=100,
    root_marker='Taurine '
)

sq.pl.spatial_scatter(
    adata_desi, color='t', 
    cmap='RdBu_r', size=1, img=False,
    title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (DESI)'
)

# High-dim gradients (x)
adata_xenium.obsm['X_z_lynx'] = res.qzx
trajectory.compute_trajectory(
    adata_xenium, 
    use_rep='X_z_lynx',
    n_neighbors=100,
    root_marker='DPT'
)

sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu_r', size=20, img=False,
    title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (Simulation)'  # Xenium
)


# %%
plot.disp_trajectory(
    adata_desi, cmap='RdBu_r',
    title='Spatial Gradients\n LYNX (DESI)'
)

# %%
# Zonation
utils.get_zonation_features(
    adata_xenium, adata_desi, n_zones=4, option='piecewise', show=True
)

sq.pl.spatial_scatter(
    adata_xenium, color='zone',
    cmap='turbo', size=20, img=False,
    title='Zonation \nLYNX'
)


# %%
# =====================================
# Diagram purpose plots

adata_xenium_gt = adata_xenium.copy()
adata_xenium_gt.obs['t'] = adata_xenium.obs['t_gt'].values.copy()
adata_xenium_gt.obsm['X_z'] = adata_xenium.obsm['z_gt'].copy()

adata_desi_gt = adata_desi.copy()
adata_desi_gt.obs['t'] = adata_desi.obs['t_gt'].values.copy()
adata_desi_gt.obsm['X_z'] = adata_desi.obsm['z_refit'].copy()

# %%
utils.get_zonation_features(
    adata_xenium_gt, adata_desi_gt, n_zones=4, option='piecewise', show=False
)

# %%
sq.pl.spatial_scatter(
    adata_desi_gt, color='t',
    cmap='RdBu_r', size=1, img=False,
    title='Zonation \nLYNX'
)

sq.pl.spatial_scatter(
    adata_desi_gt, color='zone',
    cmap='turbo', size=1, img=False,
    title='Zonation \nLYNX'
)

# %%
sc.pp.normalize_total(adata_xenium_gt)
sc.pp.log1p(adata_xenium_gt)

# %%
adata_xenium_gt.uns['zones'] = {}
zone_labels = np.unique(adata_xenium_gt.obs['zone'])
for label in zone_labels:
    sc.tl.rank_genes_groups(
        adata_xenium_gt, groupby='zone',
        method='wilcoxon'
    )
    df = sc.get.rank_genes_groups_df(adata_xenium_gt, group=str(label))
    df = df.sort_values('scores', ascending=False).reset_index(drop=True)
    adata_xenium_gt.uns['zones'][str(label)] = df

markers = []
for label in zone_labels:
    zone_markers = adata_xenium_gt.uns['zones'][str(label)].iloc[:3, 0].values
    markers.extend(zone_markers)

sc.pl.matrixplot(
    adata_xenium_gt, markers, groupby='zone', cmap='RdBu_r', standard_scale='var'
)

# %%
# =====================================

# %%
# (3). Comparison w/ ground-truth trajectory gradients (\gamma(t)) (simulation-only)
t_gt = adata_desi.obs['t_gt'].values
t_gt = (t_gt-t_gt.min()) / (t_gt.max()-t_gt.min())
t_lynx = adata_desi.obs['t'].values

plot.disp_kde_scatter(
    t_gt, t_lynx,
    xlabel=r"Ground-truth $(t)$",
    ylabel=r"LYNX prediction $(t)$",
    title="Trajectory pseudotime"
)

# %%
# (4). Latent disentanglement measure
# Check MCC (true disentanglement score)
import numpy as np
from scipy.optimize import linear_sum_assignment

def mean_corr_coef_np(x, y):
    """
    # Reference: https://github.com/siamakz/iVAE/blob/master/lib/metrics.py
    """
    d = x.shape[1]
    cc = np.abs(np.corrcoef(x, y, rowvar=False)[:d, d:])
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score

print(
    'MCC (Lynx z vs. ground-truth z):', 
    mean_corr_coef_np(adata_desi.obsm['z_refit'], adata_desi.obsm['X_z_lynx'])
)

# %%
# UMAP + spatial plots of individual q(z) & ground-truth z's
z_labels = ['z'+str(i) for i in range(n_latent)]

# UMAP plots
adata_desi.obs[z_labels] = adata_desi.obsm['X_z_lynx'].copy()
sc.pp.neighbors(adata_desi, n_neighbors=k, use_rep='X_z_lynx')
sc.tl.umap(adata_desi)
sc.pl.umap(adata_desi, color=z_labels, cmap='turbo', ncols=2)
adata_desi.obs.drop(z_labels, axis=1, inplace=True)

# Spatial plot (inferred zs vs. ground-truth zs)
z_labels = ['z'+str(i) for i in range(n_latent)]
for label, zi in zip(z_labels, adata_desi.obsm['X_z_lynx'].T):
    adata_desi.obs[label] = zi
del label, zi

sq.pl.spatial_scatter(
    adata_desi, color=z_labels, img=False, size=1, cmap='turbo', ncols=2
)
adata_desi.obs.drop(z_labels, axis=1, inplace=True)
plt.show()


z_labels = ['z'+str(i) for i in range(n_latent)]
for label, zi in zip(z_labels, adata_desi.obsm['z_refit'].T):
    adata_desi.obs[label] = zi
del label, zi

sq.pl.spatial_scatter(
    adata_desi, color=z_labels, img=False, size=1, cmap='turbo', ncols=2
)
adata_desi.obs.drop(z_labels, axis=1, inplace=True)
plt.show()

# %%
sns.clustermap(np.corrcoef(res.qzu.T), cmap='RdBu_r')
