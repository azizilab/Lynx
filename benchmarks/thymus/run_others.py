# Benchmark with traditional trajectory inference methods

# %%
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
import torch.nn.functional as F
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
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

# %%
sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')

import IO, plot, utils, trajectory
import vgae, configs, dataset

# %%
%load_ext autoreload
%autoreload 2

# %%
# Load dataset
data_path = '../data/thymus/'
sample_ids = sorted([
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
])

sample_id = sample_ids[0]
adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))
adata_protein.var_names_make_unique()

# %%
# Run PCA & Diffmap
sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
sc.pp.pca(adata_rna)
sc.pp.neighbors(adata_rna)
sc.tl.diffmap(adata_rna)

# %%
# DPT
adata_rna.uns['iroot'] = adata_rna[:, 'Dcn'].X.argmax()
sc.tl.dpt(adata_rna)

t_dpt = adata_rna.obs['dpt_pseudotime'].copy()
t_dpt = (t_dpt-t_dpt.min()) / (t_dpt.max()-t_dpt.min())
adata_rna.obs['t_dpt'] = t_dpt
adata_rna.obs.drop('dpt_pseudotime', inplace=True, axis=1)

# %%
# TODO: hyperparameter tuning for Elpigraph
curve = trajectory.get_curve(adata_rna, use_rep='X_pca')
trajectory.compute_pseudotime(adata_rna, curve, root_marker='Dcn')
adata_rna.obs['t_pca'] = adata_rna.obs['t'].copy()
adata_rna.obs.drop(['t', 'seg', 'edge', 'milestones'], inplace=True, axis=1)

# %%
curve = trajectory.get_curve(adata_rna, use_rep='X_diffmap')
trajectory.compute_pseudotime(adata_rna, curve, root_marker='Dcn')
adata_rna.obs['t_diffmap'] = adata_rna.obs['t'].copy()
adata_rna.obs.drop(['t', 'seg', 'edge', 'milestones'], inplace=True, axis=1)

# %%
ax = sq.pl.spatial_scatter(
    adata_rna, color='t_pca', 
    cmap='RdBu_r', size=100, img=False, return_ax=True
)
ax.set_title(r'Inferred spatial gradient $(t)$ - PCA', fontdict={'fontsize': 14})

ax = sq.pl.spatial_scatter(
    adata_rna, color='t_diffmap', 
    cmap='RdBu_r', size=100, img=False, return_ax=True
)
ax.set_title(r'Inferred spatial gradient $(t)$ - Diffmap', fontdict={'fontsize': 14})

ax = sq.pl.spatial_scatter(
    adata_rna, color='t_dpt', 
    cmap='RdBu_r', size=100, img=False, return_ax=True
)
ax.set_title(r'Inferred spatial gradient $(t)$ - DPT', fontdict={'fontsize': 14})

# %%
gamma_true = adata_protein.obs['CMA'].values
gamma_true = (gamma_true-gamma_true.min()) / (gamma_true.max()-gamma_true.min())

# Plot ground-truth
adata_rna.obs['CMA'] = gamma_true.copy()
ax = sq.pl.spatial_scatter(
    adata_rna, color='CMA', 
    cmap='RdBu_r', size=100, img=False, return_ax=True
)
ax.set_title('Ground-truth Cortical-Medullary Axis (CMA)', fontdict={'fontsize': 14})

ax = sq.pl.spatial_scatter(
    adata_rna, color='CML_Major', 
    cmap='RdBu_r', size=100, img=False, return_ax=True
)
ax.set_title('Ground-truth lobule layers', fontdict={'fontsize': 14})

# %%
plot.disp_kde_scatter(
    gamma_true, adata_rna.obs['t_pca'].values, ss_ratio=1.,
    xlabel=r"Ground-truth $\gamma(t)$",
    ylabel=r"PCA prediction $\gamma(t)$",
    title="CMA\n PCA vs. Ground-truth"
)

plot.disp_kde_scatter(
    gamma_true, adata_rna.obs['t_diffmap'].values, ss_ratio=1.,
    xlabel=r"Ground-truth $\gamma(t)$",
    ylabel=r"TotalVI prediction $\gamma(t)$",
    title="CMA\n Diffmap vs. Ground-truth"
)

plot.disp_kde_scatter(
    gamma_true, adata_rna.obs['t_dpt'].values, ss_ratio=1.,
    xlabel=r"Ground-truth $\gamma(t)$",
    ylabel=r"TotalVI prediction $\gamma(t)$",
    title="CMA\n DPT vs. Ground-truth"
)
