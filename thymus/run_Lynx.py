# Running LYNX on multi-modal stereo-seq Mouse Thymus to infer Medulla - Cortex-Capsule axis

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

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

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
n_subgraphs = 27
k = 8  # grid graph

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500 # Debug collapsed latent
lr = 1e-2
patience = 20

data_path = '../data/thymus/'
sample_id = 'Mouse_Thymus1'

adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))
adata_protein.var_names_make_unique()
cluster_key = 'cell_type' if 'cell_type' in adata_rna.obs.keys() else None

# %%
graph_data = dataset.HeteroDataset(
    adatas_ref=adata_rna, 
    adatas_query=adata_protein,
    n_subgraphs=n_subgraphs, 
    k=k, is_weighted=True,
    cluster_key=cluster_key,
    
    # Update modality labels
    query='protein', query_proj_key='spatial',
    ref='rna', ref_proj_key='spatial'
)
train_data, val_data = random_split(graph_data, [0.7, 0.3])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

# Training & Inference
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
    infer_cell_interaction=False
) 

model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(train_configs, train_dl=train_dl, val_dl=val_dl, DEBUG=True)
res = model.evaluate(
    adata_rna, adata_protein,
    graph_data=graph_data,
    n_subgraphs=1,
    device=torch.device('cpu')
)

# %%
# (i). Reconstruction
plot.disp_kde_scatter(
    adata_rna.X.flatten(),
    res.px.flatten(),
    xlabel=r"Observation log(x+1)",
    ylabel=r"Reconstruction log(x+1)",
    title='Stereo-seq feature reconstruction'
)
gc.collect()

# %%
sc.pp.neighbors(adata_rna, use_rep='X_z')
sc.tl.umap(adata_rna)
sc.pl.umap(adata_rna, color='leiden')

# %%
# (ii). Spatial trajectory
# Load from results
n_latent = 6
adata_rna.obsm['X_z'] = np.load('../results/thymus/lynx_rna_{0}_{1}.npy'.format(n_latent, sample_id))
adata_protein.obsm['X_z'] = adata_rna.obsm['X_z'].copy()

curve = trajectory.get_curve(adata_rna)
trajectory.compute_pseudotime(adata_rna, curve, root_marker='Dcn')
adata_protein.obs['t'] = adata_rna.obs['t'].values

# ax = sq.pl.spatial_scatter(
#     adata_rna, color='t', 
#     cmap='RdBu_r', size=100, img=False, return_ax=True,
#     title=r'Spatial Trajectory ($\gamma(t)$)'+'\nLYNX (RNA)'
# )
# ax.set_title(r'Inferred spatial gradient $(t)$ - LYNX', fontsize=14)


# plot.disp_trajectory(
#     adata_rna, cmap='RdBu',
#     title='Principal Curve - LYNX'
# )

# # %%
# # sc.pp.normalize_total(adata_rna)
# # sc.pp.log1p(adata_rna)
# if 'milestones_colors' in adata_rna.uns_keys():
#     adata_rna.uns.pop('milestones_colors')

# utils.get_zonations(adata_rna, n_zones=4) 
# sq.pl.spatial_scatter(
#     adata_rna, color='zone', 
#     size=100, img=False,
#     title='Spatial clustering'+'\nLYNX (RNA)'
# )

# %%
# Save z
np.save('../results/thymus/lynx_rna_6_{}_new.npy'.format(sample_id), adata_rna.obsm['X_z'])
# np.save('../results/thymus/lynx_protein_6_{}.npy'.format(sample_id), adata_protein.obsm['X_z'])

# %% 
# Compare w/ ground-truth CMA
t_lynx = adata_rna.obs['t'].values
t_true = adata_protein.obs['CMA'].values
t_true = (gamma_true-gamma_true.min()) / (gamma_true.max()-gamma_true.min())

plot.disp_kde_scatter(
    gamma_true, gamma_lynx, subset_ratio=0.1,
    logscale=False,
    xlabel=r"Ground-truth $(t)$",
    ylabel=r"LYNX prediction $(t)$",
    title="CMA\n LYNX vs. Ground-truth"
)

# %%
# Proof of concept validation: not biased by library size
adata_rna.obs['library_size'] = adata_rna.X.sum(1)

plot.disp_kde_scatter(
    gamma_lynx, -adata_rna.obs['library_size'].values, 
    logscale=False, subset_ratio=0.01,
    xlabel=r"LYNX prediction $\gamma(t)$",
    ylabel=r"-library size",
    title="LYNX vs. library size"
)

sq.pl.spatial_scatter(
    adata_rna, color='library_size', 
    size=100, img=False, cmap='RdBu_r',
    title='Library size'
)
