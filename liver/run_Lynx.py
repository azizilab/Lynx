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

# %%
# Dataset specs
n_subgraphs = 16
k = 20
r = 50

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500
lr = 1e-3
patience = 50

# Real data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))

# Preprocess, add cell-type labels in integers
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
cluster_key = 'cell_type' if 'cell_type' in adata_xenium.obs.keys() else None

# %%
graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs, 
    k=100,
    r=r,
    cluster_key=cluster_key,
    is_weighted=True
)

train_data, val_data = random_split(graph_data, [0.8, 0.2])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

# Training & Inference
train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience, 
    device=torch.device('cuda'),
    anneal=True,
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
    k_hop=1,
    num_heads=1,
    num_clusters=graph_data.num_clusters,
    verbose=True
) 

# %%
# del model
pyro.clear_param_store()
torch.cuda.empty_cache()
reload(vgae)

# %%
model = vgae.HeteroVGAE(model_configs, device=torch.device('cuda'))
model.fit(train_configs, train_dl=train_dl, val_dl=val_dl, DEBUG=True)
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
plot.disp_factor_corr(res.qzx)
plot.disp_spatial_latents(adata_xenium, res.qzx, ncols=3)

sc.pp.normalize_total(adata_xenium)
sc.pp.log1p(adata_xenium)
utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=5, sample_id=sample_id,
    show=True
)
