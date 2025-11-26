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
# Evaluation: Reconstruction
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
outdir = '../results/liver/'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

# Save the latent embedding
np.save(os.path.join(outdir, 'LYNX_xenium_6_debug.npy'), adata_xenium.obsm['X_z'])
np.save(os.path.join(outdir, 'LYNX_desi_6_debug.npy'), adata_desi.obsm['X_z'])
np.save(os.path.join(outdir, 'LYNX_t_debug.npy'), adata_xenium.obs['t'].values)
adata_xenium.write_h5ad(os.path.join(outdir, 'LYNX_xenium_6_debug.h5ad'))

# %%