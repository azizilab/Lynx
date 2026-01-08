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
# ---------------------------
#   multi-sample running
# ---------------------------
n_subgraphs = 16
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
        r=r, is_weighted=True,
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
    ) 
    model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
    model.fit(graph_data, train_configs, DEBUG=True)

    # Full inference with best model params
    res = model.evaluate(
        adata_xenium, adata_desi,
        graph_data=graph_data,
        device=torch.device('cpu')
    )

    curve = trajectory.get_curve(adata_xenium, epg_lambda=0.01, trim_radius_ratio=0.5)
    trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')
    curve = trajectory.get_curve(adata_desi, epg_lambda=0.01, trim_radius_ratio=0.5)
    trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine')

    # Visualization checks
    sq.pl.spatial_scatter(
        adata_xenium, color='t', 
        cmap='RdBu_r', size=25, img=False,
        title='Inferred spatial Gradient\nLYNX'
    )
    plot.disp_trajectory(
        adata_xenium, 
        cmap='RdBu_r',
        title='Inferred Spatial Gradient\nLYNX embedding'
    )

    sq.pl.spatial_scatter(
        adata_desi, color='t', 
        cmap='RdBu_r', size=1, img=False,
        title=r'Spatial Gradient $(t)$'+'\nLYNX (DESI)'
    )
    plot.disp_trajectory(
        adata_desi, 
        cmap='RdBu_r',
        title='Spatial Gradients\n LYNX (DESI)'
    )

    if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
        sc.pp.normalize_total(adata_xenium)
        sc.pp.log1p(adata_xenium)

    utils.get_zonation_features(    
        adata_xenium, adata_desi,
        n_zones=3, sample_id=sample_id,
        abundance_test=True,
        show=True
    )

    outdir = '../results/liver/downstream/gradient'
    # np.save(os.path.join(outdir, f'LYNX_{sample_id}_xenium_latent.npy'), adata_xenium.obsm['X_z'])
    # np.save(os.path.join(outdir, f'LYNX_{sample_id}_desi_latent.npy'), adata_desi.obsm['X_z'])
    # np.save(os.path.join(outdir, f'LYNX_{sample_id}_xenium_gradient.npy'), adata_xenium.obs['t'].values)
    # np.save(os.path.join(outdir, f'LYNX_{sample_id}_desi_gradient.npy'), adata_desi.obs['t'].values)
    adata_xenium.write_h5ad(os.path.join(outdir, f'LYNX_{sample_id}_xenium.h5ad'))
    adata_desi.write_h5ad(os.path.join(outdir, f'LYNX_{sample_id}_desi.h5ad'))

    del model, adata_xenium, adata_desi, graph_data
    gc.collect()
    torch.cuda.empty_cache()

# %%
