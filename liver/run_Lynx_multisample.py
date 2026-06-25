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
import IO, plot, utils, trajectory
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
patience = 50

# %%
# Process each sample individually
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_ids = [
    'NIH_F2_proseg',
    'NIH_F3_proseg',
    'NIH_F4_proseg',
    'NIH_F5_proseg',
    'NIH_M1_proseg',
    'NIH_M2_proseg',
    'NIH_M3_proseg',
    'NIH_M4_proseg',
    'NIH_M5_proseg'
]

cluster_key = 'subtype'
outdir = '../results/liver/downstream/gradient'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

for sample_id in sample_ids:
    print(f'Processing sample ID: {sample_id} ...')

    # ---------------------------
    #   Load data
    # ---------------------------
    adata_xenium = IO.load_xenium(
        os.path.join(xenium_path, sample_id),
        load_img=False
    )
    adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
    adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
    cluster_labels = adata_xenium.obs[cluster_key].cat.categories # Individual cell-types

    # ---------------------------
    #   Run LYNX
    # ---------------------------
    graph_data = dataset.HeteroDataset(
        adatas_ref=adata_xenium,
        adatas_query=adata_desi,
        n_subgraphs=n_subgraphs,
        r=r, is_weighted=True,
        alpha=0.5,
        cluster_key=cluster_key
    )

    train_configs = configs.set_train_configs(
        n_epochs=n_epochs,
        lr=lr, patience=patience,
        device=torch.device('cuda')
    )

    model_configs = configs.set_model_configs(
        graph_data=graph_data,
        c_hidden=n_hidden,
        c_latent=n_latent,
        act=nn.SiLU(),
        infer_cell_interaction=True, # TODO: test stability & zone 2 vs. 3 w/ CCI?
    )
    model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
    model.fit(graph_data, train_configs, DEBUG=True)

    res = model.evaluate(
        adata_xenium, adata_desi,
        graph_data=graph_data,
        device=torch.device('cpu')
    )

    # Save reconstructed gene expressions
    adata_xenium.layers['px'] = res['px'].copy()

    # ---------------------------
    #   Infer spatial gradient
    # ---------------------------
    curve = trajectory.get_curve(adata_xenium, epg_lambda=0.1, trim_radius_ratio=0.5)
    trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')
    curve = trajectory.get_curve(adata_desi, epg_lambda=0.1, trim_radius_ratio=0.5)
    trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine [M-H]-')

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

    # ---------------------------
    #   Infer discrete zones
    # ---------------------------
    if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
        sc.pp.normalize_total(adata_xenium)
        sc.pp.log1p(adata_xenium)

    n_zones = 4
    utils.get_zonation_features(
        adata_xenium,
        adata_desi,
        n_zones=n_zones, sample_id=sample_id,
        abundance_test=True,
        show=False
    )

    set3_cmap = plt.cm.get_cmap('Set3', n_zones+1)
    zone_colors = [set3_cmap(i) for i in range(n_zones)]
    zone_cmap = plt.cm.colors.ListedColormap(zone_colors)
    adata_xenium.uns['zone_colors'] = zone_colors
    sq.pl.spatial_scatter(
        adata_xenium, color='zone', title='LYNX inferred zones',
        size=25, img=False,
    )

    # ---------------------------
    #   Save h5ad individually
    # ---------------------------
    adata_xenium.write_h5ad(os.path.join(outdir, f'LYNX_{sample_id}_xenium_wo_cci.h5ad'))
    adata_desi.write_h5ad(os.path.join(outdir, f'LYNX_{sample_id}_desi_wo_cci.h5ad'))

    del adata_xenium, adata_desi, graph_data, model
    gc.collect()
    torch.cuda.empty_cache()

