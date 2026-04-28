# %%
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
r = 50
patience = 20

# Try cleanup xenium data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5_proseg'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
cluster_key = 'subtype'

# Update cell-type annotations
prev_cluster_labels = adata_xenium.obs[cluster_key].cat.categories.to_list()
_cluster_remap = {
    'PC-Hep': 'Hepatocytes',
    'PP-Hep': 'Hepatocytes',
    'Progenitor+Cholangiocytes': 'Cholangiocytes',
    'Endothelial': 'Vascular Endothelial',
    'Inflammatory Monocytes': 'Monocyte-derived macrophages',
    'Generic Fibroblasts': 'Perisinusoidal stroma'
}
adata_xenium.obs[cluster_key] = adata_xenium.obs[cluster_key].map(_cluster_remap).fillna(adata_xenium.obs[cluster_key])
cluster_labels = adata_xenium.obs[cluster_key].cat.categories # Individual cell-types

graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs, 
    r=r, is_weighted=True, alpha=0.5,
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
    infer_cell_interaction=True
)

# %%
# t0 = time.perf_counter()
model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(graph_data, train_configs, DEBUG=True)
res = model.evaluate(
    adata_xenium, adata_desi,
    graph_data=graph_data,
    device=torch.device('cpu')
)

# Save reconstructed gene expressions
adata_xenium.layers['px'] = res['px'].copy()

# t1 = time.perf_counter()
# with open(os.path.join("../results/liver/runtime.txt"), 'a') as f:
#     f.write(f'LYNX training time (s): {t1 - t0:.2f}\n')

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
np.save(os.path.join(outdir, 'LYNX_desi_6_0423.npy'), adata_desi.obsm['X_z'])
adata_xenium.write_h5ad(os.path.join(outdir, 'LYNX_xenium_6_0423.h5ad'))

# %%
curve = trajectory.get_curve(adata_xenium, epg_lambda=0.01, trim_radius_ratio=0.5)
trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

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

curve = trajectory.get_curve(adata_desi, epg_lambda=0.01, trim_radius_ratio=0.5)
trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine [M-H]-')

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

# %%
if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
    sc.pp.normalize_total(adata_xenium)
    sc.pp.log1p(adata_xenium)

utils.get_zonation_features(    
    adata_xenium, 
    adata_desi,
    n_zones=5, sample_id=sample_id,
    abundance_test=True,
    normalize_ref=
    show=False
)

sq.pl.spatial_scatter(
    adata_xenium, color='zone',
    size=25, img=False, palette='Set3'
)

plot.disp_joint_logfc(
    adata_xenium, 
    adata_desi,  
    zones=adata_xenium.obs['zone'].cat.categories.astype('str'),
    title='Representative zone features'
)

# %%
# TODO: debug t-test against abundance?
cci_df = plot.summarize_cell_interaction(
    adata_xenium, 
    cluster_key=cluster_key, 
    cluster_labels=cluster_labels,
    title='Omega',
    show_plot=False
)

abun_df = plot.summarize_cell_interaction(
    adata_xenium,
    cluster_key=cluster_key,
    cluster_labels=cluster_labels,
    ccc_rep='abundance',
    title='Abundance',
    show_plot=False
)

plot.disp_heatmap(
    cci_df, 
    title='Omega'
)

plot.disp_heatmap(
    abun_df,
    title='Abundance'
)

# %%
cci_df = plot.summarize_cell_interaction(
    adata_xenium, 
    cluster_key=cluster_key, 
    cluster_labels=cluster_labels,
    title='Summary of cell-cell interaction (Overall)\n w/o abundance-test',
    show_plot=False
)

cci_df, pval_df = test_assoc.test_cci(
    adata_xenium, cci_df, 
    cluster_key=cluster_key,
    cluster_labels=cluster_labels    
)

plot.disp_heatmap(
    pval_df, 
    title='Summary of cell-cell interaction (Overall)\n -log10(p-val)',
)

# %%
for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_cci_df = plot.summarize_cell_interaction(
        adata_sub, 
        cluster_key=cluster_key,
        cluster_labels=cluster_labels,
        show_plot=False
    )
    
    zone_cci_df, zone_pval_df = test_assoc.test_cci(
        adata_sub, zone_cci_df, 
        cluster_key=cluster_key,
        cluster_labels=cluster_labels,
    )

    plot.netVisual_circle(
        zone_cci_df,
        vertex_size_max=20, figsize=(15, 15),
        title=f'Interaction strength\n (Zone {int(cluster_id)})'
    )

    plot.netVisual_circle(
        zone_pval_df,
        vertex_size_max=20, figsize=(15, 15),
        title=f'Interaction significance\n (Zone {int(cluster_id)})' 
    )


del zone_cci_df# , zone_pval_df
gc.collect()

# %%
