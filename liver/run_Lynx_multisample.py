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
patience = 20

# %%
# Try joint training on multiple samples
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
adatas_xenium = []
adatas_desi = []
common_genes = []
common_molecules = []

for sample_id in sample_ids:
    adata_xenium = IO.load_xenium(
        os.path.join(xenium_path, sample_id), 
        load_img=False
    )
    adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
    adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
    adatas_xenium.append(adata_xenium)
    adatas_desi.append(adata_desi)

    # Filter common features
    if len(common_genes) == 0:
        common_genes.extend(adata_xenium.var_names.tolist())
    else:
        common_genes = np.intersect1d(
            common_genes, adata_xenium.var_names.tolist()
        ).tolist()
    
    if len(common_molecules) == 0:
        common_molecules.extend(adata_desi.var_names.tolist())
    else:
        common_molecules = np.intersect1d(
            common_molecules, adata_desi.var_names.tolist()
        ).tolist()

adatas_xenium = [
    adata[:, common_genes] 
    for adata in adatas_xenium
]
adatas_desi = [
    adata[:, common_molecules] 
    for adata in adatas_desi
]

# %%
graph_data = dataset.HeteroDataset(
    adatas_ref=adatas_xenium, 
    adatas_query=adatas_desi,
    n_subgraphs=n_subgraphs, 
    r=r, is_weighted=True,
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
    infer_cell_interaction=False,
) 
model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(graph_data, train_configs, DEBUG=True)
gc.collect()


# %%    
# Full inference with best model params
outdir = '../results/liver/downstream/gradient'
for i, sample_id in enumerate(sample_ids):
    print(f'Inference on sample ID: {sample_id} ...')
    adata_xenium = adatas_xenium[i]
    adata_desi = adatas_desi[i]

    res = model.evaluate(
        adata_xenium, adata_desi,
        graph_data=graph_data,
        device=torch.device('cpu')
    )

    # Save reconstrcuted gene expressions
    adata_xenium.layers['px'] = res['px'].copy()

    # adata_xenium = sc.read_h5ad(os.path.join(
    #     outdir, f'LYNX_{sample_id}_xenium_proseg.h5ad'
    # ))
    # adata_desi = sc.read_h5ad(os.path.join(
    #     outdir, f'LYNX_{sample_id}_desi_proseg.h5ad'
    # ))

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

    # adata_xenium.write_h5ad(os.path.join(outdir, f'LYNX_{sample_id}_xenium.h5ad'))
    # adata_desi.write_h5ad(os.path.join(outdir, f'LYNX_{sample_id}_desi.h5ad'))

    del adata_xenium, adata_desi
    gc.collect()
    torch.cuda.empty_cache()

# %%
