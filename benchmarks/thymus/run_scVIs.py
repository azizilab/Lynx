# Benchmark with scVI & TotalVI

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
sys.path.append('../../')
sys.path.append('../../models/')
sys.path.append('../../util')

import IO, plot, utils, trajectory
import vgae, configs, dataset

# %%
%load_ext autoreload
%autoreload 2

# %%
# Load dataset
data_path = '../../data/thymus/'
sample_ids = sorted([
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
])

sample_id = sample_ids[0]
adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))
adata_protein.var_names_make_unique()

# %%
# -------------
#  (1). scVI
# -------------
import scvi
scvi.settings.seed = 42

# %%
# Setup scVI
scvi.model.SCVI.setup_anndata(adata_rna)

model = scvi.model.SCVI(
    adata_rna, 
    n_layers=2, 
    gene_likelihood="nb"
)
model.train()

# %%
# Retrieve latent representation
# latent = model.get_latent_representation()
# np.save('../results/thymus/scvi_10.npy', latent)
# adata_rna.obsm['X_scvi'] = latent

adata_rna.obsm['X_scvi'] = np.load('../../results/thymus/scvi_10.npy')

# %%
curve = trajectory.get_curve(adata_rna, use_rep='X_scvi')
trajectory.compute_pseudotime(adata_rna, curve, root_marker='Dcn')

ax = sq.pl.spatial_scatter(
    adata_rna, color='t', 
    cmap='RdBu_r', size=100, img=False, return_ax=True
)
ax.set_title(r'Inferred spatial gradient $(t)$ - scVI', fontdict={'fontsize': 14})

# plot.disp_trajectory(
#     adata_rna, cmap='RdBu', use_rep='X_scvi',
#     title='Principal Curve - scVI'
# )

# %%
# Compare w/ ground-truth CMA
gamma_scvi = adata_rna.obs['t'].values
gamma_true = adata_rna.obs['CMA'].values
gamma_true = (gamma_true-gamma_true.min()) / (gamma_true.max()-gamma_true.min())

plot.disp_kde_scatter(
    gamma_true, gamma_scvi, ss_ratio=1.,
    xlabel=r"Ground-truth $(t)$",
    ylabel=r"scVI prediction $(t)$",
    title="CMA\n scVI vs. Ground-truth"
)

# %%
# ----------------
#  (2). TotalVI
# ----------------

# %%
import muon
import mudata as md

# Create joint-modality data, note: totalVI requires "unnormlized" 2nd-modality intensities
adata_protein.X = (adata_protein.X - adata_protein.X.min()) / (adata_protein.X.max() - adata_protein.X.min())
adata_protein.X = (255*adata_protein.X).astype(np.uint8)
mdata = md.MuData({'rna': adata_rna, 'protein': adata_protein})
mdata

# %%
# Setup TotalVI
scvi.model.TOTALVI.setup_mudata(
    mdata,
    rna_layer=None,
    protein_layer=None,
    modalities={
        "rna_layer": "rna",
        "protein_layer": "protein",
    },
)

model = scvi.model.TOTALVI(mdata)  # n_latent defaults to 20
model.train()

# %%
# Retrieve joint latent representation
# adata_rna.obsm['X_totalvi'] = model.get_latent_representation()
# np.save('../results/thymus/totalvi_{}.npy'.format(adata_rna.obsm['X_totalvi'].shape[1]), adata_rna.obsm['X_totalvi'])

n_latent = 20
adata_rna.obsm['X_totalvi'] = np.load('../../results/thymus/totalvi_{}.npy'.format(n_latent))

# %%
curve = trajectory.get_curve(adata_rna, use_rep='X_totalvi')
trajectory.compute_pseudotime(adata_rna, curve, root_marker='Dcn')

ax = sq.pl.spatial_scatter(
    adata_rna, color='t', 
    cmap='RdBu_r', size=100, img=False, return_ax=True
)
ax.set_title(r'Inferred spatial gradient $(t)$ - TotalVI', fontdict={'fontsize': 14})

# plot.disp_trajectory(
#     adata_rna, cmap='RdBu', use_rep='X_totalvi',
#     title='Principal Curve - TotalVI'
# )

# %%
# Compare w/ ground-truth CMA
gamma_totalvi = adata_rna.obs['t'].values
gamma_true = adata_protein.obs['CMA'].values
gamma_true = (gamma_true-gamma_true.min()) / (gamma_true.max()-gamma_true.min())

plot.disp_kde_scatter(
    gamma_true, gamma_totalvi, ss_ratio=1.,
    xlabel=r"Ground-truth $(t)$",
    ylabel=r"TotalVI prediction $(t)$",
    title="CMA\n TotalVI vs. Ground-truth"
)

