# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch

from pytorch_lightning.utilities.seed import seed_everything

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
sys.path.append('../')
sys.path.append('../util/')
import IO, plot, trajectory
from simvi.model import SimVI

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

# %%
# Dataset specs
k = 8
adata = adata_rna.copy()
SimVI.setup_anndata(adata)
edge_index = SimVI.extract_edge_index(adata, n_neighbors=k)

# %%
# Training & Inference
seed_everything(42)

model = SimVI(
    adata, kl_weight=1, kl_gatweight=0.01, lam_mi=1000, 
    permutation_rate=0.5, n_spatial=20, n_intrinsic=20
)
train_loss, val_loss = model.train(edge_index, max_epochs=200, batch_size=500, use_gpu=True, mae_epochs=25)

# %%
adata.obsm['simvi_z'] = model.get_latent_representation(edge_index, representation_kind='intrinsic', give_mean=True)
adata.obsm['simvi_s'] = model.get_latent_representation(edge_index, representation_kind='interaction', give_mean=True)

# %%
# %%
# Save model & latent variables
model.save("../results/thymus/simvi_model.pt")
np.save('../results/thymus/SIMVI_rna_z20.npy', adata.obsm['simvi_z'])
np.save('../results/thymus/SIMVI_rna_s20.npy', adata.obsm['simvi_s'])

# %%
adata.obsm['simvi_s'] = np.load('../results/thymus/SIMVI_rna_s20.npy')

curve = trajectory.get_curve(adata, use_rep='simvi_s')
trajectory.compute_pseudotime(adata, curve, root_marker='Dcn')

ax = sq.pl.spatial_scatter(
    adata, color='t', 
    cmap='RdBu_r', size=100, img=False, return_ax=True
)
ax.set_title(r'Inferred spatial gradient $(t)$ - SIMVI', fontsize=14)

# %%
plot.disp_trajectory(
    adata, use_rep='simvi_s', cmap='RdBu',
    title='Principal Curve - SIMVI'
)

# %%
%reload_ext autoreload

# %% 
# Compare w/ ground-truth CMA
gamma_simvi = adata.obs['t'].values
gamma_true = adata.obs['CMA'].values
gamma_true = (gamma_true-gamma_true.min()) / (gamma_true.max()-gamma_true.min())

plot.disp_kde_scatter(
    gamma_true, gamma_simvi, ss_ratio=1.,
    xlabel=r"Ground-truth $\gamma(t)$",
    ylabel=r"SIMVI prediction $\gamma(t)$",
    title="CMA\n SIMVI vs. Ground-truth"
)


