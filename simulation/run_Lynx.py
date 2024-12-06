# %%
# Running LYNX on simulation data

# %%
import os
import gc
import sys
import time

import pickle
import gzip
import tifffile

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import scFates as scf

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro

import seaborn as sns
import matplotlib.pyplot as plt

# %%
from ipywidgets import interact, widgets
from IPython.display import display

from matplotlib import rcParams
rcParams.update({'font.size': 10})
rcParams.update({'figure.dpi': 300})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# %%
sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import IO, utils, plot, configs, dataset, trajectory
from models import vgae, model_train

# %%
# Load data
data_path = '../data/simulation/'
adata = sc.read_h5ad(os.path.join(data_path, 'xenium_feature_matrix.h5'))
adata_desi = sc.read_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))

# Append DESI feature to Xenium matrix
adata.obsm['X_aux'] = adata_desi.X

display(adata)
display(adata_desi)

# %%
# Data configs
loader = dataset.XeniumGraphDataset(k=30, n_subgraphs=8)
graph_data = loader.load_graphs([adata])
train_data, val_data = random_split(graph_data, [0.8, 0.2])
train_dl = DataLoader(graph_data, shuffle=True)
val_dl = DataLoader(graph_data)


# %%
from importlib import reload
reload(vgae)
reload(model_train)

# %%
# Model configs
torch.manual_seed(42)
device = torch.device('cuda')

n_latent = 6

train_configs = configs.set_train_configs(
    n_epochs=300, 
    lr=1e-3, 
    gamma=1., 
    patience=50,
    device=device
)

model_configs = configs.set_model_configs(
    c_in=adata.shape[1], c_aux=adata_desi.shape[1],
    c_covariate=0, c_hidden=64, c_latent=n_latent,
    beta=1, k_hop=2, dropout=0.5, act=nn.SiLU(),

    # DEBUG NF prior / posterior
    flow_prior=True
) 

# %%
# Model training
gc.collect()
torch.cuda.empty_cache()
pyro.clear_param_store()

model = vgae.VGAE(model_configs, device=train_configs.device)
model, losses, val_losses = model_train.train_vgae(
    model, train_configs,
    dataloader=train_dl,
    val_dataloader=val_dl,
    DEBUG=True
)

# %%
plt.figure(figsize=(5, 2))
plt.plot(np.arange(len(losses)), losses, label='Train')
plt.plot(np.arange(len(val_losses)), val_losses, label='Val')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('ELBO')
plt.show()



# %%
# Inference
gc.collect()
pyro.clear_param_store()
torch.cuda.empty_cache()

k = 30
n_subgraphs = 1
device = torch.device('cpu')
model.decode.device = device

preds = model.evaluate(adata, k=k, n_subgraphs=n_subgraphs, device=device)
adata.obsm['X_z_lynx'] = preds.qz

torch.cuda.empty_cache()

# %%
# Factor disentanglement
pz_corr = np.corrcoef(preds.pz.T)
g = sns.clustermap(pz_corr, cmap='RdBu_r')
g.figure.suptitle('p(z)', fontsize=30, y=1.05)
plt.show()

qz_corr = np.corrcoef(adata.obsm['X_z'].T)
g = sns.clustermap(qz_corr, cmap='RdBu_r')
g.figure.suptitle('q(z)', fontsize=30, y=1.05)
plt.show()

# %%
# Trajectory inference
dist_metric = 'euclidean'

trajectory.compute_trajectory(
    adata, 
    use_rep='X_z_lynx',
    n_nodes=10,
    dist_metric=dist_metric,
)

plot.disp_trajectory(
    adata, 
    cmap='RdBu',
    title='Spatial Gradients\n LYNX'
)

# %%
if 'milestones_colors' in adata.uns_keys():
    adata.uns.pop('milestones_colors')

sq.pl.spatial_scatter(
    adata, color='t', 
    cmap='RdBu', size=20, img=False,
    title='Pseudotime\n'+'LYNX'
)
plt.show()

utils.get_zonations(adata, n_zones=6, show=False)
adata.obs.loc[:, 'zone'] = adata.obs['milestones'].values.to_numpy().astype(np.float32)
sq.pl.spatial_scatter(
    adata, color='milestones', 
    cmap='RdBu_r', size=20, img=False, 
    title='Zonation\n'+'LYNX'
)
plt.show()

# adata.obs.drop('zone', axis=1, inplace=True)

# %%
np.save('../results/simulation/lynx_6.npy', preds.qz)
adata.obs.to_csv('../results/simulation/lynx_obs.csv', index=True)

# %%
# -------------
#  Evaluation
# -------------
# Ground-truth gradients
def _convert_gradients(gradients):
    """TMP: convert ground-truth gradients to 0-1"""
    v = gradients + gradients.min()
    return (v-v.min()) / (v.max()-v.min())
gradients_true = np.load(os.path.join(data_path, 'gradients.npy'))
gradients_true = _convert_gradients(gradients_true)
zonation_true = np.load(os.path.join(data_path, 'zonation.npy'))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
ax1.imshow(gradients_true, cmap='RdBu_r')
ax1.axis('off')
ax1.set_title('Pseudotime \nGround-truth')
ax2.imshow(zonation_true, cmap='turbo')
ax2.axis('off')
ax2.set_title('Zonation \nGround-truth')

plt.tight_layout()
plt.show()


# %%
gamma_lynx = 1 - adata.obs['t'].values
gamma_true = gradients_true[
    tuple([adata.obsm['desi_map'][:, 1], adata.obsm['desi_map'][:, 0]])
]
assert len(gamma_lynx) == len(gamma_true)

# %%
plt.figure(figsize=(5, 5), dpi=300)
plt.scatter(gamma_true, gamma_lynx, s=.1)
plt.xlabel('Ground-truth pseudotime')
plt.ylabel('LYNX prediction')
plt.show()


# %%
sq.pl.spatial_scatter(
    adata, color=z_labels, img=False, size=20, cmap='magma', ncols=3
)
plt.show()

# %%
# Check MCC
import numpy as np
from scipy.optimize import linear_sum_assignment

def mean_corr_coef_np(x, y):
    """
    # Reference: https://github.com/siamakz/iVAE/blob/master/lib/metrics.py
    """
    d = x.shape[1]
    cc = np.abs(np.corrcoef(x, y, rowvar=False)[:d, d:])
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score

# %%
# TODO: UMAP on z: plot individual z_i along trajectory
print('MCC (Lynx z vs. ground-truth z):', mean_corr_coef_np(adata.obsm['X_z'], preds.qz))

# %%
# spatial plots of individual q(z)
z_labels = ['qz'+str(i) for i in range(n_latent)]
for label, qzi in zip(z_labels, preds.qz.T):
    adata.obs[label] = qzi
del label, qzi

sq.pl.spatial_scatter(
    adata, color=z_labels, img=False, size=20, cmap='magma'
)
adata.obs.drop(z_labels, axis=1, inplace=True)
plt.show()


# %%
plt.scatter(adata.X.A[:, :50].flatten(), px[:, :50].flatten(), s=.5)
plt.show()



# %%
indices = np.random.choice(len(gamma_true), 5000, replace=False)
gamma_df = pd.DataFrame(
    np.vstack((gamma_true[indices], gamma_lynx[indices])).T,
    columns=['Ground-truth', 'LYNX']
)
sns.kdeplot(
    data=gamma_df, x='Ground-truth', y='LYNX',
    palette=sns.color_palette('Spectral'), levels=8, fill=True
)
plt.show()
del indices, gamma_df

# %%
