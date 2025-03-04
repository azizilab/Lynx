# %%
# Running LYNX on simulation data

import os
import gc
import sys

import numpy as np
import scanpy as sc
import squidpy as sq

import pyro
import torch
import torch.nn as nn
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
from importlib import reload


# %%
# -------------
#  Load data
# -------------

# Dataset specs
n_subgraphs = 16
k = 15
r = 50

# Simulation
data_path = '../data/simulation'
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'xenium_feature_matrix.h5'))
adata_desi = sc.read_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))

# Real data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'
adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')

graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_desi,
    n_subgraphs=n_subgraphs, 
    k=k,
    r=r,
    cluster=True
)

train_data, val_data = random_split(graph_data, [0.7, 0.3])
train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

# %%
# -----------------------------
#  Model training & inference
# -----------------------------

# Model pparameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500
lr = 1e-3
patience = 50

# Configs
train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience, 
    device=torch.device('cuda'),
    anneal=False,
    verbose=True,
    #scheduler step and gamma applies gamma every step
    step_size=100,
    gamma=0.1
)

model_configs = configs.set_model_configs(
    c_in=adata_xenium.shape[1],   # ref-dim 
    c_aux=adata_desi.shape[1],  # query-dim
    c_hidden=n_hidden, 
    c_latent=n_latent,
    act=nn.LeakyReLU(),
    ref=graph_data.ref, 
    query=graph_data.query,
    k_hop=1,
    num_heads=1,
    num_clusters=graph_data.num_clusters,
    verbose=True
) 

model = vgae.HeteroVGAE(model_configs, device=torch.device('cuda'))
model.fit(train_configs, train_dl=train_dl, val_dl=val_dl, DEBUG=True)

res = model.evaluate(
    adata_xenium, adata_desi, graph_data=graph_data,
    device=torch.device('cpu')
)

# %%
np.save('../results/simulation/lynx_6_desi.npy', res.qzu)
np.save('../results/simulation/lynx_6_xenium.npy', res.qzx)
adata_desi.obs['t'].to_csv('../results/simulation/lynx_desi_pseudotime.csv', index=True)
adata_xenium.obs['t'].to_csv('../results/simulation/lynx_xenium_pseudotime.csv', index=True)

# %%
# -------------
#  Evaluation
# -------------

from scipy.special import comb
def _convert_gradients(gradients):
    """TMP: convert ground-truth gradients to 0-1"""
    v = gradients + gradients.min()
    return (v-v.min()) / (v.max()-v.min())

def plot_factor_corr(z):
    z_corr = np.corrcoef(z.T)
    z_score = np.abs(np.tril(z_corr, k=-1)).sum() / comb(z_corr.shape[0], 2)

    g = sns.clustermap(z_corr, cmap='RdBu_r')
    g.figure.suptitle(
        'q(z)\n Correlation score: {}'.format(np.round(z_score, 3)), 
        fontsize=30, y=1.05
    )
    plt.show()

# %%
# (1). Observation reconstruction
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
# (2). Trajectory inference
# High-dim gradients (x)
adata_xenium.obsm['X_z_lynx'] = res.qzx
trajectory.compute_trajectory(
    adata_xenium, 
    use_rep='X_z_lynx',
    n_nodes=10,
)
sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu', size=20, img=False,
    title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (Simulation)'  # Xenium
)

# Low-dim gradients (u)
adata_desi.obsm['X_z_lynx'] = res.qzu
trajectory.compute_trajectory(
    adata_desi, 
    use_rep='X_z_lynx',
    n_nodes=10,
)
sq.pl.spatial_scatter(
    adata_desi, color='t', 
    cmap='RdBu', size=1, img=False,
    title=r'Trajectory Pseudotime ($\gamma(t)$)'+'\nLYNX (DESI)'
)

# %%
# Zonation
sc.pp.normalize_total(adata_xenium)
sc.pp.log1p(adata_xenium)
utils.get_zonations(adata_xenium, n_zones=6)

# %%
sq.pl.spatial_scatter(
    adata_xenium, color='milestones',
    cmap='turbo', size=20, img=False,
    title='Zonation \nLYNX'
)

# %%
# (3). Comparison w/ ground-truth trajectory gradients (\gamma(t)) (simulation-only)
gradients_true = np.load(os.path.join(data_path, 'gradients.npy'))
gradients_true = _convert_gradients(gradients_true)
zonation_true = np.load(os.path.join(data_path, 'zonation.npy'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))
ax1.imshow(gradients_true, cmap='RdBu_r')
ax1.axis('off')
ax1.set_title('Pseudotime \nGround-truth')
ax2.imshow(zonation_true, cmap='turbo')
ax2.axis('off')
ax2.set_title('Zonation \nGround-truth')

plt.tight_layout()
plt.show()

# %%
gamma_lynx = 1 - adata_xenium.obs['t'].values   # PV as left-end of the axis
gamma_true = gradients_true[
    tuple([adata_xenium.obsm['desi_map'][:, 1], adata_xenium.obsm['desi_map'][:, 0]])
]
assert len(gamma_lynx) == len(gamma_true)

plot.disp_kde_scatter(
    gamma_true, gamma_lynx, 
    xlabel=r"Ground-truth $\gamma(t)$",
    ylabel=r"LYNX prediction $\gamma(t)$",
    title="Trajectory pseudotime"
)

# %%
# (4). Latent disentanglement measure
# Check MCC (true disentanglement score)
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

print(
    'MCC (Lynx z vs. ground-truth z):', 
    mean_corr_coef_np(adata_xenium.obsm['X_z'], res.qzx)
)

# %%
# UMAP + spatial plots of individual q(z) & ground-truth z's
z_labels = ['z'+str(i) for i in range(n_latent)]

# UMAP plots
adata_desi.obs[z_labels] = adata_desi.obsm['X_z_lynx'].copy()
sc.pp.neighbors(adata_desi, n_neighbors=k, use_rep='X_z_lynx')
sc.tl.umap(adata_desi)
sc.pl.umap(adata_desi, color=z_labels, cmap='turbo', ncols=3)
adata_desi.obs.drop(z_labels, axis=1, inplace=True)

# Spatial plot
z_labels = ['z'+str(i) for i in range(n_latent)]
for label, zi in zip(z_labels, adata_desi.obsm['X_z_lynx'].T):
    adata_desi.obs[label] = zi
del label, zi

sq.pl.spatial_scatter(
    adata_desi, color=z_labels, img=False, size=1, cmap='turbo', ncols=3
)
adata_desi.obs.drop(z_labels, axis=1, inplace=True)
plt.show()
# %%
