## Spatial clustering & trajectory inference via SIMVI 
## Evaluate the interpretablilty of the spatial representation (s) & spatial effects (DML)

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
# Dataset specs
k = 20

xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, _ = IO.filter_cells(adata_xenium, adata_desi, by='map')

# scvi-tools version issue, need to copy the AnnData
adata = adata_xenium.copy()
SimVI.setup_anndata(adata)
edge_index = SimVI.extract_edge_index(adata, n_neighbors=k)

# %%
# Training & Inference
seed_everything(42)

model = SimVI(
    adata, kl_weight=1, kl_gatweight=0.01, lam_mi=1000, 
    permutation_rate=0.5, n_spatial=20, n_intrinsic=20
)
train_loss, val_loss = model.train(edge_index, max_epochs=200, batch_size=500, use_gpu=False, mae_epochs=25)

# %%
# Parsing intrinsic & spatial variables
adata.obsm['simvi_z'] = model.get_latent_representation(edge_index, representation_kind='intrinsic', give_mean=True)
adata.obsm['simvi_s'] = model.get_latent_representation(edge_index, representation_kind='interaction', give_mean=True)

# %%
# Load pretrained model
model = SimVI(
    adata, kl_weight=1, kl_gatweight=0.01, lam_mi=1000, 
    permutation_rate=0.5, n_spatial=20, n_intrinsic=20
)
adata.obsm['simvi_z'] = np.load('../results/liver/SIMVI_xenium_z20.npy')
adata.obsm['simvi_s'] = np.load('../results/liver/SIMVI_xenium_s20.npy')


# %%
# Spatial effects with archetypes & Huber regression

# Log-transform expressions before archetypal analysis
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)

seed_everything(42)
n_archetypes = 5
se_list, r2_zlist, r2_slist, r2_zpvlist, r2_spvlist, S = model.get_se(
    edge_index, adata=adata, num_arch=n_archetypes, Kfold=1, transformation='none'
)

# Save archetypes (NxK) reduced on `S` feature dimension
arch_cols = ['Archetype_'+str(i) for i in range(n_archetypes)]
adata.obsm['S'] = pd.DataFrame(S, index=adata.obs_names, columns=['Archetype_'+str(i) for i in range(n_archetypes)])

# %%
sq.pl.spatial_scatter(
    sq.pl.extract(adata, 'S'), color=arch_cols, 
    cmap='Blues', size=20, img=False, ncols=3
)

# %%
# Hubert regression for "intrinsic" & "spatial" features
from sklearn.linear_model import HuberRegressor

adata_ = adata.copy()
adata_.var['r2_z'] = np.max(r2_zlist, axis=0)
adata_.var['r2_s'] = np.max(r2_slist, axis=0)

hr = HuberRegressor()
hr.fit(adata_.var['r2_z'].values.reshape(-1,1), adata_.var['r2_s'].values)

adata_.var['class'] = 'Others'
adata_.var['class'][adata_.var['r2_z']>0.6] = 'Intrinsic-specific'
adata_.var['class'][np.abs(adata_.var['r2_s'].values - hr.predict(adata_.var['r2_z'].values.reshape(-1,1)) ) / hr.scale_ > 10] = 'Spatial-induced'


adata_.uns['class_colors'] = ['#33a02c','#bdbdbd','#084594']
sc.pl.scatter(adata_.copy(),x='r2_z', y='r2_s', show=False, color='class')
plt.plot(adata_.var['r2_z'].values[np.argsort(adata_.var['r2_z'].values)],hr.predict(adata_.var['r2_z'].values.reshape(-1,1))[np.argsort(adata_.var['r2_z'].values)])

for i in range(adata_.var['r2_s'].shape[0]):
    if adata_.var['r2_s'][i] > 0.082:
        plt.text(adata_.var['r2_z'][i]-0.05,adata_.var['r2_s'][i]-0.014,adata_.var_names[i],fontsize=12)
        
plt.xlabel(r'Intrinsic variation $r^2$')
plt.ylabel(r'Spatial effect $r^2$')
plt.title('SIMVI spatial regression on features', fontsize=15)

# %%
# Principal-curve based trajectory inference
trajectory.compute_trajectory(
    adata, 
    use_rep='simvi_s',
    root_marker='DPT'
)

sq.pl.spatial_scatter(
    adata, color='t', 
    cmap='RdBu_r', size=20, img=False,
    title=r'Spatial Trajectory ($\gamma(t)$)'+'\nSIMVI (Xenium)'
)

# %%
# Comparison against the annotation ground-truth
gamma_true = np.load('../results/liver/ablation/antibody_gamma.npy')
gamma_simvi = adata.obs['t'].values

rand_indices = np.random.choice(
    np.arange(adata_xenium.shape[0]), 10000, replace=False
)
plot.disp_kde_scatter(
    gamma_simvi[rand_indices], gamma_true[rand_indices],
    xlabel=r"Antibody-annotated $\gamma(t)$",
    ylabel=r"SIMVI prediction $\gamma(t)$",
    title="Spatial Trajectory\n SIMVI vs. Antibody"
)

# %%
# Save model & latent variables
model.save("../results/liver/simvi_model.pt")
np.save('../results/liver/SIMVI_xenium_z20.npy', adata.obsm['simvi_z'])
np.save('../results/liver/SIMVI_xenium_s20.npy', adata.obsm['simvi_s'])
