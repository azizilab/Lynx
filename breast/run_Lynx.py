# Infer spatial gradients on Xenium triple-positive breast cancer 
# Histology + Xenium

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

import seaborn as sns
import scFates as scf
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import rcParams

sns.set_context('paper')
rcParams.update({'font.family': 'Arial'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')

import IO, plot, utils, trajectory
import vgae, configs, dataset

%matplotlib inline
%load_ext autoreload
%autoreload 2


# %%
# ---------------
#   LYNX runs
# ---------------

# %%
# Dataset specs
n_subgraphs = 16
k = 8
r = 50

# Model parameters
n_hidden = 32
n_latent = 6

# Training parameters
n_epochs = 500
lr = 1e-2
patience = 50

data_path = '../data/breast/dcis_fov/'
outdir = '../figures/'
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'cell_feature_matrix.h5'))
adata_he = sc.read_h5ad(os.path.join(data_path, 'he_patches_norm.h5ad'))
cluster_key = 'cell_type'

rare_labels = adata_xenium.obs[cluster_key].value_counts()[
    adata_xenium.obs[cluster_key].value_counts() < 10
].index.to_list()

labeled_mask = np.logical_and(
    adata_xenium.obs[cluster_key] != 'Unlabeled',
    ~adata_xenium.obs[cluster_key].isin(rare_labels)
)

hybrid_mask = adata_xenium.obs[cluster_key].str.contains('Hybrid', case=False)
labeled_mask = np.logical_and(labeled_mask, ~hybrid_mask)

# Label unification & filtering: 
# IMPORTANT: the author wrongly asigned 'DCIS_2' as 'DCIS_1' in this patch
# 1. As there're no true 'DCIS_1' cells, we relabel 'DCIS_1' to 'DCIS'
# 2. Filter out 'Unlabeled' cells & cells with extremely rare cell-types
# 3. Filter out hybrid annotations
adata_xenium.obs[cluster_key] = adata_xenium.obs[cluster_key].astype(str)
adata_xenium.obs.loc[adata_xenium.obs[cluster_key] == 'DCIS_1'] = 'DCIS'
adata_xenium.obs.loc[adata_xenium.obs[cluster_key] == 'Prolif_Invasive_Tumor'] = 'Invasive_Tumor'
adata_xenium.obs[cluster_key] = adata_xenium.obs[cluster_key].astype('category')

adata_xenium = adata_xenium[labeled_mask].copy()
adata_xenium.obs.index = adata_xenium.obs.index.astype(int)
adata_he = adata_he[labeled_mask].copy()
patch_size = np.sqrt(adata_he.var.shape[0] // 3).astype(int)

del rare_labels, labeled_mask
gc.collect()

# %%
# Model setup
# TODO: debug (small cross-modal radius: e.g. 1-1)
graph_data = dataset.HeteroDataset(
    adatas_ref=adata_xenium, 
    adatas_query=adata_he,
    n_subgraphs=n_subgraphs, 
    k=k, r=r, 
    # r_bigraph=1, 
    is_weighted=True,
    cluster_key=cluster_key,
    alpha=1.0,

    # Update modality labels
    query='HE', query_proj_key='spatial',
    ref='Xenium', ref_proj_key='spatial' 
)

# Training & Inference
train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience, 
    device=torch.device('cuda'),
)

model_configs = configs.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden, 
    c_latent=n_latent,
    patch_size=patch_size,
    act=nn.SiLU(),
    infer_cell_interaction=True
) 

pyro.clear_param_store()
torch.cuda.empty_cache()

model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(graph_data, train_configs, DEBUG=True)
res = model.evaluate(
    adata_xenium, adata_he,
    graph_data=graph_data,
    device=torch.device('cpu'),
)

# %%
# Evaluation
plot.disp_kde_scatter(
    adata_xenium.X.A.flatten().copy(),
    res.px.flatten().copy(),
    xlabel=r"Observation $log(x+1)$",
    ylabel=r"Reconstruction $log(x+1)$",
    title='Feature reconstruction (human breast cancer)'
)
gc.collect()

# %%
principal_graph = trajectory.get_tree(
    adata_xenium,
    use_rep='X_z',
    n_nodes=int(0.01*adata_xenium.n_obs),
    ppt_lambda=1e3,
    plot_graph=True
)

# %%
trajectory.prune_tree(adata_xenium, tips_to_keep=[67, 78, 32])
scf.pl.graph(adata_xenium, basis='pca')

# %%
trajectory.compute_pseudotime(adata_xenium, principal_graph, source=36)

# %%
sc.pl.pca(adata_xenium, color=cluster_key)

# %%
# TODO: debug cell-cell interaction
from util import test_assoc
cluster_labels = adata_xenium.obs[cluster_key].cat.categories
adata_xenium.obs_names = adata_xenium.obs_names.astype('category')


cci_df = plot.summarize_cell_interaction(
    adata_xenium,
    cluster_key=cluster_key, 
    title='Summary of cell-cell interaction\n(Overall)',
    show_plot=True
)

cci_df, cci_pval = test_assoc.test_cci(adata_xenium, cci_df, cluster_labels, cluster_key=cluster_key)

plot.disp_heatmap(
    cci_df,
    title='Summary of cell-cell interaction\n(Overall)'
)

plot.disp_heatmap(
    cci_df,
    title='Summary of cell-cell interaction\n(P-val)'
)


# %%
# Save LYNX inference results
outdir = '../results/breast/'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)
adata_xenium.obs = adata_xenium.obs.loc[:, [cluster_key, 'leiden']]
adata_xenium.write_h5ad(os.path.join(outdir, 'LYNX_xenium_cci2.h5ad'))

# %%
