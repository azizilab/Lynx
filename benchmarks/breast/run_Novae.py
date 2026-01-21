# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import rcParams

sys.path.append('../../')
sys.path.append('../../util/')
import IO


sns.set_context('paper')
rcParams.update({'font.family': 'Arial'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

import novae
%load_ext autoreload
%autoreload 2

# %%
# Load data
data_path = '../../data/breast/dcis_fov/'
adata = sc.read_h5ad(os.path.join(data_path, 'cell_feature_matrix.h5'))
cluster_key = 'cell_type'

# Filter out 'Unlabeled' cells & cells with extremely rare cell-types (DCIS2)
# FIlter out hybrid annotations
rare_labels = adata.obs[cluster_key].value_counts()[
    adata.obs[cluster_key].value_counts() < 10
].index.to_list()

labeled_mask = np.logical_and(
    adata.obs[cluster_key] != 'Unlabeled',
    ~adata.obs[cluster_key].isin(rare_labels)
)

hybrid_mask = adata.obs[cluster_key].str.contains('Hybrid', case=False)
labeled_mask = np.logical_and(labeled_mask, ~hybrid_mask)

# IMPORTANT: the author wrongly asigned 'DCIS_2' as 'DCIS_1' in this patch
# As there're no true 'DCIS_1' cells, we relabel 'DCIS_1' to 'DCIS'
adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
adata.obs.loc[adata.obs[cluster_key] == 'DCIS_1'] = 'DCIS'
adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
adata = adata[labeled_mask].copy()
# adata.obs_names = adata.obs_names.astype(str)

# %%
# Train novae model
# Build spatial graph
novae.utils.spatial_neighbors(adata, radius=50)
model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

# # %%
# # Version 1: zero-shot inference
# model.compute_representations(adata, accelerator="gpu")
# model.assign_domains(adata, level=5)
# latent = adata.obsm['novae_latent']
# clusters = adata.obs['novae_domains_5'].values
# np.save('../results/liver/Novae_xenium_zero_shot_latent.npy', latent)
# np.save('../results/liver/Novae_xenium_zero_shot_seg.npy', clusters)


# %%
# Version 2: fine-tuning
model.fine_tune(adata, accelerator="gpu")
model.compute_representations(adata, accelerator="gpu")
latent = adata.obsm['novae_latent']
np.save('../../results/breast/Novae_xenium_latent.npy', latent)

# %%
# Check latent manifold
# adata_latent = sc.AnnData(latent)
# adata_latent.obs[cluster_key] = adata.obs[cluster_key].values
# sc.pp.pca(adata_latent)

sc.set_figure_params(scanpy=True, dpi_save=300, fontsize=10)
sc.pl.pca(adata_latent, color=cluster_key)

#%%
sc.pp.neighbors(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent, color=cluster_key)



# %%
