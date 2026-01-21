# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import SpatialGlue
from SpatialGlue.preprocess import clr_normalize_each_cell, pca, fix_seed
from SpatialGlue.preprocess import construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

sys.path.append('../../')
sys.path.append('../../util/')
import IO
%load_ext autoreload
%autoreload 2

# %%
# Load dataset
data_path = '../../data/thymus/'
sample_id = 'Mouse_Thymus1'

adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))
adata_protein.var_names_make_unique()

# %%
# Train SpatialGlue model
data_type = 'Stereo-CITE-seq'
device = torch.device('cpu')
fix_seed(42)

n_pc_comps = min(adata_rna.shape[1], adata_protein.shape[1])
adata_rna.obsm['feat'] = pca(adata_rna, n_comps=n_pc_comps)
adata_protein.obsm['feat'] = pca(adata_protein, n_comps=n_pc_comps)

data = construct_neighbor_graph(adata_rna, adata_protein, datatype=data_type)
model = Train_SpatialGlue(data, datatype=data_type, device=device)
output = model.train()

# %%
np.save('../../results/thymus/SpatialGlue_embedding.npy', output['SpatialGlue'])