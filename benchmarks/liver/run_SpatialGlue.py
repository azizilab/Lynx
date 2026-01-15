# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.append('../')
sys.path.append('../util/')
import IO

import SpatialGlue
from SpatialGlue.preprocess import clr_normalize_each_cell, pca, fix_seed
from SpatialGlue.preprocess import construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
%load_ext autoreload
%autoreload 2

# %%
# Load data & preprocessing
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5_proseg'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))

# Filtered Xenium data with aligned DESI for benchmarking fairness
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map') 
sc.pp.normalize_total(adata_xenium, target_sum=1e4)
sc.pp.log1p(adata_xenium)

# %%
# Interpolate DESI to the same resolution w/ Xenium for 1-1 spatial mapping
desi_intensities = np.vstack((adata_desi.X.copy(), np.zeros(adata_desi.shape[1]))).astype(np.float32)
coords_lookup = {
    tuple(coord): i
    for i, coord in enumerate(adata_desi.obsm['spatial'])
}
indices = np.apply_along_axis(
    lambda x: 
    coords_lookup[tuple(x)] 
    if tuple(x) in coords_lookup else len(desi_intensities)-1,
    1, adata_xenium.obsm['desi_map']
)
desi_interp = desi_intensities[indices]

adata_desi_interp = sc.AnnData(desi_interp)
adata_desi_interp.obsm['spatial'] = adata_xenium.obsm['spatial'].copy()
adata_desi_interp.obs.index = adata_xenium.obs.index.copy()
adata_desi_interp.var = adata_desi.var.copy()

# %%
n_pcs = 50
adata_xenium.obsm['feat'] = pca(
    adata_xenium, 
    n_comps=n_pcs
)

adata_desi_interp.obsm['feat'] = pca(
    adata_desi_interp,
    n_comps=n_pcs
)

# %%
# Train SpatialGlue model
device = torch.device('cpu')
fix_seed(42)

data = construct_neighbor_graph(
    adata_desi_interp,
    adata_xenium,
    n_neighbors=5
)
gc.collect()

t0 = time.perf_counter()
model = Train_SpatialGlue(
    data, 
    device=device,
    random_seed=42,
    weight_factors=[1, 5, 1, 1]
)
output = model.train()
t1 = time.perf_counter()
with open(os.path.join("../results/liver/runtime.txt"), 'a') as f:
    f.write(f'SpatialGlue training time (s): {t1 - t0:.2f}\n')

joint_latent = output['SpatialGlue']
np.save('../results/liver/SpatialGlue_xenium_latent.npy', output['SpatialGlue'])
# %%
