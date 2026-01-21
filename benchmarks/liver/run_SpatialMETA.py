# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.append('../../')
sys.path.append('../../util/')
import IO
import spatialmeta as smt

%load_ext autoreload
%autoreload 2

# %%
# Load data & preprocessing
xenium_path = '../../data/xenium/'
desi_path = '../../data/desi/'
sample_id = 'NIH_F5_proseg'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))

# Filtered Xenium data with aligned DESI for benchmarking fairness
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map') \

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
# Append dummy "spot name" & coords
adata_xenium.obs['spot_name'] = adata_xenium.obs_names.copy()
adata_xenium.obs['x_coord'] = adata_xenium.obs.x_centroid.values.copy()
adata_xenium.obs['y_coord'] = adata_xenium.obs.y_centroid.values.copy()

adata_desi_interp.obs['spot_name'] = adata_desi_interp.obs_names.copy()
adata_desi_interp.obs['x_coord'] = adata_xenium.obs.x_centroid.values.copy()
adata_desi_interp.obs['y_coord'] = adata_xenium.obs.y_centroid.values.copy()

adata_xenium.X = adata_xenium.X.astype(np.int32)
adata_desi_interp.X = (adata_desi_interp.X*256).astype(np.int32)

joint_adata = smt.pp.joint_adata_sm_st(
    adata_xenium,
    adata_desi_interp
)
joint_adata.layers["counts"] = joint_adata.X.copy()
smt.pp.normalize_total_joint_adata_sm_st(joint_adata,
                         target_sum_SM=1e4,
                         target_sum_ST=1e4)
joint_adata.layers["normalized"] = joint_adata.X.copy()
joint_adata.raw = joint_adata

# %%
# Running SpatialMETA to get integrated latent embedding
joint_adata.X = joint_adata.layers["counts"]
smt.pp.normalize_total_joint_adata_sm_st(
    joint_adata,
    target_sum_SM=1e3,
    target_sum_ST=None
)

# Note: No need to further subset SVFs based on the tutorial
t0 = time.perf_counter()
model = smt.model.ConditionalVAESTSM(
    joint_adata,
    device='cuda:0',
    reconstruction_method_sm='g',
    reconstruction_method_st='zinb',
)

loss_dict = model.fit(
    max_epoch=1,
    lr=1e-4,
    mode='single'
)

t1 = time.perf_counter()
with open(os.path.join("../../results/liver/runtime.txt"), 'a') as f:
    f.write(f'SpatialMETA training time (s): {t1 - t0:.2f}\n')

# %% [markdown]
# SpatialMETA run failed due to NaNs in optimization
