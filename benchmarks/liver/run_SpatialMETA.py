# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from IPython.display import display

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
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')


# %%
# Assign spatial modalities by KNN graph
# Reference: https://spatialmeta.readthedocs.io/en/latest/notebooks/Reassignment.html
# Use ST as alignment reference, set K=30 for fair comparison

ref_key = 'xenium_map'
ST_coords_df = smt.pp.ST_spot_sample(adata_xenium, 'spatial')  
adata_desi_interp, adata_xenium_interp = smt.pp.spot_align_byknn(
    ST_coords_df,
    adata_desi,
    adata_xenium,
    spatail_key_SM=ref_key,
)

# %%
# Double check spatial expression distributions (which modality got smoothed)
sc.pl.spatial(adata_desi_interp, img_key=None, color='Taurine [M-H]-', size=25)
sc.pl.spatial(adata_xenium_interp, img_key=None, color='DPT', size=25)


# %%
# Dataset setup
adata_xenium_interp.X = adata_xenium_interp.X.astype(np.int32)
adata_desi_interp.X = (adata_desi_interp.X*256).astype(np.int32)
adata_desi_interp.obs['spot_name'] = adata_desi_interp.obs_names.copy()  # Unify `spot_name` to integer

joint_adata = smt.pp.joint_adata_sm_st(
    adata_desi_interp,
    adata_xenium_interp
)
joint_adata.layers["counts"] = joint_adata.X.copy()
smt.pp.normalize_total_joint_adata_sm_st(
    joint_adata,
    target_sum_SM=1e4,
    target_sum_ST=1e4
)
joint_adata.layers["normalized"] = joint_adata.X.copy()
joint_adata.raw = joint_adata

joint_adata.X = joint_adata.layers["counts"]
smt.pp.normalize_total_joint_adata_sm_st(
    joint_adata,
    target_sum_SM=1e3,
    target_sum_ST=None
)

# %%
# Running SpatialMETA to get integrated latent embedding
# Note: No need to further subset SVFs based on the tutorial
t0 = time.perf_counter()
model = smt.model.ConditionalVAESTSM(
    joint_adata,
    device='cuda:0',
    reconstruction_method_sm='mse',
    reconstruction_method_st='zinb',
)

# Note: 
# - setting > 100 leads to NaN (lr=1e-4)
# - setting > 50 leads to NaN (lr=1e-3)
loss_dict = model.fit(
    max_epoch=100,
    lr=1e-4,
    mode='single'
)
t1 = time.perf_counter()

with open(os.path.join("../../results/liver/runtime.txt"), 'a') as f:
    f.write(f'SpatialMETA training time (s): {t1 - t0:.2f}\n')

# %%
fig,axes=plt.subplots(3,3,figsize=(20,10))
axes=axes.flatten()
for ax,(k,v) in zip(axes, loss_dict.items()):
    ax.plot(v)
    ax.set_title(k)


# %%
Z = model.get_latent_embedding()
C = model.get_modality_contribution()
joint_adata.obsm['X_emb']=Z
joint_adata.obs['contribution_st']=C
joint_adata.obs['contribution_sm']=1-C

np.save('../../results/liver/SpatialMETA_embedding.npy', Z)


# %% [markdown]
# SpatialMETA run failed due to NaNs in optimization
