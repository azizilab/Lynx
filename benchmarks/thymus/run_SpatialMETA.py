# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch

sys.path.append('../../')
sys.path.append('../../util/')
import IO
import spatialmeta as smt

%load_ext autoreload
%autoreload 2

# %%
# Load dataset
data_path = '../../data/thymus/'
sample_id = 'Mouse_Thymus1'

adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
# sc.pp.normalize_total(adata_rna, target_sum=1e4)
# sc.pp.log1p(adata_rna)
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))
adata_protein.var_names_make_unique()

# %%
# Dataset setup
adata_rna.obs['spot_name'] = adata_rna.obs_names.copy()
adata_rna.obs['x_coord'] = adata_rna.obs.x.values.copy()
adata_rna.obs['y_coord'] = adata_rna.obs.y.values.copy()

adata_protein.obs['spot_name'] = adata_protein.obs_names.copy()
adata_protein.obs['x_coord'] = adata_rna.obs.x.values.copy()
adata_protein.obs['y_coord'] = adata_rna.obs.y.values.copy()

joint_adata = smt.pp.joint_adata_sm_st(
    adata_SM_new=adata_protein,
    adata_ST_new=adata_rna
)

joint_adata = smt.pp.removeHSP_MT_RPL_DNAJ(joint_adata)
joint_adata.layers["counts"] = joint_adata.X.copy()
smt.pp.normalize_total_joint_adata_sm_st(
    joint_adata,
    target_sum_SM=None,
    target_sum_ST=1e4
)
joint_adata.X = joint_adata.layers["counts"]

# %%
# Running SpatialMETA to get integrated latent embedding
model = smt.model.ConditionalVAESTSM(
    joint_adata,
    device='cuda:0',
    reconstruction_method_sm='mse',
    reconstruction_method_st='zinb',
)

loss_dict = model.fit(
    max_epoch=150,
    lr=1e-3,
    mode='single'
)

# %%
fig,axes=plt.subplots(3,3,figsize=(20,10))
axes=axes.flatten()
for ax,(k,v) in zip(axes, loss_dict.items()):
    ax.plot(v)
    ax.set_title(k)

# %%
# Parse embedding & contributions
Z = model.get_latent_embedding()
C = model.get_modality_contribution()
joint_adata.obsm['X_emb']=Z
joint_adata.obs['contribution_st']=C
joint_adata.obs['contribution_sm']=1-C

np.save('../../results/thymus/SpatialMETA_embedding.npy', Z)
