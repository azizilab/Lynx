## ## Spatial trajectory inference via SpaceFlow

# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import omicverse as ov

# %%
sys.path.append('../')
sys.path.append('../util/')
import IO, plot, trajectory

# %%
%load_ext autoreload
%autoreload 2

# %%
# Dataset specs
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))

# Filtered Xenium data with aligned DESI for benchmarking fairness
adata, _ = IO.filter_cells(adata_xenium, adata_desi, by='map') 



# %%
# SpaceFlow training: backbone architecture - DGI
sf_obj=ov.space.pySpaceFlow(adata)
sf_obj.train(
    spatial_regularization_strength=0.1, 
    z_dim=50, lr=1e-3, epochs=1000, 
    max_patience=50, min_stop=100, 
    random_seed=42, gpu=0, 
    regularization_acceleration=True, 
    edge_subset_sz=1000000
)


# %%
# Compute & extract pSM values
sf_obj.cal_pSM(
    n_neighbors=20,resolution=1,
    max_cell_for_subsampling=5000,psm_key='pSM_spaceflow'
)


# %%
# Visualize pSM
sq.pl.spatial_scatter(
    sf_obj.adata, color='pSM_spaceflow', 
    size=20,
    cmap='RdBu_r', 
    img=False
)

# %%
# Save pSM values
sf_obj.adata.obs[['pSM_spaceflow']].to_csv('../results/SpaceFlow_50_pseudotime.csv')

