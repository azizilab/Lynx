## Spatial trajectory inference via spaTrack

# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import spaTrack as spt

# %%
import matplotlib.pyplot as plt
import seaborn as sns

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
adata.obs['cluster'] = adata.obs['cell_type'].copy()

print("# cells:", adata.shape)

del adata_xenium, adata_desi
gc.collect()

# %%
# Preprocessing
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %%
# spaTrack analysis
# (1). Select the starting cell-label with Cholangiocytes
start_cells=spt.set_start_cells(adata,select_way='cell_type',cell_type='Cholangiocytes + Progenitor')


# %%
# (2). Comute cell transition prob.
gc.collect()
adata.obsm['X_spatial'] = adata.obsm['spatial'].copy()
adata.obsp["trans"] = spt.get_ot_matrix(adata, data_type="spatial", alpha1=0.5, alpha2=0.5)

# TODO: full OT isn't even feasible for ~60,000 cells



# %%
# (3). Compute cell pseudotime
adata.obs["ptime"] = spt.get_ptime(adata, start_cells)

# %%
