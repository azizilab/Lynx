# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.append('../../')
sys.path.append('../../util/')
import IO

import novae
%load_ext autoreload
%autoreload 2

# %%
# Load data
xenium_path = '../../data/xenium/'
desi_path = '../../data/desi/'
sample_id = 'NIH_F5_proseg'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))

# Filtered Xenium data with aligned DESI for benchmarking fairness
adata, _ = IO.filter_cells(adata_xenium, adata_desi, by='map') 

# %%
# Train novae model
# Build spatial graph
novae.utils.spatial_neighbors(adata, radius=50)
model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

# %%
# Version 1: zero-shot inference
model.compute_representations(adata, accelerator="gpu")
model.assign_domains(adata, level=5)
latent = adata.obsm['novae_latent']
clusters = adata.obs['novae_domains_5'].values
np.save('../../results/liver/Novae_xenium_zero_shot_latent.npy', latent)
np.save('../../results/liver/Novae_xenium_zero_shot_seg.npy', clusters)

# %%
# Version 2: fine-tuning
t0 = time.perf_counter()
model.fine_tune(adata, accelerator="gpu")
model.compute_representations(adata, accelerator="gpu")
t1 = time.perf_counter()

with open(os.path.join("../results/liver/runtime.txt"), 'a') as f:
    f.write(f'Novae training time (s): {t1 - t0:.2f}\n')

model.assign_domains(adata, level=5)
latent = adata.obsm['novae_latent']
clusters = adata.obs['novae_domains_5'].values
np.save('../../results/liver/Novae_xenium_latent.npy', latent)
np.save('../../results/liver/Novae_xenium_seg.npy', clusters)