# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.append('../')
sys.path.append('../util/')
import IO

import novae
%load_ext autoreload
%autoreload 2


# %%
# Load dataset
data_path = '../data/thymus/'
sample_ids = sorted([
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
])
sample_id = sample_ids[0]
adata = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))

# %%
from sklearn.neighbors import NearestNeighbors
# Find radius to k=8 nearest neighbors
spatial_coords = adata.obsm['spatial']

# Fit nearest neighbors (k+1 to include the point itself, then exclude it)
nbrs = NearestNeighbors(n_neighbors=9).fit(spatial_coords)
distances, indices = nbrs.kneighbors(spatial_coords)

# Remove the first column (distance to itself, which is 0)
neighbor_distances = distances[:, 1:]

# Compute average distance to 8 nearest neighbors for each point
avg_distances = np.mean(neighbor_distances, axis=1)

print(f"Overall average distance to 8 nearest neighbors: {np.mean(avg_distances):.3f}")
print(f"Standard deviation: {np.std(avg_distances):.3f}")


# %%
# Train novae model
# Build spatial graph
novae.utils.spatial_neighbors(adata, radius=123)
model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")


# %%
# Version 2: fine-tuning
# model.fine_tune(adata, accelerator="gpu")
# model.compute_representations(adata, accelerator="gpu")

model.assign_domains(adata, level=4)
latent = adata.obsm['novae_latent']
clusters = adata.obs['novae_domains_4'].values
np.save('../results/thymus/Novae_xenium_latent.npy', latent)
np.save('../results/thymus/Novae_xenium_seg.npy', clusters)

# %%
sc.pl.spatial(adata, color='novae_domains_4', size=100)