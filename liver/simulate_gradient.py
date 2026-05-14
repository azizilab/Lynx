# %% [markdown]
# Simulate ground-truth liver zonation gradient based on antibody staining

# %%
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import squidpy as sq

# Add project root to path
sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')

import IO, zonation

# %% 
def correct_mislabel_veins(adata, use_rep='ab_label', k=10):
    spatial_coords = adata.obsm["spatial"]
    labels = adata.obs[use_rep].values

    # Get indices for each label
    idx_label_1 = np.where(labels == 1)[0]
    idx_label_2 = np.where(labels == 2)[0]
    idx_label_3 = np.where(labels == 3)[0]

    # Build KD-trees for fast nearest-neighbor search
    tree_label_1 = cKDTree(spatial_coords[idx_label_1])
    tree_label_2 = cKDTree(spatial_coords[idx_label_2])

    # Find average distances from each label 3 cell to label 1 and label 2
    d1, _ = tree_label_1.query(spatial_coords[idx_label_3], k=k, workers=-1)
    d2, _ = tree_label_2.query(spatial_coords[idx_label_3], k=k, workers=-1)

    avg_d1 = d1.mean(axis=1)
    avg_d2 = d2.mean(axis=1)

    # Identify mislabeled 3s where avg distance to 1 is smaller than to 2
    mislabeled = avg_d1 < avg_d2
    labels[idx_label_3[mislabeled]] = 0  # Correct misclassified labels

    # Account for over-corrected '0's 
    # For all labels 0, if they're adjacent to 3, assign them back to 3
    idx_label_0 = np.where(labels == 0)[0]
    idx_label_3 = np.where(labels == 3)[0]
    
    if len(idx_label_3) > 0:  # Only proceed if there are label 3 cells
        tree_label_3 = cKDTree(spatial_coords[idx_label_3])
        for idx_0 in idx_label_0:
            # Find nearest label 3 cell
            dist, _ = tree_label_3.query(spatial_coords[idx_0], k=1)
            if dist < 100:  # Adjust threshold as needed
                labels[idx_0] = 3
    
    # Denoise
    labels = adata.obs[use_rep].values

    # Get indices for labels 0 & 3
    idx_label_0 = np.where(labels == 0)[0]
    idx_label_3 = np.where(labels == 3)[0]
    idx_0_3 = np.concatenate([idx_label_0, idx_label_3])  # Only process 0 & 3

    # Build KD-tree for spatial queries
    tree = cKDTree(spatial_coords)

    # Query nearest neighbors (excluding self)
    _, neighbors = tree.query(spatial_coords[idx_0_3], k=k+1, workers=-1)  # k+1 to exclude self

    # Count majority labels in neighbors
    for i, idx in enumerate(idx_0_3):
        neighbor_labels = labels[neighbors[i, 1:]]  # Exclude self
        majority_label = np.bincount(neighbor_labels).argmax()  # Most frequent label

        # Only update if the majority is different from the current label
        if majority_label in {0, 3} and majority_label != labels[idx]:
            labels[idx] = majority_label

    # Update AnnData object
    adata.obs[use_rep] = labels
    return None

# %% 
# Load data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id+'_proseg'), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'_proseg.h5'))
adata_ref, _ = IO.filter_cells(adata_xenium, adata_desi, by='map')

adata_norm = adata_ref.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)

# %% 
# Load antibody validation data
ab_path = '../data/integrated/antibody/'

adata_ab = IO.load_ab_stain(
    os.path.join(ab_path, 'NIH_F5.ome.tif'),
    adata_ref=adata_ref
)

# Normalize to [0, 1] per channel
scaled_chans = np.zeros_like(adata_ab.X)
for i, chan in enumerate(adata_ab.X.T):
    chan = (chan-chan.min()) / (chan.max()-chan.min())
    scaled_chans[:, i] = chan
adata_ab.X = scaled_chans

ab_dict = {
    'Opal 690-GS': 'Central Vein',
    'Opal 780-CYP3A4': 'Peri-central',
    'Opal 570-ASS1': 'Peri-portal',
    'Opal 520-Col1': 'Portal Vein'
}
ab_labels = list(ab_dict.keys())

# %% 
# (1). Refine CV / PV labels based on antibody markers
argmax_expr = adata_ab.X.argmax(1)
adata_ab.obs['ab_label'] = argmax_expr
correct_mislabel_veins(adata_ab, k=50)

# (2). PDE-based simulation
adata_ab.obs['boundary'] = np.nan

# Annotate CV
cv_mask = (
    (adata_ab.obs['ab_label']==0) &
    (adata.obs['subtype'] != 'Portal Fibroblasts') &
    (adata.obs['subtype'] != 'PP-Hep') &
    (adata.obs['subtype'] != 'Endothelial') &
    (adata.obs['subtype'] != 'SMCs') &
    (adata.obs['subtype'] != 'Progenitor+Cholangiocytes')
)
adata_ab.obs.loc[cv_mask, 'boundary'] = 1  # CV

# Annotate PV
pv_mask = (
    (adata_ab.obs['ab_label'] == 3)  &
    ((adata.obs['subtype'] == 'PP-Hep') |
     (adata.obs['subtype'] == 'Portal Fibroblasts'))
)
adata_ab.obs.loc[pv_mask, 'boundary'] = 0

# Annotate PC/PP boundary
label_1_mask = adata_ab.obs['ab_label'] == 1
label_2_mask = adata_ab.obs['ab_label'] == 2
coords = adata_ab.obsm['spatial']
idx_1 = np.where(label_1_mask)[0]
idx_2 = np.where(label_2_mask)[0]
tree = cKDTree(coords)
k = 1
for idx in idx_1:
    distances, neighbors = tree.query(coords[idx], k=k+1)  
    neighbors = neighbors[1:]  # Exclude self
    if np.any(label_2_mask.iloc[neighbors]):
        adata_ab.obs.loc[adata_ab.obs.index[idx], 'boundary'] = 0.5
for idx in idx_2:
    distances, neighbors = tree.query(coords[idx], k=k+1)  
    neighbors = neighbors[1:]  # Exclude self
    if np.any(label_1_mask.iloc[neighbors]):
        adata_ab.obs.loc[adata_ab.obs.index[idx], 'boundary'] = 0.5

# %% 
# Solve Laplace Equation
uptake_model = zonation.MetabolicZonation(beta=0, D=1)
uptake_model.fit(adata_ab)

# %% 
# Visualization
fig, ax = plt.subplots()
sq.pl.spatial_scatter(
    adata_ab, color='t_porto_central', img=False, size=20, cmap='RdBu_r',
    title='Ground-truth portal-central gradient',
    fig=fig, ax=ax, return_ax=True, colorbar=False
)

sm = ax.collections[0] 
cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label(r'Gradient coordinate $(t)$ (PV $\rightarrow$ CV)', fontsize=8)
plt.show()

# %% 
# Save results
adata_ab.write_h5ad('../results/liver/ab_validation.h5ad')
