# %%
# ---------------------------------------------------------
# Cell-type annotations (default segmentation vs. Proseg)
# ---------------------------------------------------------

# %%
import os
import sys
import gc
import zarr
import json
import tifffile

import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import spatialdata_plot
import matplotlib.pyplot as plt

from IPython.display import display

sys.path.append('..')
from util import IO, utils

# %%
from importlib import reload
%load_ext autoreload
%autoreload 2

# %%
# ----------------------------
#  Cluster-based annotations
# ----------------------------

# %%
xenium_path = '../data/xenium/'
sample_id = 'NIH_F5'


# %%
# adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id))
# adata_xenium_norm = adata_xenium.copy()

# TODO: load from proseg results
sdata = sd.read_zarr(os.path.join(xenium_path, sample_id, 'output.zarr'))
adata_xenium = sdata['table'].copy()
adata_xenium_norm = adata_xenium.copy()

sc.pp.normalize_total(adata_xenium_norm)
sc.pp.log1p(adata_xenium_norm)
sc.pp.pca(adata_xenium_norm)
sc.pp.neighbors(adata_xenium_norm)
sc.tl.umap(adata_xenium_norm)



# %%
# High-level markers
fig = sc.pl.umap(
    adata_xenium_norm, 
    color=['CYP3A4', 'CYP2A7', 'ADH1C', 'HAMP'],  # Hepatocytes
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('Hepatocytes', fontsize=20)
plt.show()

# Cholangiocytes
fig = sc.pl.umap(
    adata_xenium_norm, 
    color=['KRT7', 'CFTR', 'TM4SF4', 'EHF'],
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('Cholangiocytes', fontsize=20)
plt.show()

# Fibroblasts
fig = sc.pl.umap(
    adata_xenium_norm, 
    color=['FBN1', 'PDGFRA', 'THY1', 'ASPN'],  # Fibroblasts
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('Fibroblasts', fontsize=20)
plt.show()

# Smooth Muscle cells
fig = sc.pl.umap(
    adata_xenium_norm,
    color=['MYH11', 'ACTA2', 'CNN1', 'RERGL'],
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('Smooth Muscle cells', fontsize=20)
plt.show()

# Endothelial
fig = sc.pl.umap(
    adata_xenium_norm,
    color=['SNCG', 'CD34', 'PECAM1'], 
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('Endothelial', fontsize=20)
plt.show()

# Pan sinusoidal
fig = sc.pl.umap(
    adata_xenium_norm, 
    color=['LYVE1', 'PDPN'], 
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('Sinusoidal', fontsize=20)
plt.show()

# Kupffer
fig = sc.pl.umap(
    adata_xenium_norm,
    color=['CD68', 'CD163', 'MARCO', 'CD14'],
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('Kupffer', fontsize=20)
plt.show()

# T-cell
fig = sc.pl.umap(
    adata_xenium_norm, 
    color=['CD3E', 'CD4', 'PTPRC', 'CD69'], 
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('T-cells', fontsize=20)
plt.show()

gc.collect()


# %%
sc.tl.leiden(adata_xenium_norm, resolution=1.5, flavor='igraph', n_iterations=2)
sc.pl.umap(adata_xenium_norm, color='leiden', s=5)

# %%
# Interactive debugging??
sc.pl.umap(
    adata_xenium_norm, color='leiden', 
    groups=[
        '0','1', '2', '7', '8', '10', '11', '12', '14'
    ],
    s=10
)


# %%
adata_subset = adata_xenium_norm[adata_xenium_norm.obs['leiden'] == '4'].copy()
adata_subset.shape

# %%
sc.pl.umap(
    adata_subset,
    color=['MARCO', 'CD163', 'CD4', 'CD3E', 'IL7R', 'CD14'],
    ncols=2, s=20, cmap='magma'
)

# %%
# Major cell-type assignment
major_cell_types = [
    'LSECs',
    'Hepatocytes',
    'T-cells',
    'Fibroblasts',
    'SMCs',
    'Endothelial',
    'Myeloid',
    'Progenitor + Cholangiocytes'
]

cell_type_assignments = {
    '0':    'Unknown',
    '1':    'Hepatocytes',
    '2':    'Hepatocytes',
    '3':    'Fibroblasts',
    '4':    'Myeloid',
    '5':    'SMCs',
    '6':    'Fibroblasts',    
    '7':    'Hepatocytes',
    '8':    'Hepatocytes',
    '9':    'Progenitor + Cholangiocytes',
    '10':   'Hepatocytes',
    '11':   'Hepatocytes',
    '12':   'Hepatocytes',
    '13':   'Myeloid',
    '14':   'Hepatocytes',
    '15':   'Endothelial',
    '16':   'LSECs',
    '17':   'T-cells'
}

adata_xenium_norm.obs['cell_type'] = adata_xenium_norm.obs['leiden'].apply(
    lambda x: cell_type_assignments[x]
)
adata_xenium_norm.obs['cell_type'] = adata_xenium_norm.obs['cell_type'].astype('category')

# Visualization
sc.pl.umap(adata_xenium_norm, color='cell_type')
adata_xenium_norm.obs['cell_type'].value_counts() / len(adata_xenium_norm)


# %%
# Save annotations w/ raw-count matrix
adata_xenium.obs['cell_type'] = adata_xenium_norm.obs['cell_type'].values.copy()
adata_xenium = adata_xenium[adata_xenium.obs['cell_type'] != 'Unknown'].copy()
adata_xenium.write_h5ad(os.path.join(xenium_path, sample_id, 'cell_feature_matrix.h5'))

# %%
sc.pl.umap(adata_xenium_norm, color='cell_type')

# %%
sc.pl.umap(adata_xenium_norm, color=['subtype'], ncols=2, wspace=0.5, s=5)

# %%
sc.pl.umap(
    adata_xenium_norm,
    color=['CD68', 'CD163', 'MARCO', 'CD14', 'CD4', 'CD3E', 'PTPRC', 'FCGR3A'],
    ncols=2, s=5, cmap='magma'
)

# %%
# Is it monocyte markers??
sc.pl.umap(
    adata_xenium_norm,
    color=['CCR7', 'SLAMF7', 'PDPN', 'CSF2RA', 'CXCR4', 'CCL19', 'LAMP3'],
    ncols=2, s=5, cmap='magma'
)


# %%
# Finer-level annotations
adata_xenium_norm.obs['subtype'] = 'NA'

# %%
# (a). Hepatocytes: 
adata_hep = adata_xenium[adata_xenium.obs['cell_type'] == 'Hepatocytes'].copy()
sc.pp.normalize_total(adata_hep)
sc.pp.log1p(adata_hep)
sc.pp.pca(adata_hep)
sc.pp.neighbors(adata_hep)
sc.tl.umap(adata_hep)

sc.tl.leiden(adata_hep, flavor='igraph', resolution=0.5, n_iterations=2)
sc.pl.umap(adata_hep, color='leiden', s=5)


# %%
fig = sc.pl.umap(
    adata_hep, 
    color=['CYP3A4','ADH4', 'ADH1C', 'APOA5'],
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('PC-Hep', fontsize=20)
plt.show()

fig = sc.pl.umap(
    adata_hep, 
    color=['CYP2A7', 'CYP2B6'],
    cmap='magma', ncols=2, s=10, return_fig=True
)
fig.suptitle('PP-Hep', fontsize=20)
plt.show()

# %%
# By avg. filtering???
pp_markers = ['CYP2A7', 'CYP2B6']
pc_markers = ['CYP3A4', 'APOA5']
is_pp = (adata_hep[:, pp_markers].X.A.mean(1) >= adata_hep[:, pc_markers].X.A.mean(1)) 
adata_hep.obs['subtype'] = pd.Series(is_pp).apply(lambda x: 'PP-Hep' if x else 'PC-Hep').values
adata_hep.obs['subtype'] = adata_hep.obs['subtype'].astype('category')
sc.pl.umap(adata_hep, color='subtype', s=5)

del pp_markers, pc_markers, is_pp
gc.collect()

# %%
adata_xenium_norm.obs['subtype'] = adata_xenium_norm.obs['subtype'].astype('str')
adata_xenium_norm.obs.loc[adata_xenium_norm.obs['cell_type'] == 'Hepatocytes', 'subtype'] = adata_hep.obs['subtype'].values


# %%
# (b). Fibroblasts
adata_fib = adata_xenium[adata_xenium.obs['cell_type'] == 'Fibroblasts'].copy()
sc.pp.normalize_total(adata_fib)
sc.pp.log1p(adata_fib)
sc.pp.pca(adata_fib)
sc.pp.neighbors(adata_fib)
sc.tl.umap(adata_fib)

sc.tl.leiden(adata_fib, flavor='igraph', resolution=0.1, n_iterations=2)
sc.pl.umap(adata_fib, color='leiden', s=5)

# %%
generic_fib_markers = ['FBN1', 'PDGFRA', 'ASPN']
portal_fib_markers = ['THY1', 'PDGFRB', 'PTGDS']
HSC_markers = ['ACTA2', 'FCN2', 'SMA']

fig = sc.pl.umap(
    adata_fib, 
    color=generic_fib_markers + portal_fib_markers + HSC_markers,
    cmap='magma', ncols=3, s=10, return_fig=True
)
fig.suptitle('Fibroblasts', fontsize=20)

# %%
adata_fib.obs['subtype'] = adata_fib.obs['leiden'].apply(
    lambda x: 'Generic-Fibroblasts' if x == '0' else 'Portal-Fibroblasts'
)
adata_fib.obs['subtype'] = adata_fib.obs['subtype'].astype('category')
sc.pl.umap(adata_fib, color='subtype', s=5)
del generic_fib_markers, portal_fib_markers, HSC_markers
gc.collect()

# %%
adata_xenium_norm.obs['subtype'] = adata_xenium_norm.obs['subtype'].astype('str')
adata_xenium_norm.obs.loc[adata_xenium_norm.obs['cell_type'] == 'Fibroblasts', 'subtype'] = adata_fib.obs['subtype'].values

# %%
# (c). Myeloid
adata_xenium_norm.obs['subtype'] = adata_xenium_norm.obs['subtype'].astype('str')
adata_xenium_norm.obs.loc[adata_xenium_norm.obs['leiden'] == '4', 'subtype'] = 'Monocyte'
adata_xenium_norm.obs.loc[adata_xenium_norm.obs['leiden'] == '13', 'subtype'] = 'Kupffer'

# %%
# Label remaining 'subtype' values with 'NA' to match 'cell_type' labels
mask = adata_xenium_norm.obs['subtype'] == 'NA'
adata_xenium_norm.obs['subtype'] = adata_xenium_norm.obs['subtype'].astype('str')
adata_xenium_norm.obs.loc[mask, 'subtype'] = adata_xenium_norm.obs.loc[mask, 'cell_type']
adata_xenium_norm.obs['subtype'] = adata_xenium_norm.obs['subtype'].astype('category')

# %%
sc.pl.umap(
    adata_xenium_norm, 
    color=['cell_type', 'subtype'],
    wspace=0.5,
    ncols=2, s=5
)

# %%
# TODO: double check myeloid DEGs (it's not monocytes)
adata_myeloid = adata_xenium[adata_xenium.obs['cell_type'] == 'Myeloid'].copy()
sc.pp.normalize_total(adata_myeloid)
sc.pp.log1p(adata_myeloid)
sc.pp.pca(adata_myeloid)
sc.pp.neighbors(adata_myeloid)
sc.tl.umap(adata_myeloid)



# %%
sc.tl.leiden(adata_myeloid, flavor='igraph', resolution=0.5, n_iterations=2)
sc.pl.umap(adata_myeloid, color='leiden', s=5, title='Myeloid clusters')

# %%
# Check canonical macrophage / monocyte markers
sc.pl.umap(
    adata_myeloid, 
    color=['CCR7', 'SLAMF7', 'PDPN', 'CSF2RA', 'CXCR4', 'CCL19', 'LAMP3'],
    ncols=2, s=10, cmap='magma'
)

# %%
sc.tl.rank_genes_groups(adata_myeloid, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata_myeloid, n_genes=10)
sc.pl.rank_genes_groups_dotplot(
    adata_myeloid, groupby="leiden", standard_scale="var", n_genes=10
)

# %%
print('Top DEGs per each myeloid cluster:')
for cluster in np.unique(adata_myeloid.obs['leiden']):
    print(f"Cluster {cluster}:")
    display(sc.get.rank_genes_groups_df(adata_myeloid, group=cluster).head(20))
    print("\n")

del cluster

# %%
# Save annotations w/ raw-count matrix
adata_xenium.obs['subtype'] = adata_xenium_norm.obs['subtype'].values.copy()
adata_xenium.write_h5ad(os.path.join(xenium_path, 'NIH_F5_proseg', 'cell_feature_matrix.h5'))

# %%
# Update the spatial data object with annotations
sdata['table'] = adata_xenium
sdata.write(os.path.join(xenium_path, sample_id, 'output_annotated.zarr'))
