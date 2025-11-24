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
sample_id = 'NIH_F5_proseg'

# Load proseg results
adata = sc.read_h5ad(os.path.join(xenium_path, sample_id, 'cell_feature_matrix.h5'))
adata_norm = adata.copy()

sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)
sc.pp.pca(adata_norm)
sc.pp.neighbors(adata_norm)
sc.tl.umap(adata_norm)

# %%
# High-level markers
# Hepatocytes
fig = sc.pl.umap(
    adata_norm, 
    color=['CYP3A4', 'CYP2A7', 'CYP2B6', 'APOA5'],  # Hepatocytes
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('Hepatocytes', fontsize=20)
plt.show()

# Cholangiocytes & progenitors
fig = sc.pl.umap(
    adata_norm, 
    color=['KRT7', 'EPCAM'],
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('Cholangiocytes', fontsize=20)
plt.show()

# Fibroblasts
fig = sc.pl.umap(
    adata_norm, 
    color=['FBN1', 'THY1', 'ASPN'],  # Fibroblasts
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('Fibroblasts', fontsize=20)
plt.show()

# Smooth Muscle cells
fig = sc.pl.umap(
    adata_norm,
    color=['MYH11', 'ACTA2'],
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('Smooth Muscle cells', fontsize=20)
plt.show()

# Endothelial
fig = sc.pl.umap(
    adata_norm,
    color=['SNCG', 'CD34', 'PECAM1'], 
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('Endothelial', fontsize=20)
plt.show()

# Pan sinusoidal
fig = sc.pl.umap(
    adata_norm, 
    color=['LYVE1'], 
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('Sinusoidal', fontsize=20)
plt.show()

# Myeloid
fig = sc.pl.umap(
    adata_norm,
    color=['CD68', 'CD163', 'MARCO', 'SPI1', 'CLEC10A', 'FCGR3A'],
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('Kupffer', fontsize=20)
plt.show()

# Lymphocytes
fig = sc.pl.umap(
    adata_norm, 
    color=['CD3E', 'CD4', 'CD8A', 'BANK1', 'GNLY', 'MZB1'], 
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('T-cells', fontsize=20)
plt.show()

# HSCs
fig = sc.pl.umap(
    adata_norm, 
    color=['DES'], 
    cmap='magma', ncols=2, s=5, return_fig=True
)
fig.suptitle('Hepatic Stellate cells', fontsize=20)
gc.collect()


# %%
sc.tl.leiden(adata_norm, resolution=1.5, flavor='igraph')
sc.pl.umap(adata_norm, color='leiden', s=5)

# %%
# Interactive debugging??
sc.pl.umap(
    adata_norm, color='leiden', 
    groups=[
        '9',
    ],
    s=5
)

# %%
# Major cell-type assignment
# major_cell_types = [
#     'LSECs',
#     'Hepatocytes',
#     'T-cells',
#     'Fibroblasts',
#     'SMCs',
#     'Endothelial',
#     'Myeloid',
#     'Progenitor + Cholangiocytes'
# ]

cell_type_assignments = {
    '0':    'Hepatocytes',
    '1':    'Hepatocytes',
    '2':    'Hepatocytes',
    '3':    'Hepatocytes',
    '4':    'Hepatocytes',
    '5':    'Hepatocytes',
    '6':    'Myeloid',    
    '7':    'LSECs',
    '8':    'Hepatocytes',
    '9':    'Fibroblasts',
    '10':   'Fibroblasts',
    '11':   'Lymphocytes',
    '12':   'Endothelial+SMCs',
    '13':   'Progenitor+Cholangiocyte',
}

adata_norm.obs['cell_type'] = adata_norm.obs['leiden'].apply(
    lambda x: cell_type_assignments[x]
)
adata_norm.obs['cell_type'] = adata_norm.obs['cell_type'].astype('category')
sc.pl.umap(adata_norm, color='cell_type')
display(
    adata_norm.obs['cell_type'].value_counts() / len(adata_norm)
)

# %%
# Save annotations w/ raw-count matrix
adata.obs['cell_type'] = adata_norm.obs['cell_type'].values.copy()
adata.write_h5ad(os.path.join(xenium_path, sample_id, 'cell_feature_matrix.h5'))

# %%
# Finer-level annotations
def compute_subcluster_deg(adata, cluster_key, cluster, leiden_res=0.1, return_leiden=True):
    adata_subset = adata[adata.obs[cluster_key] == cluster].copy()

    if adata_subset.X[adata_subset.X > 0].min() >= 1.0:
        sc.pp.normalize_total(adata_subset, target_sum=1e4)
        sc.pp.log1p(adata_subset)
        sc.pp.pca(adata_subset)
        sc.pp.neighbors(adata_subset)
        sc.tl.umap(adata_subset)

    sc.tl.leiden(adata_subset, flavor='igraph', resolution=leiden_res)
    sc.pl.umap(adata_subset, color='leiden', s=10, title=cluster+ ' subclusters')

    sc.tl.rank_genes_groups(adata_subset, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata_subset, n_genes=20)
    sc.pl.rank_genes_groups_dotplot(
        adata_subset, groupby="leiden",
        standard_scale="var", n_genes=20
    )

    deg_dict = {}
    for c in np.unique(adata_subset.obs['leiden']):
        print(f"Sub-cluster {cluster}:")
        deg_df = sc.get.rank_genes_groups_df(adata_subset, group=c)
        display(deg_df.head(20))
        print("\n")

        deg_dict[c] = deg_df[
            (deg_df['pvals_adj'] < 0.05) & (deg_df['logfoldchanges'] > 2.0)
        ].names.tolist()

    if return_leiden:
        return adata_subset, deg_dict
    else:
        return None

adata_norm.obs['subtype'] = 'NA'

# %%
# (a). Hepatocytes: 
adata_hep, _ = compute_subcluster_deg(adata, 'cell_type', 'Hepatocytes', leiden_res=0.1, return_leiden=True)


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
adata_norm.obs['subtype'] = adata_norm.obs['subtype'].astype('str')
adata_norm.obs.loc[adata_norm.obs['cell_type'] == 'Hepatocytes', 'subtype'] = adata_hep.obs['subtype'].values

# %%
# (b). Fibroblasts
adata_fib, _ = compute_subcluster_deg(adata, 'cell_type', 'Fibroblasts', leiden_res=0.1, return_leiden=True)
generic_fib_markers = ['FBN1', 'PDGFRA', 'ASPN']
portal_fib_markers = ['THY1', 'PDGFRB', 'PTGDS']

fig = sc.pl.umap(
    adata_fib, 
    color=generic_fib_markers + portal_fib_markers + HSC_markers,
    cmap='magma', ncols=3, s=5, return_fig=True
)
fig.suptitle('Fibroblasts', fontsize=20)

# %%
adata_fib.obs.loc[adata_fib.obs['leiden'] == '0', 'subtype'] = 'Generic Fibroblasts'
adata_fib.obs.loc[adata_fib.obs['leiden'] == '1', 'subtype'] = 'Portal Fibroblasts'
adata_fib.obs['subtype'] = adata_fib.obs['subtype'].astype('category')
sc.pl.umap(adata_fib, color='subtype', s=5)
gc.collect()

# %%
adata_norm.obs['subtype'] = adata_norm.obs['subtype'].astype('str')
adata_norm.obs.loc[adata_norm.obs['cell_type'] == 'Fibroblasts', 'subtype'] = adata_fib.obs['subtype'].values

# %%
# (c). Myeloid
adata_myeloid, deg_dict = compute_subcluster_deg(adata, 'cell_type', 'Myeloid', leiden_res=0.2, return_leiden=True)
deg_dict

# %%
adata_myeloid.obs.loc[adata_myeloid.obs['leiden'] == '0', 'subtype'] = 'Kupffer'
adata_myeloid.obs.loc[adata_myeloid.obs['leiden'] == '1', 'subtype'] = 'Inflammatory Monocytes'
sc.pl.umap(adata_myeloid, color='subtype', s=5)
gc.collect()

# %%
adata_norm.obs['subtype'] = adata_norm.obs['subtype'].astype('str')
adata_norm.obs.loc[adata_norm.obs['cell_type'] == 'Fibroblasts', 'subtype'] = adata_fib.obs['subtype'].values


# %%
# (e). SMCs + Endothelial
adata_smc_endo, deg_dict = compute_subcluster_deg(adata, 'cell_type', 'Endothelial+SMCs', leiden_res=0.5, return_leiden=True)
deg_dict

# %%
adata_smc_endo.obs.loc[adata_smc_endo.obs['leiden'] == '0', 'subtype'] = 'Endothelial'
adata_smc_endo.obs.loc[adata_smc_endo.obs['leiden'] == '1', 'subtype'] = 'SMCs'
adata_smc_endo.obs.loc[adata_smc_endo.obs['leiden'] == '2', 'subtype'] = 'Endothelial'
adata_smc_endo.obs.loc[adata_smc_endo.obs['leiden'] == '3', 'subtype'] = 'LSECs'

# %%
adata_norm.obs['subtype'] = adata_norm.obs['subtype'].astype('str')
adata_norm.obs.loc[adata_norm.obs['cell_type'] == 'Endothelial+SMCs', 'subtype'] = adata_smc_endo.obs['subtype'].values
adata_norm.obs.loc[adata_norm.obs['subtype'] == 'LSECs', 'cell_type'] = 'LSECs'

# %%
# (f). Lymphocytes
adata_lymph, deg_dict = compute_subcluster_deg(adata, 'cell_type', 'Lymphocytes', leiden_res=0.1, return_leiden=True)
deg_dict

# %%
adata_lymph.obs.loc[adata_lymph.obs['leiden'] == '0', 'subtype'] = 'T-cells'
adata_lymph.obs.loc[adata_lymph.obs['leiden'] == '1', 'subtype'] = 'Plasma/B-cells'
sc.pl.umap(adata_lymph, color='subtype', s=5)
gc.collect()

# %%
adata_norm.obs['subtype'] = adata_norm.obs['subtype'].astype('str')
adata_norm.obs.loc[adata_norm.obs['cell_type'] == 'Lymphocytes', 'subtype'] = adata_lymph.obs['subtype'].values

# %%
# (g). Update the rest the same as general cell-type labels
adata_norm.obs.loc[adata_norm.obs['subtype'] == 'NA', 'subtype'] = adata_norm.obs.loc[
    adata_norm.obs['subtype'] == 'NA', 'cell_type'
].values.copy()

# %%
sc.pl.umap(
    adata_norm, color=['cell_type', 'subtype'], s=5
)


# %%
# Save annotations w/ raw-count matrix
adata.obs['subtype'] = adata_norm.obs['subtype'].values.copy()
adata.write_h5ad(os.path.join(xenium_path, sample_id, 'cell_feature_matrix.h5'))

# %%
# Update the spatial data object with annotations
# sdata['table'] = adata
# sdata.write(os.path.join(xenium_path, sample_id, 'output_annotated.zarr'))
