# %%
# -----------------------
# Cell-type annotations 
# -----------------------

# %%
import os
import sys
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import spatialdata as sd
import spatialdata_plot
import matplotlib.pyplot as plt

sys.path.append('..')
from util import IO, utils
from IPython.display import display
from importlib import reload
%load_ext autoreload
%autoreload 2

xenium_path = '../data/xenium/'

sample_ids = [
    'NIH_F2_proseg',
    'NIH_F3_proseg',
    'NIH_F4_proseg',
    'NIH_M1_proseg',
    'NIH_M2_proseg',
    'NIH_M3_proseg',
    'NIH_M4_proseg',
    'NIH_M5_proseg',
]


# %%
# ----------------------------
#  Cluster-based annotations
# ----------------------------

# %%
# Finish one-sample and transfer with scanpy.ingest

# Load proseg results
sample_id = 'NIH_F5_proseg'
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

# Cholangiocytes
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


# %% [markdown]
# Marker-based assignment from Leiden fine-grained clusterin
# High-level cell-types followed by specific cell-state corrections

# %%
sc.tl.leiden(adata_norm, resolution=1.5, flavor='igraph')
sc.pl.umap(adata_norm, color='leiden', s=5)


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
    '13':   'Cholangiocyte',
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
    color=generic_fib_markers + portal_fib_markers,
    cmap='magma', ncols=3, s=5, return_fig=True
)
fig.suptitle('Fibroblasts', fontsize=20)

# %%
adata_fib.obs.loc[adata_fib.obs['leiden'] == '0', 'subtype'] = 'Perisinusoidal stroma'
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
# Refining myeloid: Kupffer vs. Monocyte-derived macrophages via Kupffer signature score
adata_myeloid.obs['subtype'] = adata_myeloid.obs['subtype'].astype('str')
kupffer_signature = [g for g in ['MARCO', 'CD5L', 'VSIG4'] if g in adata_myeloid.var_names]
sc.tl.score_genes(adata_myeloid, gene_list=kupffer_signature, score_name='kupffer_score')

is_kupffer = adata_myeloid.obs['kupffer_score'] > 0
prev_mono = adata_myeloid.obs['subtype'] == 'Inflammatory Monocytes' # Archived subtype, set to Monocyte-derived Macs
adata_myeloid.obs['subtype'] = np.where(is_kupffer & ~prev_mono, 'Kupffer', 'Monocyte-derived macrophages')
adata_myeloid.obs['subtype'] = adata_myeloid.obs['subtype'].astype('category')

sc.pl.umap(adata_myeloid, color='subtype', s=5)
gc.collect()

# %%
adata_norm.obs['subtype'] = adata_norm.obs['subtype'].astype('str')
adata_norm.obs.loc[adata_norm.obs['cell_type'] == 'Myeloid', 'subtype'] = adata_myeloid.obs['subtype'].to_numpy()


# %%
# (e). SMCs + Endothelial
adata_smc_endo, deg_dict = compute_subcluster_deg(adata, 'cell_type', 'Endothelial+SMCs', leiden_res=0.5, return_leiden=True)
deg_dict

# %%
adata_smc_endo.obs.loc[adata_smc_endo.obs['leiden'] == '0', 'subtype'] = 'Vascular Endothelial'
adata_smc_endo.obs.loc[adata_smc_endo.obs['leiden'] == '1', 'subtype'] = 'SMCs'
adata_smc_endo.obs.loc[adata_smc_endo.obs['leiden'] == '2', 'subtype'] = 'Vascular Endothelial'
adata_smc_endo.obs.loc[adata_smc_endo.obs['leiden'] == '3', 'subtype'] = 'Vascular Endothelial'

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
adata_norm.obs['subtype'].value_counts()


# %%
# (h). Update cell-type annotations
# - relaxing / increasing granularity levels and more rigorous naming
adata_norm.obs['subtype'] = adata_norm.obs['subtype'].astype('str')
cluster_dict = {
    'PC-Hep': 'Hepatocytes',
    'PP-Hep': 'Hepatocytes',
    'Progenitor+Cholangiocytes': 'Cholangiocytes',
}
adata_norm.obs['subtype'] = adata_norm.obs['subtype'].map(cluster_dict).fillna(adata_norm.obs['subtype'])


# %%
# Save annotations w/ raw-count matrix
adata.obs['subtype'] = adata_norm.obs['subtype'].values.copy()
adata.write_h5ad(os.path.join(xenium_path, sample_id, 'cell_feature_matrix.h5'))

# %%
# transfer annotations to the rest of post-Proseg samples
for target_id in sample_ids:
    print(f'Cell type transferring to sample {target_id}...')
    # adata_target = sc.read_10x_h5(os.path.join(xenium_path, target_id, 'cell_feature_matrix.h5'))
    adata_target = sc.read_h5ad(os.path.join(xenium_path, target_id, 'cell_feature_matrix.h5'))
    adata_target = adata_target[:, adata.var_names]
    adata_target_norm = adata_target.copy()

    sc.pp.normalize_total(adata_target_norm, target_sum=1e4)
    sc.pp.log1p(adata_target_norm)
    sc.pp.pca(adata_target_norm)
    sc.pp.neighbors(adata_target_norm)
    sc.tl.umap(adata_target_norm)
    sc.pl.umap(adata_target_norm, s=5)

    # Cell-type transfer
    sc.tl.ingest(adata_target_norm, adata_norm, obs='subtype')
    sc.pl.umap(adata_target_norm, color='subtype', s=5)

    adata_target.obs['subtype'] = adata_target_norm.obs['subtype'].values.copy()
    adata_target.write_h5ad(os.path.join(xenium_path, target_id, 'cell_feature_matrix.h5'))


# %%
# TMP: 
# Propagate subtype to the liver_multimodal_analysis project copies.
multimodal_path = '../../liver_multimodal_analysis/data/'
pp_markers = ['CYP2A7', 'CYP2B6']
pc_markers = ['CYP3A4', 'APOA5']

for sid in sample_ids:
    src_path = os.path.join(xenium_path, sid, 'cell_feature_matrix.h5')
    tgt_path = os.path.join(multimodal_path, f'LYNX_{sid}_xenium.h5ad')

    adata_src = sc.read_h5ad(src_path)
    adata_tgt = sc.read_h5ad(tgt_path)

    missing = adata_tgt.obs_names.difference(adata_src.obs_names)
    assert len(missing) == 0, f'{sid}: {len(missing)} target cells not found in source'

    new_subtype = adata_src.obs['subtype'].reindex(adata_tgt.obs_names).astype(str)
    old_subtype = adata_tgt.obs['subtype'].astype(str)

    keep_mask = old_subtype.isin(['PP-Hep', 'PC-Hep']).values
    merged = new_subtype.copy()
    merged.values[keep_mask] = old_subtype.values[keep_mask]

    # Residual Hepatocytes (src says Hepatocytes but target wasn't PP/PC-Hep) → CYP-based split
    hep_mask = (merged.values == 'Hepatocytes')
    if hep_mask.sum() > 0:
        X_hep = adata_tgt[hep_mask, :]
        pp_expr = np.asarray(X_hep[:, pp_markers].X.mean(axis=1)).ravel()
        pc_expr = np.asarray(X_hep[:, pc_markers].X.mean(axis=1)).ravel()
        merged.values[hep_mask] = np.where(pp_expr >= pc_expr, 'PP-Hep', 'PC-Hep')

    adata_tgt.obs['subtype'] = pd.Categorical(merged.values)

    print(f'=== {sid} ===')
    print(f'  PP/PC-Hep preserved: {keep_mask.sum()}')
    print(f'  residual Hepatocytes reclassified: {hep_mask.sum()}')
    print(f'  changed vs old: {(merged.values != old_subtype.values).sum()}/{adata_tgt.n_obs}')
    print(adata_tgt.obs["subtype"].value_counts().to_string())
    print()
    adata_tgt.write_h5ad(tgt_path) 
