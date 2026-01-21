# %%
# ----------------------
#  Downstream analysis
# ----------------------

# %%
import os
import sys
import gc

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import squidpy as sq

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('..')
from util import IO, utils, plot, test_assoc, trajectory

from IPython.display import display
from matplotlib import rcParams
from matplotlib.axes import Axes
rcParams['font.family'] = 'Arial'
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 150})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')

%load_ext autoreload
%autoreload 2

# %%
# Load data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
indir = '../results/liver/downstream/gradient/'
outdir = '../figures/joint_paper/'

# %% [markdown]
# ----------------------------------
#     Sex-specific joint analysis
# ----------------------------------

# (1). Continuous statistical test: joint analysis of significant features
#  - along PV - CV trajectory
#  - sex-specific analysis

# %%
# Perminantly update all DESI adata files
# mz_annot_df = pd.read_csv('../data/DESI_annotation.csv', header=0)
# mz_annot_df.columns = ['mz', 'annotation']
# mz_dict = {
#     k.strip(): v.strip() for k, v in zip(mz_annot_df.iloc[:, 0], mz_annot_df.iloc[:, 1])
#     if not pd.isna(v)
# }

# desi_path = '../data/desi/'
# sample_ids = sorted([
#     sample_id for sample_id in os.listdir(desi_path)
#     if sample_id.endswith('.h5')
# ])
# sample_ids

# for sample_id in sample_ids:
#     adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id))
#     adata_desi.var_names = [
#         mz_dict[c.strip()] if c.strip() in mz_dict else c.strip()
#         for c in adata_desi.var_names
#     ]
#     adata_desi.write_h5ad(os.path.join(desi_path, sample_id))
#     del adata_desi
#     gc.collect()

# %%
sample_ids = [
    'NIH_F2_proseg',
    'NIH_F3_proseg',
    'NIH_F4_proseg',
    'NIH_F5_proseg',
    'NIH_M1_proseg',
    'NIH_M2_proseg',
    'NIH_M3_proseg',
    'NIH_M4_proseg',
    'NIH_M5_proseg'
]

# %%
n_latent = 6
n_zones = 3
n_bins = 50
cluster_key = 'subtype'

# Binned expression per sample
gexps = [] 
mexps = []
celltype_dynamics = []

adatas_xenium = []
adatas_desi = []

for sample_id in sample_ids:
    print('Computing for  {}...'.format(sample_id))
    adata_xenium = sc.read_h5ad(os.path.join(indir, f'LYNX_{sample_id}_xenium.h5ad'))
    adata_desi = sc.read_h5ad(os.path.join(indir, f'LYNX_{sample_id}_desi.h5ad'))

    # curve = trajectory.get_curve(adata_xenium, trim_radius_ratio=0.25)
    # trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')
    # curve = trajectory.get_curve(adata_desi, trim_radius_ratio=0.25)
    # trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine [M-H]-')


    if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
        sc.pp.normalize_total(adata_xenium)
        sc.pp.log1p(adata_xenium)

    utils.get_zonation_features(
        adata_xenium, adata_desi,
        n_zones=n_zones, sample_id=sample_id,
        abundance_test=True, show=False
    )

    sq.pl.spatial_scatter(
        adata_xenium, color=['t', 'zone'],
        cmap='RdBu_r', img=False, size=25, ncols=2
    )
    plot.disp_trajectory(adata_xenium, cmap='RdBu_r')

    sq.pl.spatial_scatter(
        adata_desi, color=['t', 'zone'],
        cmap='RdBu_r', img=False, size=1, ncols=2
    )
    plot.disp_trajectory(adata_desi, cmap='RdBu_r')

    # Compute feature dynamics along trajectory
    # Sorting & binning genes
    indices = np.argsort(adata_xenium.obs['t']).values
    gexp_df = utils.get_binned_expr(
        #adata_xenium.to_df().iloc[indices].T,
        pd.DataFrame(
            adata_xenium.layers['px'],
            index=adata_xenium.obs_names,
            columns=adata_xenium.var_names
        ).iloc[indices].T,
        n_bins=n_bins,
    )

    gamma = utils.get_binned_expr(
        pd.DataFrame(adata_xenium.obs['t'].sort_values()).T,
        n_bins=n_bins
    ).values.flatten()
    gexp_df['t'] = gamma
    gexp_df['sample_id'] = sample_id
    gexp_df['sex'] = 'M' if 'M' in sample_id else 'F'

    gexps.append(gexp_df)

    # Sorting & binning metabolites
    indices = np.argsort(adata_desi.obs['t']).values
    mexp_df = utils.get_binned_expr(
        adata_desi.to_df().iloc[indices].T,
        n_bins=n_bins,
    )

    gamma = utils.get_binned_expr(
        pd.DataFrame(adata_desi.obs['t'].sort_values()).T,
        n_bins=n_bins
    ).values.flatten()
    mexp_df['t'] = gamma
    mexp_df['sample_id'] = sample_id
    mexp_df['sex'] = 'M' if 'M' in sample_id else 'F'

    mexps.append(mexp_df)

    celltype_dynamics_df = utils.get_celltype_dynamics(adata_xenium, adata_xenium.obs[cluster_key], n_bins=n_bins)
    celltype_dynamics_df['t'] = gamma
    celltype_dynamics_df['sample_id'] = sample_id
    celltype_dynamics_df['sex'] = 'M' if 'M' in sample_id else 'F'
    celltype_dynamics.append(celltype_dynamics_df)

    adatas_xenium.append(adata_xenium)
    adatas_desi.append(adata_desi)

    del adata_xenium, adata_desi, gamma, gexp_df, mexp_df
    del sample_id, indices
    gc.collect()

# %%
# Subset annotated metabolomics features
mz_annots = set(pd.read_csv('../data/DESI_annotation.csv', header=0).iloc[:, 1].values)
mz_features = [
    c for c in adatas_desi[0].var_names
    if c in mz_annots
]

# %%
# Finding sex-associated features w/ mixed-effect models & tests
# to find trajectory & sex-associated features
all_gexp_df = pd.concat(gexps, axis=0)
_, gene_test_assocs = test_assoc.get_test_associations(all_gexp_df)
fitted_gexp_df = all_gexp_df.copy()  # No need to refit gexps
fitted_gexp_df.drop('t', axis=1, inplace=True)

all_mexp_df = pd.concat(mexps, axis=0)
fitted_mexp_df, metabolite_test_assocs = test_assoc.get_test_associations(all_mexp_df)
metabolite_test_assocs = metabolite_test_assocs.loc[mz_features].copy()
fitted_mexp_df = fitted_mexp_df.loc[:, mz_features+['sample_id', 'sex']].copy()

gene_test_assocs.to_csv(os.path.join(indir, 'gene_test_assocs.csv'), index=True)
metabolite_test_assocs.to_csv(os.path.join(indir, 'metabolite_test_assocs.csv'), index=True)
gc.collect()

# %% [markdown]
# -------------------------------------------------------------
#  Figure 3&4: Summary of spatial gradients along trajectory
# -------------------------------------------------------------

# %%
# Visualize sample trajectory
for i in range(len(sample_ids)):
    sample_id = sample_ids[i].rpartition('_')[0] \
        if 'proseg' in sample_ids[i] else sample_ids[i]

    ax = sq.pl.spatial_scatter(
        adatas_xenium[i], color='t', size=20,
        cmap='RdBu_r', img=False, colorbar=False, return_ax=True,
        title=f'Spatial gradient - {sample_id}'
    )
    sm = ax.collections[0]
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label('PV → CV', fontsize=10)
    plt.savefig(os.path.join(outdir, f'spatial_gradient_{sample_id}.png'), bbox_inches='tight')

    ax = sq.pl.spatial_scatter(
        adatas_xenium[i], color='zone', size=20,
        cmap='RdBu_r', img=False, return_ax=True,
        title=f'Zonation - {sample_id}'
    )
    plt.savefig(os.path.join(outdir, f'spatial_zone_{sample_id}.png'), bbox_inches='tight')

del ax, sample_id
gc.collect()


# %% [markdown]
# (I). Continuous gradient analysis

# %%
# Helper functions
def smooth_zone_assignments(adata, n_bins, zone_labels=None):
    r"""Smooth discrete zone assignments"""
    assert 't' in adata.obs.keys() and 'zone' in adata.obs.keys(), \
        "Please run trajectory & zonation inference first"
    if zone_labels is not None:
        assert len(zone_labels) == len(np.unique(adata.obs['zone'])), \
            "Please provide correct zone label #"

    df = pd.DataFrame(adata.obs['t'].sort_values()).T
    smoothed_t = utils.get_binned_expr(df,n_bins=n_bins).values.flatten()
    zone_cutoffs = [
        adata[adata.obs['zone'] == i].obs['t'].max()
        for i in np.unique(adata.obs['zone'])
    ]
    smoothed_zones = np.digitize(smoothed_t, zone_cutoffs[:-1])
    assignments = np.array([zone_labels[z] for z in smoothed_zones]) \
        if zone_labels is not None else  \
        np.array(['Zone '+str(z+1) for z in smoothed_zones])

    return assignments

def plot_expr_gradient(
    binned_df, 
    zone_assignments, 
    cbar_pad=1.2,
    title='Gradient expression heatmap',
    features_to_annot=None, cmap='RdBu_r',
    dpi=100, figsize=(12, 8), show=True
):
    """
    Plot binned expression along the gradient with genes sorted 
    by their peak position using AxesDivider for perfect alignment.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    expr_data = binned_df.copy()
    
    gene_argmax_positions = expr_data.idxmax(axis=0)
    sorted_indices = np.argsort(gene_argmax_positions)
    sorted_genes = expr_data.columns[sorted_indices]
    
    sorted_expr = expr_data[sorted_genes].T 
    sorted_expr = (sorted_expr - sorted_expr.values.mean(axis=1, keepdims=True)) / \
                  (sorted_expr.values.std(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(sorted_expr, cmap=cmap, aspect='auto')

    divider = make_axes_locatable(ax)

    # Append colorbar & zone bar
    cax = divider.append_axes("right", size="5%", pad=cbar_pad)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Expression (zscore)', fontsize=10)

    unique_zones = np.unique(zone_assignments)
    n_zones_actual = len(unique_zones)
    zone_colors = plt.cm.get_cmap('Set3', n_zones_actual)
    zone_to_idx = {zone: i for i, zone in enumerate(unique_zones)}
    zone_indices = np.array([zone_to_idx[m] for m in zone_assignments])
    n_cols = sorted_expr.shape[1]

    zone_ax = divider.append_axes("bottom", size="5%", pad=0.3, sharex=ax)
    zone_ax.imshow(
        zone_indices.reshape(1, -1), 
        aspect='auto', 
        cmap=zone_colors,
        extent=[-0.5, n_cols-0.5, 0, 1]
    )
    
    zone_ax.set_yticks([])
    zone_ax.set_xticks([])
    
    zone_positions = []
    zone_labels = []
    for zone in unique_zones:
        zone_mask = zone_assignments == zone
        if np.any(zone_mask):
            indices = np.where(zone_mask)[0]
            center_pos = (indices[0] + indices[-1]) / 2
            zone_positions.append(center_pos)
            zone_labels.append(zone)
    
    for pos, label in zip(zone_positions, zone_labels):
        zone_ax.text(pos, 0.5, label, ha='center', va='center', 
                        fontsize=8, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'Pseudotime $(t)$'+ ' (PV → CV (bins)', fontsize=12)
    ax.set_ylabel('Genes', fontsize=12)
    ax.set_title(title, fontsize=15)

    if features_to_annot is not None:
        feature_positions = []
        for feature in features_to_annot:
            if feature in sorted_expr.index:
                pos = list(sorted_expr.index).index(feature)
                feature_positions.append((feature, pos))
        
        feature_positions.sort(key=lambda x: x[1])
        min_spacing = len(sorted_expr) * .03
        
        adjusted_positions = []
        for i, (feature, pos) in enumerate(feature_positions):
            if i == 0:
                adjusted_positions.append((feature, pos, pos))
            else:
                prev_adjusted = adjusted_positions[-1][2]
                if pos - prev_adjusted < min_spacing:
                    new_pos = prev_adjusted + min_spacing
                    adjusted_positions.append((feature, pos, new_pos))
                else:
                    adjusted_positions.append((feature, pos, pos))
        
        for feature, original_pos, adjusted_pos in adjusted_positions:
            if abs(original_pos - adjusted_pos) > 0.1:
                ax.annotate('', 
                            xy=(n_cols - 0.5, original_pos), 
                            xytext=(n_cols + n_cols*0.02, adjusted_pos),
                            arrowprops=dict(arrowstyle='-', color='black', lw=0.8, alpha=0.7),
                            annotation_clip=False)
                ax.text(n_cols + n_cols*0.03, adjusted_pos, feature, 
                        va='center', ha='left', fontsize=8, weight='bold')
            else:
                ax.annotate('', 
                            xy=(n_cols - 0.5, original_pos), 
                            xytext=(n_cols + n_cols*0.02, original_pos),
                            arrowprops=dict(arrowstyle='-', color='black', lw=0.8, alpha=0.7),
                            annotation_clip=False)
                ax.text(n_cols + n_cols*0.03, original_pos, feature, 
                        va='center', ha='left', fontsize=8, weight='bold')

    if show:
        plt.show()
        return None
    else:
        return fig, ax

# %%
# [markdown]
# Evaluate common & sex-specific diverging features
# with 

# %%
# Gradient summary heatmap by sex
adata_xenium_female = sc.concat([
    adatas_xenium[i] for i in range(len(sample_ids))
    if 'F' in sample_ids[i]
])
adata_desi_female = sc.concat([
    adatas_desi[i] for i in range(len(sample_ids))
    if 'F' in sample_ids[i]
])
adata_desi_female = adata_desi_female[:, mz_features].copy()

adata_xenium_male = sc.concat([
    adatas_xenium[i] for i in range(len(sample_ids))
    if 'M' in sample_ids[i]
])
adata_desi_male = sc.concat([
    adatas_desi[i] for i in range(len(sample_ids))
    if 'M' in sample_ids[i]
])
adata_desi_male = adata_desi_male[:, mz_features].copy()

# %%
# Sex-specific zonation clustering
utils.get_zonation_features(
    adata_xenium_male, adata_desi_male, n_zones=3, 
    sample_id='Pooled Male', abundance_test=True, show=True
)

utils.get_zonation_features(
    adata_xenium_female, adata_desi_female, n_zones=3, 
    sample_id='Pooled Female', abundance_test=True, show=True,
)
gc.collect()


# %%
male_gexps_df = fitted_gexp_df[fitted_gexp_df['sex'] == 'M'].copy()
male_gexps_df.reset_index(inplace=True)
numeric_cols = male_gexps_df.select_dtypes(include=[np.number]).columns
male_gexps_df = male_gexps_df.groupby('index')[numeric_cols].mean()
male_gexps_df.drop('index', axis=1, inplace=True)

female_gexps_df = fitted_gexp_df[fitted_gexp_df['sex'] == 'F'].copy()
female_gexps_df.reset_index(inplace=True)  
numeric_cols = female_gexps_df.select_dtypes(include=[np.number]).columns
female_gexps_df = female_gexps_df.groupby('index')[numeric_cols].mean()
female_gexps_df.drop('index', axis=1, inplace=True)

male_mexps_df = fitted_mexp_df[fitted_mexp_df['sex'] == 'M'].copy()
male_mexps_df.reset_index(inplace=True)
numeric_cols = male_mexps_df.select_dtypes(include=[np.number]).columns
male_mexps_df = male_mexps_df.groupby('index')[numeric_cols].mean()
male_mexps_df.drop('index', axis=1, inplace=True)

female_mexps_df = fitted_mexp_df[fitted_mexp_df['sex'] == 'F'].copy()
female_mexps_df.reset_index(inplace=True)  
numeric_cols = female_mexps_df.select_dtypes(include=[np.number]).columns
female_mexps_df = female_mexps_df.groupby('index')[numeric_cols].mean()
female_mexps_df.drop('index', axis=1, inplace=True)


# %%
# Saving pooled male / female expressions
male_gexp_gradients = male_gexps_df.copy()
male_gexp_gradients['zone'] = smooth_zone_assignments(adata_xenium_male, n_bins=n_bins)
female_gexp_gradients = female_gexps_df.copy()
female_gexp_gradients['zone'] = smooth_zone_assignments(adata_xenium_female, n_bins=n_bins)
male_gexp_gradients.to_csv(os.path.join(indir, 'male_gexp_gradients.csv'), index=True)
female_gexp_gradients.to_csv(os.path.join(indir, 'female_gexp_gradients.csv'), index=True)

male_mexp_gradients = male_mexps_df.copy()
male_mexp_gradients['zone'] = smooth_zone_assignments(adata_desi_male, n_bins=n_bins)
female_mexp_gradients = female_mexps_df.copy()
female_mexp_gradients['zone'] = smooth_zone_assignments(adata_desi_female, n_bins=n_bins)
male_mexp_gradients.to_csv(os.path.join(indir, 'male_mexp_gradients.csv'), index=True)
female_mexp_gradients.to_csv(os.path.join(indir, 'female_mexp_gradients.csv'), index=True)


# %%
# ---

# %%
# Save full heatmap of trajectory gradients
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

top_sex_genes = gene_test_assocs.sort_values('adj-pval.sex').head(10).index

smoothed_zones = smooth_zone_assignments(adata_xenium_male, n_bins=n_bins)
fig1, _ = plot_expr_gradient(
    male_gexps_df, zone_assignments=smoothed_zones,
    features_to_annot=top_sex_genes,
    figsize=(8, 8), cmap='seismic',  dpi=300, show=False,
    title='Pooled gene expression along \nPV-CV axis (Male)'
)

smoothed_zones = smooth_zone_assignments(adata_xenium_female, n_bins=n_bins)
fig2, _ = plot_expr_gradient(
    female_gexps_df, zone_assignments=smoothed_zones,
    features_to_annot=top_sex_genes,
    figsize=(8, 8), cmap='seismic', dpi=300, show=False,
    title='Pooled gene expression along \nPV-CV axis (Female)'
)

del top_sex_genes, smoothed_zones
gc.collect()

fig1.savefig(os.path.join(outdir, 'Fig3_gene_gradient_male.svg'), bbox_inches='tight')
fig2.savefig(os.path.join(outdir, 'Fig3_gene_gradient_female.svg'), bbox_inches='tight')


# %%
top_sex_metabolites = metabolite_test_assocs.sort_values('adj-pval.sex').head(10).index

smoothed_zones = smooth_zone_assignments(adata_desi_male, n_bins=n_bins)
fig1, _ = plot_expr_gradient(
    male_mexps_df, zone_assignments=smoothed_zones,
    features_to_annot=top_sex_metabolites, cbar_pad=2.1,
    figsize=(8, 8), cmap='seismic', dpi=300, show=False,
    title='Pooled metabolite intensity \nalong PV-CV axis (Male)'
)

smoothed_zones = smooth_zone_assignments(adata_desi_female, n_bins=n_bins)
fig2, _ = plot_expr_gradient(
    female_mexps_df, zone_assignments=smoothed_zones,
    features_to_annot=top_sex_metabolites, cbar_pad=2.1,
    figsize=(8, 8), cmap='seismic', dpi=300, show=False,
    title='Pooled metabolite intensity \nalong PV-CV axis (Female)'
)

del top_sex_metabolites, smoothed_zones
gc.collect()

fig1.savefig(os.path.join(outdir, 'Fig4_metabolite_gradient_male.svg'), bbox_inches='tight')
fig2.savefig(os.path.join(outdir, 'Fig4_metabolite_gradient_female.svg'), bbox_inches='tight')


# %%
# Visualize sex-differential genes & metabolites
print('sex-disparity genes')
print('===================================')

sex_genes = gene_test_assocs[
    gene_test_assocs['interact_feature']==1
].index

idx = 0
ncols = 4
while idx < len(sex_genes):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(sex_genes):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                all_gexp_df, 
                feature=sex_genes[idx], 
                ax=ax, show=False
            )
        idx += 1
    plt.show()
print('\n\n')
del idx
gc.collect()

# %%
feature = 'CYP3A4'
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
ax = plot.disp_sex_feature_dynamics(
    all_gexp_df, feature=feature,
    ax=ax, show=False
)
fig.savefig(os.path.join(outdir, f'{feature}_sex_diff.svg'), bbox_inches='tight')


# %%
print('sex-disparity metabolites')
print('===================================')

sex_metabolites = metabolite_test_assocs[
    metabolite_test_assocs['interact_feature']==1
].index

idx = 0
ncols = 4
while idx < len(sex_metabolites):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(sex_metabolites):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                all_mexp_df, feature=sex_metabolites[idx], 
                ax=ax, show=False, 
            )
        idx += 1
    plt.show()

# %% 
feature = 'TG 52:4 [M+Na]+'
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
ax = plot.disp_sex_feature_dynamics(
    all_mexp_df, feature=feature,
    ax=ax, show=False
)
fig.savefig(os.path.join(outdir, f'{feature}_sex_diff.svg'), bbox_inches='tight')


# %% [markdown]
# (2). Discrete zonation markers pooled across sex

# %%
# TMP: Assign zone labels back to each sample
female_xenium_zones = adata_xenium_female.obs['zone'].values
male_xenium_zones = adata_xenium_male.obs['zone'].values
female_desi_zones = adata_desi_female.obs['zone'].values
male_desi_zones = adata_desi_male.obs['zone'].values

female_xenium_idx, male_xenium_idx = 0, 0
female_desi_idx, male_desi_idx = 0, 0

female_xenium_ids, male_xenium_ids = [], []
female_desi_ids, male_desi_ids = [], []

for i, sample_id in enumerate(sample_ids):
    if 'F' in sample_id:
        female_xenium_ids.extend([sample_id]*adatas_xenium[i].n_obs)
        adatas_xenium[i].obs['zone'] = female_xenium_zones[
            female_xenium_idx:female_xenium_idx + adatas_xenium[i].n_obs
        ]
        female_xenium_idx += adatas_xenium[i].n_obs
        
        female_desi_ids.extend([sample_id]*adatas_desi[i].n_obs)
        adatas_desi[i].obs['zone'] = female_desi_zones[
            female_desi_idx:female_desi_idx + adatas_desi[i].n_obs
        ]
        female_desi_idx += adatas_desi[i].n_obs


    elif 'M' in sample_id:
        male_xenium_ids.extend([sample_id]*adatas_xenium[i].n_obs)
        adatas_xenium[i].obs['zone'] = male_xenium_zones[
            male_xenium_idx:male_xenium_idx + adatas_xenium[i].n_obs
        ]
        male_xenium_idx += adatas_xenium[i].n_obs

        male_desi_ids.extend([sample_id]*adatas_desi[i].n_obs)
        adatas_desi[i].obs['zone'] = male_desi_zones[
            male_desi_idx:male_desi_idx + adatas_desi[i].n_obs
        ]
        male_desi_idx += adatas_desi[i].n_obs

adata_xenium_female.obs['sample_id'] = female_xenium_ids
adata_xenium_male.obs['sample_id'] = male_xenium_ids
adata_desi_female.obs['sample_id'] = female_desi_ids
adata_desi_male.obs['sample_id'] = male_desi_ids

del female_xenium_zones, male_xenium_zones, female_desi_zones, male_desi_zones
del female_xenium_idx, male_xenium_idx, female_desi_idx, male_desi_idx, sample_id

# Visualize sample zone assignments
for i in range(len(sample_ids)):
    ax = sq.pl.spatial_scatter(
        adatas_xenium[i], color='zone', size=20,
        cmap='RdBu_r', img=False, return_ax=True,
        title=f'Zonation - {sample_ids[i]}'
    )
    # plt.savefig(os.path.join(outdir, f'spatial_zone_{sample_ids[i]}.png'), bbox_inches='tight')

del ax
gc.collect()


# %%
# Save up-regulated features per zone
for i in range(n_zones):
    zone_id = str(i+1)
    deg_female = adata_xenium_female.uns['zones'][zone_id]
    deg_male = adata_xenium_male.uns['zones'][zone_id]
    
    # deg_female.to_csv(os.path.join(indir, f'zone_{zone_id}_degs_female.csv'), index=True)
    deg_female[(deg_female['pvals_adj'] < 0.05) & (deg_female['logFC'] > 0)].sort_values('logFC', ascending=False).to_csv(
        os.path.join(indir, f'zone_{zone_id}_degs_female_up.csv'), index=False
    )

    # deg_male.to_csv(os.path.join(outdir, f'zone_{zone_id}_degs_male.csv'), index=True)
    deg_male[(deg_male['pvals_adj'] < 0.05) & (deg_male['logFC'] > 0)].sort_values('logFC', ascending=False).to_csv(
        os.path.join(indir, f'zone_{zone_id}_degs_male_up.csv'), index=False
    )
    
for i in range(n_zones):
    zone_id = str(i+1)
    dem_female = adata_desi_female.uns['zones'][zone_id]
    dem_male = adata_desi_male.uns['zones'][zone_id]
    
    # dem_female.to_csv(os.path.join(indir, f'zone_{zone_id}_dems_female.csv'), index=True)
    dem_female[(dem_female['pvals_adj'] < 0.05) & (dem_female['logFC'] > 0)].sort_values('logFC', ascending=False).to_csv(
        os.path.join(indir, f'zone_{zone_id}_dems_female_up.csv'), index=False
    )   
    
    # dem_male.to_csv(os.path.join(indir, f'zone_{zone_id}_dems_male.csv'), index=True)
    dem_male[(dem_male['pvals_adj'] < 0.05) & (dem_male['logFC'] > 0)].sort_values('logFC', ascending=False).to_csv(
        os.path.join(indir, f'zone_{zone_id}_dems_male_up.csv'), index=False    
    )

del zone_id


# %%
# -------------------------------------------
#  Fig3&4: feature set enrichment analysis
# -------------------------------------------

# GSEA pooled per sex
import gseapy as gp

def get_enrichr(deg_df, title=None, ax1=None, ax2=None, show_plot=False):
    """
    GSEA Enrichr analysis per zone
    """
    degs_up = deg_df[
        (deg_df['pvals_adj'] < 0.05) & (deg_df['logFC'] > 0)
    ]['gene'].tolist()

    enr_up = gp.enrichr(
        degs_up,
        gene_sets='GO_Biological_Process_2021',
        outdir=None
    )
    enr_up.res2d.Term = enr_up.res2d.Term.str.split(" \(GO").str[0]

    enr_up.res2d['UP_DW'] = "UP"
    enr_res = enr_up.res2d.head(10).copy()

    if show_plot:
        ax1 = gp.dotplot(
            enr_res, figsize=(5, 8),
            x='UP', x_order = ["UP"],
            cmap = 'Reds', size=5, show_ring=True, 
            ax=ax1, title=f'GSEA GO_BP\n{title}'
        )

        ax2 = gp.barplot(
            enr_res, figsize=(5, 8),
            group ='UP', title ="GSEA GO_BP\n{}".format(title),
            ax=ax2, color = ['r']
        )

    return (enr_res, ax1, ax2) if show_plot else enr_res

# %%
# Comparison of zone 1 vs. the rests
zone_id = '1'
fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=300)
gsea_female = get_enrichr(
    adata_xenium_female.uns['zones'][zone_id],
    ax1=axes[0, 0], ax2=axes[1, 0], title=f'Female zone_{zone_id}',
    show_plot=True
)

gsea_male = get_enrichr(
    adata_xenium_male.uns['zones'][zone_id],
    ax1=axes[0, 1], ax2=axes[1, 1], title=f'Male zone_{zone_id}',
    show_plot=True
)
fig.tight_layout()
# fig.savefig(os.path.join(outdir, f'Fig3_GSEA_zone_{zone_id}.pdf'), bbox_inches='tight') 
gc.collect()


# %%
# Post-hoc analysis: zone 2 (PV) vs. zone 3 (CV)
# - joint DE analysis
# - Sex-specific DE analysis
def get_DE_features(adata, zone_label, feature_name='gene'):
    df = sc.get.rank_genes_groups_df(adata, group=zone_label)
    df = df.sort_values('scores', ascending=False).reset_index(drop=True)

    df = df.loc[:, ['names', 'scores', 'pvals_adj', 'logfoldchanges']]
    df.columns = [feature_name, 'TS', 'pvals_adj', 'logFC']

    adata.uns['zones'][str(zone_label)] = df
    adata.uns['zones']['names'][str(zone_label)] = df.iloc[:, 0].values
    adata.uns['zones']['scores'][str(zone_label)] = df.iloc[:, 1].values     
    return None

adata_xenium_female_zone_23 = adata_xenium_female[adata_xenium_female.obs['zone'].isin([2, 3])].copy()
sc.tl.rank_genes_groups(
    adata_xenium_female_zone_23,
    groupby='zone', method='wilcoxon'
)
adata_xenium_female_zone_23.uns['zones'] = {'names': {}, 'scores': {}}

get_DE_features(
    adata_xenium_female_zone_23, zone_label='2'
)
get_DE_features(
    adata_xenium_female_zone_23, zone_label='3'
)

adata_xenium_male_zone_23 = adata_xenium_male[adata_xenium_male.obs['zone'].isin([2, 3])].copy()
sc.tl.rank_genes_groups(
    adata_xenium_male_zone_23,
    groupby='zone', method='wilcoxon'
)
adata_xenium_male_zone_23.uns['zones'] = {'names': {}, 'scores': {}}

get_DE_features(
    adata_xenium_male_zone_23, zone_label='2'
)
get_DE_features(
    adata_xenium_male_zone_23, zone_label='3'
)

adata_zone_23 = ad.concat([adata_xenium_female_zone_23, adata_xenium_male_zone_23])
sc.tl.rank_genes_groups(
    adata_zone_23,
    groupby='zone', method='wilcoxon'
)
adata_zone_23.uns['zones'] = {'names': {}, 'scores': {}}

get_DE_features(
    adata_zone_23, zone_label='2'
)
get_DE_features(
    adata_zone_23, zone_label='3'
)


# %%
# Sex-specific divergence (zone 2 & 3)
zone_id = '2'
fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=300)
gsea_female = get_enrichr(
    adata_xenium_female_zone_23.uns['zones'][zone_id],
    ax1=axes[0, 0], ax2=axes[1, 0], title=f'Female zone_{zone_id}',
    show_plot=True
)

gsea_male = get_enrichr(
    adata_xenium_male_zone_23.uns['zones'][zone_id],
    ax1=axes[0, 1], ax2=axes[1, 1], title=f'Male zone_{zone_id}',
    show_plot=True
)
fig.tight_layout()
fig.savefig(os.path.join(outdir, f'Fig3_GSEA_zone_{zone_id}.pdf'), bbox_inches='tight') 

# %%
zone_id = '3'
fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=300)
gsea_female = get_enrichr(
    adata_xenium_female_zone_23.uns['zones'][zone_id],
    ax1=axes[0, 0], ax2=axes[1, 0], title=f'Female zone_{zone_id}',
    show_plot=True
)

gsea_male = get_enrichr(
    adata_xenium_male_zone_23.uns['zones'][zone_id],
    ax1=axes[0, 1], ax2=axes[1, 1], title=f'Male zone_{zone_id}',
    show_plot=True
)
fig.tight_layout()
fig.savefig(os.path.join(outdir, f'Fig3_GSEA_zone_{zone_id}.pdf'), bbox_inches='tight') 

# %%
# Joint convergence analysis (zone 2 & 3)
zone_id = '2'
fig, axes = plt.subplots(2, 1, figsize=(10, 20), dpi=300)
gsea_joint = get_enrichr(
    adata_zone_23.uns['zones'][zone_id],
    ax1=axes[0], ax2=axes[1], title=f'Joint zone_{zone_id}',
    show_plot=True
)
fig.tight_layout()
fig.savefig(os.path.join(outdir, f'Fig3_GSEA_joint_zone_{zone_id}.pdf'), bbox_inches='tight') 

zone_id = '3'
fig, axes = plt.subplots(2, 1, figsize=(10, 20), dpi=300)
gsea_joint = get_enrichr(
    adata_zone_23.uns['zones'][zone_id],
    ax1=axes[0], ax2=axes[1], title=f'Joint zone_{zone_id}',
    show_plot=True
)
fig.tight_layout()
fig.savefig(os.path.join(outdir, f'Fig3_GSEA_joint_zone_{zone_id}.pdf'), bbox_inches='tight') 

del adata_zone_23, adata_xenium_female_zone_23, adata_xenium_male_zone_23
gc.collect()


# %% 
# Metabolite (MSEA) preparation for MetaboAnalyst
import re
def clean_metabolite_list(raw_text):
    """
    Parses raw copy-paste metabolite data, removes m/z and adduct info,
    splits composite entries, and standardizes lipid names to Class(Chains).
    """
    
    # 1. Split raw text into lines
    lines = raw_text.strip().split('\n')
    cleaned_list = []

    # Regex patterns
    # Matches lines that are just numbers (m/z) or "x"
    noise_pattern = re.compile(r'^(\d+\.?\d*(\s*m/z.*)?|x)$', re.IGNORECASE)
    
    # Matches adducts like [M+H]+, [M+Na]+ to remove them
    adduct_pattern = re.compile(r'\[M.*?\]\+?')
    
    # Matches "Class Chains" (e.g. PC 34:1) to convert to PC(34:1)
    # Looks for Word followed by Space followed by Digit:Digit
    lipid_format_pattern = re.compile(r'^([A-Za-z0-9]+)\s+([OP]?\-?\d+:\d+.*)$')

    for line in lines:
        line = line.strip()
        if not line: 
            continue
            
        # Skip noise lines (m/z values, ppm errors)
        if noise_pattern.match(line):
            continue

        # Remove adduct info (e.g., [M+H]+)
        line = adduct_pattern.sub('', line)

        # Handle splitters: |, /, " or ", " + "
        # We split by | first, then handle the slash carefully
        # (Slash can be a separator between metabolites OR part of a chain like 18:0/18:1)
        
        # Strategy: Split by '|' or '+' first as these definitely separate distinct species
        initial_splits = re.split(r'\||\+| or ', line)
        
        for item in initial_splits:
            item = item.strip()
            if not item: continue
            
            # Identify composite items separated by '/' that are NOT lipid chains
            # Logic: If we see "Name1/Name2", split. 
            # If we see "18:0/18:1", keep it.
            
            # Simple heuristic: Split on '/' if it is between letters (e.g. Inositol/Galactose)
            # or if it separates two clearly defined lipids (PC 34:1/PE 34:1)
            # We assume if it looks like "Class Chain/Class Chain", it needs splitting.
            
            if re.search(r'[A-Za-z].*/.*[A-Za-z]', item) and not re.search(r'\d:\d/\d:\d', item):
                 sub_items = item.split('/')
            else:
                 sub_items = [item]

            for sub in sub_items:
                sub = sub.strip()
                
                # Cleanup specific noise like trailing "; T" or "; G" found in input
                sub = re.sub(r';\s*[TG]$', '', sub)
                
                # Standardize format: "PC 34:1" -> "PC(34:1)"
                match = lipid_format_pattern.match(sub)
                if match:
                    lipid_class = match.group(1)
                    lipid_chain = match.group(2)
                    # Remove spaces in chain definition if any
                    lipid_chain = lipid_chain.replace(' ', '')
                    final_name = f"{lipid_class}({lipid_chain})"
                else:
                    final_name = sub

                if final_name and final_name not in cleaned_list:
                    cleaned_list.append(final_name)

    return cleaned_list

raw_data_input = """
"""

cleaned = clean_metabolite_list(raw_data_input)
for feature in cleaned:
    print(feature)
del feature


# %%
# -----------------------------------------
#   Fig5: phenotype & metabolic dynamics
# -----------------------------------------
# %%
# all_celltype_dynamics_df = pd.concat(celltype_dynamics, axis=0)
# _, phenotype_test_assocs = test_assoc.get_test_associations(all_celltype_dynamics_df)
phenotype_test_assocs

# %% [markdown]
# Phenotype distributions (PV-CV) doesn't show sex-specific patterns, with only limited 
# differences in Kupffers & LSECs.

# %%
fig, ax = plot.disp_stacked_dynamics(
    all_celltype_dynamics_df.loc[
        :,
        all_celltype_dynamics_df.columns[:-3]
    ].groupby(level=0).mean(),
    figsize=(8, 3),
    title='Cell-type Dynamics'
)
fig.savefig(os.path.join(outdir, 'Fig5_stacked_dynamics.pdf'), bbox_inches='tight')

# %%
# TODO: signatures / pathways dynamics along trajectory specific to phenotypes
# e.g. activated T? 



# %%


