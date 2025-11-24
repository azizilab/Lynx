# %%
# ----------------------
#  Downstream analysis
# ----------------------

# %%
import os
import sys
import gc

import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('..')
from util import IO, utils, plot, test_assoc, trajectory

# %%
from IPython.display import display

from matplotlib import rcParams
from matplotlib.axes import Axes
rcParams['font.family'] = 'Liberation Sans'
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 150})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')

# %%
%load_ext autoreload
%autoreload 2

# %%
# Load data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
indir = '../results/'
outdir = '../results/liver/downstream/gradient'


# %%
# ---------------------------------
#  I. Trajectory analysis
# ---------------------------------
n_latent = 6
n_zones = 5
n_bins = 50
sample_id = 'NIH_F5_proseg'

# Binned expression per sample
print('Analyzing {}...'.format(sample_id))
adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')

qs = np.load('../results/liver/LYNX_xenium_6_debug.npy')
qz = np.load('../results/liver/LYNX_desi_6_debug.npy')

adata_xenium.obsm['X_z'] = qs
adata_desi.obsm['X_z'] = qz

curve = trajectory.get_curve(adata_xenium, epg_lambda=0.01)
trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

curve = trajectory.get_curve(adata_desi, epg_lambda=0.01)       
trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine ')

sc.pp.normalize_total(adata_xenium)
sc.pp.log1p(adata_xenium)

# Compute discrete zonations
utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=n_zones, sample_id=sample_id,
    abundance_test=True, show=True
)

# Compute feature dynamics along trajectory
# Sorting & binning genes
indices = np.argsort(adata_xenium.obs['t']).values
gexp_df = utils.get_binned_expr(
    adata_xenium.to_df().iloc[indices].T,
    n_bins=n_bins,
)

gamma = utils.get_binned_expr(
    pd.DataFrame(adata_xenium.obs['t'].sort_values()).T,
    n_bins=n_bins
).values.flatten()
gexp_df = gexp_df.T
gexp_df['t'] = gamma

del indices, gamma
gc.collect()

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

mexp_df = mexp_df.T
mexp_df['t'] = gamma

# Compute phenotype dynamics along the trajectory
celltype_dynamics_df = utils.get_celltype_dynamics(adata_xenium, adata_xenium.obs['cell_type'], n_bins=n_bins)
del indices, gamma
gc.collect()

# %%
utils.get_zonation_features(    
    adata_xenium, adata_desi,
    abundance_test=True,
    n_zones=n_zones, sample_id=sample_id,
    show=True
)

# %%
from importlib import reload
reload(plot)

# %%
# Visualization
# phenotype dynamics along the trajectory
celltype_dynamics_df = utils.get_celltype_dynamics(adata_xenium, adata_xenium.obs['subtype'], n_bins=n_bins)
plot.disp_celltype_dynamics(celltype_dynamics_df)
plot.disp_feature_dynamics(celltype_dynamics_df.T, figsize=(6, 3), feature='Endothelial')
plot.disp_feature_dynamics(celltype_dynamics_df.T, figsize=(6, 3), feature='LSECs')
plot.disp_feature_dynamics(celltype_dynamics_df.T, figsize=(6, 3), feature='SMCs')
plot.disp_feature_dynamics(celltype_dynamics_df.T, figsize=(6, 3), feature='T-cells')


# %% [markdown]
# ----------------------------------
#  II.  Sex-specific joint analysis
# ----------------------------------

# (1). Continuous statistical test: joint analysis of significant features
#  - along PV - CV trajectory
#  - sex-specific

# %%
sample_ids = sorted([
    sample_id for sample_id in os.listdir(xenium_path)
    if os.path.isdir(os.path.join(xenium_path, sample_id))
    and len(sample_id.split('_')) == 2
])
sample_ids = sample_ids[1:] # exclude NIH_F1 (outlier)

# Update  metabolite m/z w/ annotations
metabolite_annots_df = pd.read_csv('../data/metabolite_annotations_pos_mode.csv')
metabolite_dict = {
    k: v for k, v in zip(metabolite_annots_df.iloc[:, 0], metabolite_annots_df.iloc[:, 1])
    if not pd.isna(v)
}
del metabolite_annots_df

n_latent = 6
n_zones = 3
n_bins = 50
cluster_key = 'cell_type'

# Binned expression per sample
gexps = [] 
mexps = []
celltype_dynamics = []

adatas_xenium = []
adatas_desi = []

for sample_id in sample_ids:
    print('Computing for  {}...'.format(sample_id))
    adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
    adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
    adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')


    qs = np.load(os.path.join(outdir, f'LYNX_{sample_id}_xenium_latent.npy'))
    qz = np.load(os.path.join(outdir, f'LYNX_{sample_id}_desi_latent.npy'))
    
    adata_xenium.obsm['X_z'] = qs
    adata_desi.obsm['X_z'] = qz

    curve = trajectory.get_curve(adata_xenium)
    trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

    curve = trajectory.get_curve(adata_desi)
    trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine ')

    sc.pp.normalize_total(adata_xenium)
    sc.pp.log1p(adata_xenium)

    utils.get_zonation_features(
        adata_xenium, adata_desi,
        n_zones=n_zones, sample_id=sample_id,
        abundance_test=True, show=False
    )

    # Compute feature dynamics along trajectory
    # Sorting & binning genes
    indices = np.argsort(adata_xenium.obs['t']).values
    gexp_df = utils.get_binned_expr(
        adata_xenium.to_df().iloc[indices].T,
        n_bins=n_bins,
    )

    gamma = utils.get_binned_expr(
        pd.DataFrame(adata_xenium.obs['t'].sort_values()).T,
        n_bins=n_bins
    ).values.flatten()

    gexp_df = gexp_df.T
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

    mexp_df = mexp_df.T
    mexp_df['t'] = gamma
    mexp_df['sample_id'] = sample_id
    mexp_df['sex'] = 'M' if 'M' in sample_id else 'F'

    mexps.append(mexp_df)

    # Compute phenotype dynamics along the trajectory
    # TODO: re-annotate cell types for M2
    # celltype_dynamics_df = utils.get_celltype_dynamics(adata_xenium, adata_xenium.obs[cluster_key], n_bins=n_bins)
    # celltype_dynamics_df['t'] = gamma
    # celltype_dynamics_df['sample_id'] = sample_id
    # celltype_dynamics_df['sex'] = 'M' if 'M' in sample_id else 'F'
    # celltype_dynamics.append(celltype_dynamics_df)

    adatas_xenium.append(adata_xenium)
    adatas_desi.append(adata_desi)

    del adata_xenium, adata_desi, gamma, gexp_df, mexp_df
    del sample_id, indices
    gc.collect()


# %%
# Extract significant features with
# Linear-mixed effect models on each feature
all_gexp_df = pd.concat(gexps, axis=0)
fitted_gexp_df, gene_test_assocs = test_assoc.get_test_associations(all_gexp_df)
all_mexp_df = pd.concat(mexps, axis=0)
fitted_mexp_df, metabolite_test_assocs = test_assoc.get_test_associations(all_mexp_df)
gc.collect()


# %%
# tmp: avg. intensity per zone per sample
for i in range(len(sample_ids)):
    adata_xenium = adatas_xenium[i]

    zone_means = []
    for zone in range(n_zones):
        cells_in_zone = adata_xenium.obs['zone'] == str(zone+1)
        mean_expr = adata_xenium[cells_in_zone].X.toarray().mean(axis=0)
        zone_means.append(mean_expr)
    zone_means = pd.DataFrame(
        np.array(zone_means),
        columns=adata_xenium.var_names,
        index=[f'Zone_{z}' for z in range(n_zones)]
    ).T
    zone_means.to_csv(os.path.join(outdir, f'{sample_ids[i]}_xenium_zone_means.csv'))
    del adata_xenium, cells_in_zone, mean_expr, zone_means
    
    adata_desi = adatas_desi[i]
    zone_means = []
    for zone in range(n_zones):
        cells_in_zone = adata_desi.obs['zone'] == str(zone+1)
        mean_expr = adata_desi[cells_in_zone].X.toarray().mean(axis=0)
        zone_means.append(mean_expr)
    zone_means = pd.DataFrame(
        np.array(zone_means),
        columns=adata_desi.var_names,
        index=[f'Zone_{z}' for z in range(n_zones)]
    ).T
    features = [
        metabolite_dict[c] if c in metabolite_dict else c
        for c in zone_means.index
    ]
    zone_means.index = features

    zone_means.to_csv(os.path.join(outdir, f'{sample_ids[i]}_desi_zone_means.csv'))
    del adata_desi, cells_in_zone, mean_expr, zone_means
    
    gc.collect()
    

# %%
zone_means.head()




# %%
# -------------------------------------------------------------
#. Figure 3&4: Summary of spatial gradients along trajectory
# -------------------------------------------------------------

# %%
# Visualize sample trajectory & zones
for i in range(len(sample_ids)):
    ax = sq.pl.spatial_scatter(
        adatas_xenium[i], color='t', size=20,
        cmap='RdBu_r', img=False, colorbar=False, return_ax=True,
        title=f'Spatial gradient - {sample_ids[i]}'
    )
    sm = ax.collections[0]
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label('PV → CV', fontsize=10)
    plt.savefig(os.path.join(outdir, f'spatial_gradient_{sample_ids[i]}.png'), bbox_inches='tight')

    ax = sq.pl.spatial_scatter(
        adatas_xenium[i], color='zone', size=20,
        cmap='RdBu_r', img=False, return_ax=True,
        title=f'Zonation - {sample_ids[i]}'
    )
    plt.savefig(os.path.join(outdir, f'spatial_zone_{sample_ids[i]}.png'), bbox_inches='tight')

del ax
gc.collect()


# %%
# (1). Continuous gradients
def plot_binned_expr_gradient(
    binned_df, features_to_annot=None, title=None,
    dpi=100, pad=0.25, figsize=(12, 8), cmap='RdBu_r',
    show=True
):
    """
    Plot binned expression along the gradient with genes sorted by their peak position.
    
    Parameters:
    -----------
    binned_df : pd.DataFrame
        DataFrame with N_BINS as rows, genes as columns (except last column which is the gradient)
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the heatmap
    """
    # Extract expression data and gradient
    expr_data = binned_df.iloc[:, :-1]  # All columns except last (gradient)
    gradient = binned_df.iloc[:, -1].values  # Last column is the gradient
    
    # Find argmax position for each gene (along rows/bins)
    gene_argmax_positions = expr_data.idxmax(axis=0)
    
    # Sort genes by their argmax position (PV->CV direction)
    sorted_indices = np.argsort(gene_argmax_positions)
    sorted_genes = expr_data.columns[sorted_indices]
    
    # Create the sorted expression matrix (transpose for proper orientation)
    sorted_expr = expr_data[sorted_genes].T  # Transpose to have genes as rows

    # Z-score normalize each gene (row) across bins
    sorted_expr = (sorted_expr - sorted_expr.values.mean(axis=1, keepdims=True)) / (sorted_expr.values.std(axis=1, keepdims=True) + 1e-8)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(sorted_expr, cmap=cmap, aspect='auto')
    
    # Main colorbar for expression
    cbar1 = plt.colorbar(im, ax=ax, shrink=0.8, pad=pad)
    cbar1.set_label('Expression (zscore)', fontsize=10)
    
    # Create second colorbar at bottom for gradient
    import matplotlib.cm as cm
    gradient_norm = plt.Normalize(vmin=0, vmax=1)
    gradient_cmap = cm.get_cmap('turbo')
    gradient_mappable = cm.ScalarMappable(norm=gradient_norm, cmap=gradient_cmap)
    
    # Add colorbar at bottom
    ax_pos = ax.get_position()
    cbar2_ax = fig.add_axes([ax_pos.x0-0.075, ax_pos.y0 - 0.12, 
                            ax_pos.width*1.2, 0.01])
    cbar2 = plt.colorbar(
        gradient_mappable, 
        cax=cbar2_ax, 
        orientation='horizontal',
    )
    cbar2.set_label('PV → CV Gradient', fontsize=10)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('Genes', fontsize=12)
    ax.set_xlabel('')
    ax.set_title(title, fontsize=15)

    # Annotate specific genes if provided
    if features_to_annot is not None:
        # Get positions of features to annotate
        feature_positions = []
        for feature in features_to_annot:
            if feature in sorted_expr.index:
                pos = list(sorted_expr.index).index(feature)
                feature_positions.append((feature, pos))
        
        if feature_positions:
            # Sort by position to handle overlaps
            feature_positions.sort(key=lambda x: x[1])
            
            # Calculate minimum spacing to avoid overlap
            min_spacing = len(sorted_expr) * .03  # 3% of total height
            
            # Adjust positions to avoid overlaps
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
            
            # Feature annotations
            for feature, original_pos, adjusted_pos in adjusted_positions:
                # Always draw a line if there's any adjustment OR for consistency
                if abs(original_pos - adjusted_pos) > 0.1:  # Lower threshold
                    # Draw line from original gene position to adjusted text position
                    ax.annotate('', 
                               xy=(len(sorted_expr.columns) - 0.5, original_pos), 
                               xytext=(len(sorted_expr.columns) + 1.5, adjusted_pos),
                               arrowprops=dict(arrowstyle='-', color='black', lw=0.8, alpha=0.7))
                    # Add text at adjusted position
                    ax.text(len(sorted_expr.columns) + 2, adjusted_pos, feature, 
                           va='center', ha='left', fontsize=8, weight='bold')
                else:
                    # Add text directly with a short line for consistency
                    ax.annotate('', 
                               xy=(len(sorted_expr.columns) - 0.5, original_pos), 
                               xytext=(len(sorted_expr.columns) + 1.5, original_pos),
                               arrowprops=dict(arrowstyle='-', color='black', lw=0.8, alpha=0.7))
                    ax.text(len(sorted_expr.columns) + 2, original_pos, feature, 
                           va='center', ha='left', fontsize=8, weight='bold')
            
    fig.tight_layout()
    if show:
        plt.show()
        return None
    else:
        return fig, ax

# %%
trajectory_genes = gene_test_assocs[gene_test_assocs['trajectory_feature'] == 1].index

male_gexps_df = fitted_gexp_df[fitted_gexp_df['sex'] == 'M'].copy()
male_gexps_df.reset_index(inplace=True)
numeric_cols = male_gexps_df.select_dtypes(include=[np.number]).columns
male_gexps_df = male_gexps_df.groupby('index')[numeric_cols].mean()
male_gexps_df.drop('index', axis=1, inplace=True)
male_gexps_df = male_gexps_df[trajectory_genes]

female_gexps_df = fitted_gexp_df[fitted_gexp_df['sex'] == 'F'].copy()
female_gexps_df.reset_index(inplace=True)  
numeric_cols = female_gexps_df.select_dtypes(include=[np.number]).columns
female_gexps_df = female_gexps_df.groupby('index')[numeric_cols].mean()
female_gexps_df.drop('index', axis=1, inplace=True)
female_gexps_df = female_gexps_df[trajectory_genes]

# %%
trajectory_metabolites = metabolite_test_assocs[metabolite_test_assocs['trajectory_feature'] == 1].index

male_mexps_df = fitted_mexp_df[fitted_mexp_df['sex'] == 'M'].copy()
male_mexps_df.reset_index(inplace=True)
numeric_cols = male_mexps_df.select_dtypes(include=[np.number]).columns
male_mexps_df = male_mexps_df.groupby('index')[numeric_cols].mean()
male_mexps_df.drop('index', axis=1, inplace=True)
male_mexps_df = male_mexps_df[trajectory_metabolites]

female_mexps_df = fitted_mexp_df[fitted_mexp_df['sex'] == 'F'].copy()
female_mexps_df.reset_index(inplace=True)  
numeric_cols = female_mexps_df.select_dtypes(include=[np.number]).columns
female_mexps_df = female_mexps_df.groupby('index')[numeric_cols].mean()
female_mexps_df.drop('index', axis=1, inplace=True)
female_mexps_df = female_mexps_df[trajectory_metabolites]


# %%
top_sex_genes = gene_test_assocs[gene_test_assocs['pval.sex'] < .05].index
fig1, _ = plot_binned_expr_gradient(
    male_gexps_df,  top_sex_genes,
    figsize=(8, 8), cmap='seismic', pad=0.15, dpi=300, show=False,
    title='Pooled gene expression along \nPV-CV axis (Male)'
)
fig2, _ = plot_binned_expr_gradient(
    female_gexps_df, top_sex_genes,
    figsize=(8, 8), cmap='seismic', pad=0.15, dpi=300, show=False,
    title='Pooled gene expression along \nPV-CV axis (Female)'
)
del top_sex_genes
gc.collect()

# %%
fig1.savefig(os.path.join(outdir, 'Fig3_gene_gradient_male.svg'), bbox_inches='tight')
fig2.savefig(os.path.join(outdir, 'Fig3_gene_gradient_female.svg'), bbox_inches='tight')


# %%
top_sex_metabolites = metabolite_test_assocs.sort_values('adj-pval.sex').head(20).index

# Update with annotations
top_sex_metabolites = [
    metabolite_dict[c] if c in metabolite_dict else c
    for c in top_sex_metabolites
]
male_mexps_df.columns = [
    metabolite_dict[c] if c in metabolite_dict else c
    for c in male_mexps_df.columns 
]
female_mexps_df.columns = [
    metabolite_dict[c] if c in metabolite_dict else c
    for c in female_mexps_df.columns
]

fig1, _ = plot_binned_expr_gradient(
    male_mexps_df, top_sex_metabolites,
    figsize=(8, 8), cmap='seismic', pad=0.25, dpi=300, show=False,
    title='Pooled metabolite intensity \nalong PV-CV axis (Male)'
)
fig2, _ = plot_binned_expr_gradient(
    female_mexps_df, top_sex_metabolites,
    figsize=(8, 8), cmap='seismic', pad=0.25, dpi=300, show=False,
    title='Pooled metabolite intensity \nalong PV-CV axis (Female)'
)

del top_sex_metabolites
gc.collect()

# %%
fig1.savefig(os.path.join(outdir, 'Fig4_metabolite_gradient_male.svg'), bbox_inches='tight')
fig2.savefig(os.path.join(outdir, 'Fig4_metabolite_gradient_female.svg'), bbox_inches='tight')


# %%
# ---------------------------------------------
#  LMEs for sex & dynasmics statistical tests
# ---------------------------------------------
# (1). Features with sex-dependent dynamics
sex_genes = gene_test_assocs[gene_test_assocs['adj-pval.sex'] < .05].index
print('sex-disparity genes')
print('===================================')
idx = 0
ncols = 4

while idx < len(sex_genes):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(sex_genes):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                fitted_gexp_df, 
                feature=sex_genes[idx], 
                ax=ax, show=False
            )
        idx += 1
    plt.show()
print('\n\n')

other_genes = gene_test_assocs[gene_test_assocs['adj-pval.sex'] >= .05].index
print('Other genes:')
print('===================================')
idx = 0
ncols = 4

while idx < len(other_genes):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(other_genes):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                fitted_gexp_df, 
                feature=other_genes[idx], 
                ax=ax, show=False
            )
        idx += 1
    plt.show()
print('\n\n')


# %% Plot example sex-differential genes
feature = 'IGF1'
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
ax = plot.disp_sex_feature_dynamics(
    fitted_gexp_df, feature=feature,
    ax=ax, show=False
)
fig.savefig(os.path.join(outdir, f'{feature}_sex_diff.svg'), bbox_inches='tight')

# %%
fitted_mexp_df.columns = [
    metabolite_dict[c] if c in metabolite_dict else c
    for c in fitted_mexp_df.columns
]
fitted_mexp_df = fitted_mexp_df.loc[:,~fitted_mexp_df.columns.duplicated()].copy()
sex_metabolites = metabolite_test_assocs[metabolite_test_assocs['adj-pval.sex'] < .05].index
sex_metabolites = [
    metabolite_dict[c] if c in metabolite_dict else c
    for c in sex_metabolites
]
sex_metabolites = np.unique(sex_metabolites)

# %%
print('sex-disparity metabolites')
print('===================================')
idx = 0
ncols = 4

while idx < len(sex_metabolites):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(sex_metabolites):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                fitted_mexp_df, feature=sex_metabolites[idx], 
                ax=ax, show=False, 
            )
        idx += 1
    plt.show()

other_metabolites = metabolite_test_assocs[metabolite_test_assocs['adj-pval.sex'] < .05].index
print('Other metabolites')
print('===================================')
idx = 0
ncols = 4

while idx < len(other_metabolites):
    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    for ax in axes:
        if idx >= len(other_metabolites):
            ax.axis('off')
        else:
            ax = plot.disp_sex_feature_dynamics(
                fitted_mexp_df, feature=other_metabolites[idx], 
                ax=ax, show=False
            )
        idx += 1
    plt.show()


# %% Plot example sex-differential metabolites
feature = 'TG 48:1'
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
ax = plot.disp_sex_feature_dynamics(
    fitted_mexp_df, feature=feature,
    ax=ax, show=False
)
fig.savefig(os.path.join(outdir, f'{feature}_sex_diff.svg'), bbox_inches='tight')

feature = 'DG 42:6'
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
ax = plot.disp_sex_feature_dynamics(
    fitted_mexp_df, feature=feature,
    ax=ax, show=False
)
fig.savefig(os.path.join(outdir, f'{feature}_sex_diff.svg'), bbox_inches='tight')


# %%
# Case study: DG/TG molecule distributions across sex
# DG + TG (hypothesis: enrichment in male + CV)
glycerides = metabolite_test_assocs[
    np.logical_or(
        metabolite_test_assocs.index.str.contains('DG'),
        metabolite_test_assocs.index.str.contains('TG')
    )
].index

# %%
# DG/TG sex-dependent coefficients vs. randomized samples
glycerides_test_assocs = metabolite_test_assocs.loc[glycerides].copy()
glycerides_test_assocs['Category'] = 'glycerides'

random_test_assocs = metabolite_test_assocs.loc[
    np.random.choice(metabolite_test_assocs.index, len(glycerides), replace=False)
].copy()
random_test_assocs['Category'] = 'random'

glycerides_test_assocs = pd.concat((glycerides_test_assocs, random_test_assocs))
glycerides_test_assocs.head()

from statannotations.Annotator import Annotator
rcParams.update({'font.size': 10})

fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
sns.violinplot(glycerides_test_assocs, x='Category', y='coeff.sex',  linewidth=1.5, palette='seismic', ax=ax)
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xlabel('Metabolite Category')
ax.set_ylabel('Regression coefficients\n (Male > Female)')


pairs = [('glycerides', 'random')]
annotator = Annotator(
    ax, pairs, data=glycerides_test_assocs, x='Category', y='coeff.sex',
)
annotator.configure(test='Mann-Whitney', text_format='full', loc='outside')
annotator.apply_and_annotate()
fig.suptitle('Sex-specific abundance (Glycerides)', fontsize=14, y=1.02)
fig.show()

# %%
# Save sex-dependent & trajectory dependent test statistics
gene_test_assocs.to_csv(os.path.join(outdir, 'gene_test_assocs.csv'), index=True)
    
# Update metabolite +/- annotations
ion_modes = [
    IO.check_ion_mode(
        ion, 
        pos_path='../data/desi/desi_2d/pos/NIH_F1.ome.tif',
        neg_path='../data/desi/desi_2d/neg/NIH_F1.ome.tif',
    )
    for ion in metabolite_test_assocs.index
]
metabolite_test_assocs['+/-'] = ion_modes
metabolite_test_assocs.to_csv(os.path.join(outdir, 'metabolite_test_assocs.csv'), index=True)


# %%
# (2). Zonation markers pooled across sex
adata_xenium_female = sc.concat([
    adatas_xenium[i] for i in range(len(sample_ids))
    if 'F' in sample_ids[i]
])
adata_desi_female = sc.concat([
    adatas_desi[i] for i in range(len(sample_ids))
    if 'F' in sample_ids[i]
])
adata_desi_female.var_names = [
    metabolite_dict[c] if c in metabolite_dict else c
    for c in adata_desi_female.var_names
]
adata_desi_female.var_names_make_unique()

utils.get_zonation_features(
    adata_xenium_female, adata_desi_female, n_zones=3, 
    sample_id='Pooled Female', abundance_test=True, show=True
)

adata_xenium_male = sc.concat([
    adatas_xenium[i] for i in range(len(sample_ids))
    if 'M' in sample_ids[i]
])
adata_desi_male = sc.concat([
    adatas_desi[i] for i in range(len(sample_ids))
    if 'M' in sample_ids[i]
])
adata_desi_male.var_names = [
    metabolite_dict[c] if c in metabolite_dict else c
    for c in adata_desi_male.var_names
]
adata_desi_male.var_names_make_unique()

utils.get_zonation_features(
    adata_xenium_male, adata_desi_male, n_zones=3, 
    sample_id='Pooled Male', abundance_test=True, show=True
)

gc.collect()


# %%
# Differential expression analysis using scanpy
for i in range(n_zones):
    zone_id = str(i+1)
    deg_female = adata_xenium_female.uns['zones'][zone_id]
    deg_male = adata_xenium_male.uns['zones'][zone_id]
    deg_female.to_csv(os.path.join(outdir, f'zone_{zone_id}_degs_female.csv'), index=True)
    deg_male.to_csv(os.path.join(outdir, f'zone_{zone_id}_degs_male.csv'), index=True)

for i in range(n_zones):
    zone_id = str(i+1)
    dem_female = adata_desi_female.uns['zones'][zone_id]
    dem_male = adata_desi_male.uns['zones'][zone_id]
    dem_female.to_csv(os.path.join(outdir, f'zone_{zone_id}_dems_female.csv'), index=True)
    dem_male.to_csv(os.path.join(outdir, f'zone_{zone_id}_dems_male.csv'), index=True)

del zone_id


# %%
# -------------------------------------------
#  Fig3&4: feature set enrichment analysis
# -------------------------------------------

# GSEA pooled per sex
import networkx as nx
import gseapy as gp

def get_enrichr(deg_df, title=None, ax1=None, ax2=None, show_plot=False):
    """
    GSEA Enrichr analysis per zone
    """
    degs_up = deg_df[
        (deg_df['pvals_adj'] < 0.05) & (deg_df['logFC'] > 0)
    ]['gene'].tolist()
    degs_dw = deg_df[
        (deg_df['pvals_adj'] < 0.05) & (deg_df['logFC'] < 0)
    ]['gene'].tolist()

    enr_up = gp.enrichr(
        degs_up,
        gene_sets='GO_Biological_Process_2021',
        outdir=None
    )
    enr_up.res2d.Term = enr_up.res2d.Term.str.split(" \(GO").str[0]

    enr_dw = gp.enrichr(
        degs_dw,
        gene_sets='GO_Biological_Process_2021',
        outdir=None
    )
    enr_dw.res2d.Term = enr_dw.res2d.Term.str.split(" \(GO").str[0]

    enr_up.res2d['UP_DW'] = "UP"
    enr_dw.res2d['UP_DW'] = "DOWN"
    enr_res = pd.concat([enr_up.res2d.head(10), enr_dw.res2d.head(10)])


    if show_plot:
        # gp.dotplot(
        #     enr_up.res2d, figsize=(5, 8),
        #     cmap='Reds',
        #     title=f'GSEA Enriched \n{title}'
        # ) 
        # Combined plot
        ax1 = gp.dotplot(
            enr_res, figsize=(5, 8),
            x='UP_DW', x_order = ["UP","DOWN"],
            cmap = 'Reds', size=5, show_ring=True, 
            ax=ax1, title=f'GSEA GO_BP\n{title}'
        )

        ax2 = gp.barplot(
            pd.concat([enr_up.res2d.head(), enr_dw.res2d.head()]), figsize=(5, 8),
            group ='UP_DW', title ="GSEA GO_BP\n{}".format(title),
            ax=ax2, color = ['b','r']
        )

        # Network plot
        # nodes, edges = gp.enrichment_map(enr_up.res2d)

        # # build graph
        # G = nx.from_pandas_edgelist(
        #     edges, source='src_idx', target='targ_idx',
        #     edge_attr=['jaccard_coef', 'overlap_coef', 'overlap_genes']
        # )

        # # Add missing node if there is any
        # for node in nodes.index:
        #     if node not in G.nodes():
        #         G.add_node(node)

        # fig, ax = plt.subplots(figsize=(10, 10))
        # pos=nx.layout.spiral_layout(G)
        # nx.draw_networkx_nodes(
        #     G,
        #     pos=pos,
        #     cmap=plt.cm.RdYlBu,
        #     node_size=list(nodes.Hits_ratio *1000)
        # )
        # nx.draw_networkx_labels(
        #     G,
        #     pos=pos,
        #     labels=nodes.Term.to_dict()
        # )
        # edge_weight = nx.get_edge_attributes(G, 'jaccard_coef').values()
        # nx.draw_networkx_edges(
        #     G,
        #     pos=pos,
        #     width=list(map(lambda x: x*10, edge_weight)),
        #     edge_color='#CDDBD4'
        # )
        # plt.title('Enrichr Networks \n{}'.format(title), fontsize=15)
        # plt.show()

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
fig.savefig(os.path.join(outdir, f'Fig3_GSEA_zone_{zone_id}.pdf'), bbox_inches='tight') 
gc.collect()

# %%
adata_xenium_female.uns['zones']['1'].head()


# %%
# Post-hoc analysis: zone 2 (PV) vs. zone 3 (CV)
def get_DE_features(adata, zone_label, feature_name='gene'):
    df = sc.get.rank_genes_groups_df(adata, group=zone_label)
    df = df.sort_values('scores', ascending=False).reset_index(drop=True)

    df = df.loc[:, ['names', 'scores', 'pvals_adj', 'logfoldchanges']]
    df.columns = [feature_name, 'TS', 'pvals_adj', 'logFC']

    adata.uns['zones'][str(zone_label)] = df
    adata.uns['zones']['names'][str(zone_label)] = df.iloc[:, 0].values
    adata.uns['zones']['scores'][str(zone_label)] = df.iloc[:, 1].values     
    return None

adata_xenium_female_zone_23 = adata_xenium_female[adata_xenium_female.obs['zone'].isin(['2', '3'])].copy()
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

adata_xenium_male_zone_23 = adata_xenium_male[adata_xenium_male.obs['zone'].isin(['2', '3'])].copy()
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

# %%
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





# %%
# -----------------------------------------
#   Fig5: phenotype & metabolic dynamics
# -----------------------------------------
# %%
# all_celltype_dynamics_df = pd.concat(celltype_dynamics, axis=0)
# phenotype_test_assocs = test_assoc.get_test_associations(all_celltype_dynamics_df)
# phenotype_test_assocs

