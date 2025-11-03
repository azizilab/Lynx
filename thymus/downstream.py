# %%
# ----------------------
#  Downstream analysis
# ----------------------

# %%
# %%
import os
import sys
import gc

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
from scipy.interpolate import UnivariateSpline
import numpy as np
warnings.filterwarnings('ignore')

# %%
%load_ext autoreload
%autoreload 2


# %%
# Load data & processed latent embeddings
data_path = '../data/thymus/'
sample_ids = sorted([
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
])
sample_id = sample_ids[0]
adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))
adata_protein.var_names_make_unique()

n_latent = 6
adata_rna.obsm['X_z'] = np.load('../results/thymus/lynx_rna_{0}_{1}_new.npy'.format(n_latent, sample_id))
adata_protein.obsm['X_z'] = adata_rna.obsm['X_z'].copy()
# adata_protein.obsm['X_z'] = np.load('../results/thymus/lynx_protein_{0}_{1}.npy'.format(n_latent, sample_id))

# %%
# (1). CMA trajectory inference
curve = trajectory.get_curve(
    adata_rna, 
    use_rep='X_z',
    epg_mu=.01,
    root_marker='Dcn',
)
trajectory.compute_pseudotime(adata_rna, curve)
adata_protein.obs['t'] = adata_rna.obs['t'].values 

plot.disp_trajectory(
    adata_rna, cmap='RdBu',
    title='Principal Curve - LYNX'
)


# %%
# (2). Discrete zonation analysis
if 'milestones_colors' in adata_rna.uns_keys():
    adata_rna.uns.pop('milestones_colors')

if adata_rna.X[adata_rna.X > 0].min() == 1.0:
    sc.pp.normalize_total(adata_rna)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)

utils.get_zonation_features(    
    adata_rna, adata_protein,
    n_zones=4, sample_id=sample_id,
    abundance_test=True, show=True, 
)

# %%
sq.pl.spatial_scatter(
    adata_rna, color='zone',
    size=100, img=False, title='Inferred spatial zones - LYNX'
)

# %%
# (3). Spatial dynamics of cell-type markers & chemokines along the trajectory
# Markers of interest
tec_markers = [
    'Psmb11', 'Ly75', 'Ccl25', 'H2-Aa', 'H2-Ab1',   # Pan-cTEC 
    'Tbata', 'Tp53aip1', 'Dll4', # cTEC subtypes
    'Dlk2', 'Igfbp5', 'Igfbp6', 'Ccn2', 'Ccl2', 'Krt15', 'Itga6', 'Mki67',  # mcTEC subtypes (KTR15 -> Krt15)
    'Epcam', # Pan-mTEC
    'Ascl1', 'Ccl21a',   # mTECI (CCL21 -> Ccl21a)
    'Aire', 'Fezf2', 'Crip1',  # mTECII
    'Slpi', 'Ivl', 'Krt10', 'Cdkn2a'  # mTEC subtypes
]
tec_markers = [m for m in tec_markers if m in adata_rna.var_names]

marophage_markers = [
    'Cd68', 'Cd163', 'Cd11b', 'Cd11c',
    'Timd4', 'Hpgd', 'Serpinb6a', 'Slc40a1', 'Cd81',  # Cortex-enriched Timd4+ markers
    'Cx3cr1', 'Ctsz', 'Cd63', 'Pmepa1',' Zmynd15', # Medulla-enriched Cx3cr1+ markers
]
macrophage_markers = [m for m in marophage_markers if m in adata_rna.var_names]

immune_markers = [
    'Cd3d', 'Cd3e', 'Cd4', 'Cd8a', 'Cd8b1',  # T cells
    'Cd19', 'Ptprc', 'Ighd',  # B cells
    'Cd5', 'Cd27', 'Cd44'   # General thymocytes
]
immune_markers = [m for m in immune_markers if m in adata_rna.var_names]

# %%
chemokines = [
    'Ccl25', 'Cxcl12', 'Ccl19', 'Ccl21', 'Ccl22', 'Cxcl9', 'Cxcl10', 'Cxcl11'
]
chemokines = [m for m in chemokines if m in adata_rna.var_names]
chemokines


# %%
# Create binned expression data along trajectory
# (1). RNA
n_bins = 200
indices = np.argsort(adata_rna.obs['t']).values
gexp_df = utils.get_binned_expr(
    adata_rna.to_df().iloc[indices].T,
    n_bins=n_bins,
)

t = utils.get_binned_expr(
    pd.DataFrame(adata_rna.obs['t'].sort_values()).T,
    n_bins=n_bins
).values.flatten()
gexp_df = gexp_df.T
gexp_df['t'] = t


# %%
# Create heatmap of markers along trajectory
# Get subset of expression data for markers
# marker_data = gexp_df.copy()
# labels = immune_markers + ['t']
# labels = tec_markers + ['t']
labels = macrophage_markers + ['t']
marker_data = gexp_df[labels].copy()
del labels

marker_data_norm = marker_data.copy()
for gene in marker_data.columns:
    if gene != 't':  # Skip the time column
        gene_values = marker_data[gene].values

# Calculate peak positions for each normalized marker to sort them
peak_positions = {}
for gene in marker_data_norm.columns:
    if gene != 't':  # Skip the time column
        peak_idx = np.argmax(marker_data_norm[gene].values)
        peak_positions[gene] = marker_data_norm['t'].iloc[peak_idx]

# Sort markers by their peak positions (early to late along trajectory)
sorted_markers = sorted([col for col in marker_data_norm.columns if col != 't'], 
                       key=lambda x: peak_positions[x])

# %%
# Create heatmap
fig, ax = plt.subplots(figsize=(8, 5))
# heatmap_data = marker_data_norm.T
heatmap_data = marker_data_norm[sorted_markers].T

sns.heatmap(
    heatmap_data, 
    cmap='seismic',
    ax=ax,
    cbar_kws={'label': 'Expressions (Z-score)'},
    xticklabels=False
)


ax.set_xlabel(r'Trajectory ($t$) (Cortex -> Medulla)', fontsize=15)
ax.set_ylabel('Genes', fontsize=15)
ax.set_title('Dynamics of Macrophage Markers', fontsize=20, y=1.1)

# Add trajectory position labels
n_ticks = 5
tick_positions = np.linspace(0, len(marker_data)-1, n_ticks) + 0.5
tick_labels = np.linspace(0, 1, n_ticks)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
#ax.set_yticks(np.arange(len(sorted_markers))+0.5)
#ax.set_yticklabels(sorted_markers, fontsize=10)

plt.tight_layout()
plt.show()


# %%
# (2). Protein
adata_protein_norm = adata_protein.copy()
sc.pp.log1p(adata_protein_norm)
sc.pp.scale(adata_protein_norm)

# %%
# Use standardized protein expression??
n_bins = 200
indices = np.argsort(adata_protein_norm.obs['t']).values
pexp_df = utils.get_binned_expr(
    adata_protein_norm.to_df().iloc[indices].T,
    n_bins=n_bins,
)

t = utils.get_binned_expr(
    pd.DataFrame(adata_protein_norm.obs['t'].sort_values()).T,
    n_bins=n_bins
).values.flatten()
pexp_df = pexp_df.T
pexp_df['t'] = t

# %%
#label = ['CD3', 'CD4', 'CD8a', 'CD31_Pecam', 'CD44', 'CD45R_B220'] + ['t']
#marker_data = pexp_df[label].copy()
marker_data = pexp_df.copy()
#del label

# First z-score normalize each gene
marker_data_norm = marker_data.copy()
for gene in marker_data.columns:
    if gene != 't':  # Skip the time column
        gene_values = marker_data[gene].values

# Calculate peak positions for each normalized marker to sort them
peak_positions = {}
for gene in marker_data_norm.columns:
    if gene != 't':  # Skip the time column
        peak_idx = np.argmax(marker_data_norm[gene].values)
        peak_positions[gene] = marker_data_norm['t'].iloc[peak_idx]

# Sort markers by their peak positions (early to late along trajectory)
sorted_markers = sorted([col for col in marker_data_norm.columns if col != 't'], 
                       key=lambda x: peak_positions[x])

# Create heatmap
fig, ax = plt.subplots(figsize=(8, 10))
heatmap_data = marker_data_norm[sorted_markers].T

sns.heatmap(
    heatmap_data, 
    cmap='seismic',
    ax=ax,
    cbar_kws={'label': 'Expressions (Z-score)'},
    xticklabels=False
)

ax.set_xlabel(r'Trajectory ($t$) (Cortex -> Medulla)', fontsize=15)
ax.set_ylabel('Genes', fontsize=15)
ax.set_title('Dynamics of CITE-seq Protein expressions', fontsize=20)

# Add trajectory position labels
n_ticks = 5
tick_positions = np.linspace(0, len(marker_data)-1, n_ticks) + 0.5
tick_labels = np.linspace(0, 1, n_ticks)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_yticks(np.arange(len(sorted_markers))+0.5)
ax.set_yticklabels(sorted_markers, fontsize=8)

plt.tight_layout()
plt.show()


# %%
# Example dynamics of markers
def disp_feature_dynamics(
    expr_df, 
    features, 
    std_df=None,
    title=None,
    figsize=(6, 2.5),
):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.interpolate import interp1d
    
    # Handle single feature or list of features
    if isinstance(features, str):
        features = [features]
    
    # Set default colors if not provided
    # if colors is None:
    #     colors = plt.cm.tab20(np.linspace(0, 1, len(features)))
    # elif len(colors) < len(features):
    #     colors = colors * (len(features) // len(colors) + 1)
    
    n_bins = expr_df.shape[1]
    xx = np.arange(n_bins)

    plt.figure(figsize=figsize)
    
    for i, feature in enumerate(features):
        if feature not in expr_df.index:
            print(f"Warning: {feature} not found in expression data")
            continue
            
        yy = expr_df.loc[feature]
        # color = colors[i]
        
        if std_df is None:
            plt.scatter(xx, yy, s=5, alpha=.5, label=feature)
        else:
            plt.plot(xx, yy, linewidth='.5', linestyle='-.', alpha=0.7, label=feature)
            if feature in std_df.index:
                plt.fill_between(xx, yy-std_df.loc[feature], yy+std_df.loc[feature], 
                                 alpha=.1)

        # Add fitted line using spline interpolation
        # Remove any NaN values and ensure x,y have same length
        mask = ~np.isnan(yy)
        if np.sum(mask) > 3:  # Need at least 4 points for fitting
            xx_clean = xx[mask]
            yy_clean = yy[mask]
            
            # Use LOWESS (locally weighted scatterplot smoothing) for more flexible fit
            smoothed = lowess(yy_clean, xx_clean, frac=0.2, it=3)
            
            # Interpolate back to original x positions
            interp_func = interp1d(smoothed[:, 0], smoothed[:, 1], 
                     kind='linear', bounds_error=False, fill_value='extrapolate')
            y_smooth = interp_func(xx)
            plt.plot(xx, y_smooth, linewidth=3, alpha=0.3)

    plt.xlabel(r"Cortex $\rightarrow$ Medulla axis (sliding window)", fontsize=12)
    plt.ylabel('Expression', fontsize=12)
    
    # Set title based on number of features
    if len(features) == 1:
        plt.title(features[0], fontsize=20)
    else:
        plt.title(title, fontsize=15)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add legend if multiple features
    if len(features) > 1:
        plt.legend(bbox_to_anchor=(0.8, 0.9), loc='center left')
        plt.tight_layout()
    
    plt.show()


disp_feature_dynamics(
    pexp_df.T, 
    features=['CD31_Pecam', 'CD8a', 'CD169'],
    title='Dynamics of CITE-seq Protein expressions',
    figsize=(8, 5))

# %%
disp_feature_dynamics(
    gexp_df.T,
    features=['Ccl25', 'Ccl22'],
    title='Dynamics of Stereo-seq Chemokines',
)

# %%
adata_protein.var_names


# %%
