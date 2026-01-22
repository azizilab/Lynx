# %%
import os
import gc
import sys

import numpy as np
import scanpy as sc
import pandas as pd
import squidpy as sq
import scFates as scf

import seaborn as sns
import matplotlib.pyplot as plt 
from IPython.display import display
from matplotlib import rcParams

sns.set_context('paper')
rcParams.update({'font.family': 'Arial'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')

import plot, utils, trajectory, test_assoc

%matplotlib inline
%load_ext autoreload
%autoreload 2

# %%
# Load saved anndata w/ LYNX results
data_path = '../results/breast/'
outdir = '../figures/'
cluster_key = 'cell_type'
adata = sc.read_h5ad(os.path.join(data_path, 'LYNX_xenium_cci.h5ad'))

# Unify cluster namings
adata.obs[cluster_key] = adata.obs[cluster_key].astype('str')
adata.obs.loc[adata.obs[cluster_key] == 'DCIS_1', cluster_key] = 'DCIS'
adata.obs.loc[adata.obs[cluster_key] == 'Prolif_Invasive_Tumor', cluster_key] = 'Invasive_Tumor'
adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')


# %% [markdown]
# (1). Trajectonry inference
principal_graph = trajectory.get_tree(
    adata,
    use_rep='X_z',
    n_nodes=int(0.01*adata.n_obs),
    ppt_lambda=1e3,
    plot_graph=True
)


# %%
# [Optional]: Select root & leave nodes to cleanup the graph
# trajectory.prune_tree(adata, tips_to_keep=[33, 34, 12])
scf.pl.graph(adata, basis='pca')


# %%
# Visualize principal graph
sc.set_figure_params(scanpy=True, dpi_save=300, fontsize=10)
rcParams.update({'font.size': 12})
scf.pl.graph(
    adata, tips=False, forks=False, basis='pca', 
    title='Principal graph',
    save='LYNX_Fig4_pc_tree.pdf'
)


# %% [markdown]
# From the principal tree visualization
# we assign the root, leaves & branching nodes
# We can now further extract principal tree segments w.r.t the branching points:
# - (1). root to branching point (fork)
# - (2). fork to leaves

# %%
root_node = 84
branch_node = 100
leave_nodes = [93, 28]
trajectory.compute_pseudotime(adata, principal_graph, source=root_node)

# %%
root_path = trajectory.sort_nodes(
    adata, root_node=root_node, term_node=branch_node
)

dcis_path = trajectory.sort_nodes(
    adata, root_node=branch_node, term_node=leave_nodes[0]
)[1:]   # Avoid repeating the branching node

invasive_path = trajectory.sort_nodes(
    adata, root_node=branch_node, term_node=leave_nodes[1]
)[1:]   # Avoid repeating the branching node

segments = []
principal_assignments = adata.obsm['X_R'].argmax(1)
for i, assign in enumerate(principal_assignments):
    if assign in root_path:
        segments.append('root')
    elif assign in dcis_path:
        segments.append('dcis')
    else:
        segments.append('invasive')

adata.obs['zone'] = segments
adata.obs['seg'] = pd.Categorical(segments).codes

fig, ax = plt.subplots(dpi=300)
sc.pl.pca(
    adata, color=['zone'],
    ax=ax, title='', show=False)
ax.set_title('Principal graph hub assignment', fontsize=12)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_hub.pdf'), bbox_inches='tight')

# %% [markdown]
# Spatial visualizations

# %%
# Visualize cell-type distributions
fig, ax = plt.subplots(dpi=300)
sc.pl.pca(adata, color='t', ax=ax, title='', cmap='RdBu_r', show=False)
ax.set_title('Inferred spatial gradient\nLYNX latent embedding', fontsize=12)
cb = plt.gcf().axes[-1]
cb.set_ylabel(r'Pseudotime $(t)$', fontsize=8)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_pseudotime.pdf'), bbox_inches='tight')

fig, ax = plt.subplots(dpi=300)
sq.pl.spatial_scatter(
    adata, 
    color='t', cmap='RdBu_r',
    size=20, img=False, edgecolor='none', 
    ax=ax, return_ax=True, title='',
)
cb = plt.gcf().axes[-1]
cb.set_ylabel(r'Pseudotime $(t)$', fontsize=8)
ax.set_title('Inferred spatial gradient', fontsize=12)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_spatial_pseudotime.pdf'), bbox_inches='tight')

# %%
# 3D UMAP w/' principal tree
sc.tl.umap(adata, n_components=3)
scf.pl.trajectory_3d(adata, basis='umap', color='milestones')

# %%
# sc.set_figure_params(scanpy=True, fontsize=10)
fig, ax = plt.subplots(dpi=300)
sq.pl.spatial_scatter(
    adata, color='cell_type',
    img=False, size=20, ax=ax, return_ax=True,
    title=''
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_spatial.pdf'), bbox_inches='tight')

sc.set_figure_params(scanpy=True, fontsize=10)
fig, ax = plt.subplots(dpi=300)
sc.pl.pca(
    adata, 
    color='cell_type', 
    groups=['Stromal', 'DCIS', 'Invasive_Tumor'],
    na_in_legend=False,
    ax=ax, title='', show=False)
ax.set_title('Stromal & tumor cell distributions\n'+'LYNX latent embedding', fontsize=12)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_tumor.pdf'), bbox_inches='tight')

fig, ax = plt.subplots(dpi=300)
sc.pl.pca(
    adata, 
    color='cell_type', 
    groups=[
        'B_Cells',
        'CD4+_T_Cells',
        'CD8+_T_Cells',
        'IRF7+_DCs',
        'LAMP3+_DCs',
        'Macrophages_1',
        'Macrophages_2',
        'Mast_Cells',           
    ],
    na_in_legend=False,
    ax=ax, title='', show=False)
ax.set_title('Immune cell distributions\n'+r'LYNX latent embedding', fontsize=12)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_immune.pdf'), bbox_inches='tight')


# %% [markdown]
# **Case Study**: stromal cell subsetting w.r.t subtree segments (zones)

# %%
stromal_states = np.array(['NA']*len(adata)).astype('>U20')
stromal_states[
    np.logical_and(
        adata.obs[cluster_key] == 'Stromal',
        adata.obs['zone'] == 'invasive'
    )
] = 'Invasive_adjacent'

stromal_states[
    np.logical_and(
        adata.obs[cluster_key] == 'Stromal',
        adata.obs['zone'] == 'dcis'
    )
] = 'DCIS_adjacent'

stromal_states[
    np.logical_and(
        adata.obs[cluster_key] == 'Stromal',
        adata.obs['zone'] == 'root'
    )
] = 'Root_adjacent'

adata.obs['stromal_state'] = stromal_states
fig, ax = plt.subplots(dpi=300)
sc.pl.pca(
    adata, color='stromal_state',
    groups=['DCIS_adjacent', 'Invasive_adjacent', 'Root_adjacent'],
    na_in_legend=False, ax=ax, title='', show=False)
ax.set_title('Stromal state assignment', fontsize=12)
# plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_stromal_state.pdf'), bbox_inches='tight')

# %%
# Marker genes for stromal states

# Remove duplicate columns
if any(adata.obs.columns.duplicated()):
    print(f"Removing duplicate columns: {adata.obs.columns[adata.obs.columns.duplicated()].tolist()}")
    adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]

adata.obs_names = adata.obs.index.astype(str)
adata_stromal = adata[adata.obs['cell_type'] == 'Stromal'].copy()
sc.pp.normalize_total(adata_stromal, target_sum=1e4)
sc.pp.log1p(adata_stromal)
sc.pp.scale(adata_stromal)
sc.tl.rank_genes_groups(adata_stromal, groupby="stromal_state", method="wilcoxon")
sc.pl.rank_genes_groups(adata_stromal, n_genes=10, sharey=False)

dcis_stromal_markers = sc.get.rank_genes_groups_df(adata_stromal, group='DCIS_adjacent').head(10).names.to_list()
invasive_stromal_markers = sc.get.rank_genes_groups_df(adata_stromal, group='Invasive_adjacent').head(10).names.to_list()

sc.set_figure_params(scanpy=True, dpi_save=300, fontsize=10)
fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)
sc.pl.rank_genes_groups_matrixplot(
    adata_stromal, groupby="stromal_state",
    dendrogram=False, n_genes=5, cmap='RdBu_r',
    ax=ax, show=False
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_stromal_marker_heatmap.pdf'), bbox_inches='tight')

# %%
# PC visualization of stromal markers
dcis_stromal_markers = sc.get.rank_genes_groups_df(adata_stromal, group='DCIS_adjacent').head(10).names.to_list()
invasive_stromal_markers = sc.get.rank_genes_groups_df(adata_stromal, group='Invasive_adjacent').head(10).names.to_list()

adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)

sc.pl.pca(
    adata_norm, 
    color=dcis_stromal_markers+invasive_stromal_markers,
    s=10, ncols=5, cmap='magma'
)
del adata_stromal
gc.collect()

# %% [markdown]
# Compute cell-type & feature dynamics along the branching trajectories

# %%
from scipy.interpolate import UnivariateSpline
from typing import Iterable
from scipy.stats import ttest_rel

def disp_tree_dynamics(
    dynamic_dfs, labels, feature, colors,
    std_dfs=None, ylabel='Expression', 
    spline_factor=1e-3, dpi=100, figsize=(6, 3),
    zone_assignments=None, zone_cmap='Set3'
):
    r"""
    Plot tree dynamics with optional zone colorbar.
    """
    assert len(dynamic_dfs) == len(labels)
    if isinstance(colors, Iterable) and not isinstance(colors, str):
        assert len(colors) == len(dynamic_dfs)
    
    n_bins = dynamic_dfs[0].shape[0]

    # Adjust figure layout if zone are provided
    if zone_assignments is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create main plot with space for zone colorbar
        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=8)
        zone_ax = plt.subplot2grid((10, 1), (9, 0), rowspan=1)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot trajectories
    for i, df in enumerate(dynamic_dfs):
        x = np.arange(n_bins)
        y = df[feature]
        
        color = colors[i] if isinstance(colors, (list, tuple)) else colors
    
        if std_dfs is None:
            # Spline regression
            spline = UnivariateSpline(x, y, s=len(x)*spline_factor) 
            xx = np.linspace(x.min(), x.max(), 500)
            yy = spline(xx)
            
            # Compute residuals and standard deviation for uncertainty
            y_pred = spline(x)
            residuals = y - y_pred
            std_residual = np.std(residuals)
            
            # Plot with uncertainty bands
            ax.scatter(x, y, s=5, c=color, label=labels[i])
            ax.plot(xx, yy, linewidth=1, c=color)
            ax.fill_between(xx, yy - std_residual, yy + std_residual, 
                    color=color, alpha=0.5)
        else:
            ax.plot(x, y, linewidth=2, color=color, linestyle='-.', label=labels[i])
            ax.fill_between(x, y-std_dfs[i][feature], y+std_dfs[i][feature], 
                          color=color, alpha=0.5)

    ax.grid(False)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    ax.set_title(feature, fontsize=15)
    
    # Add zone colorbar if provided
    if zone_assignments is not None:
        # Create zone colorbar
        unique_zone = np.unique(zone_assignments)
        n_zone = len(unique_zone)
        
        # Create colormap and normalization
        zone_colors = plt.cm.get_cmap(zone_cmap, n_zone)
        zone_to_idx = {zone: i for i, zone in enumerate(unique_zone)}
        
        # Create array for colorbar
        zone_indices = np.array([zone_to_idx[m] for m in zone_assignments])
        
        # Plot zone assignments as image - align with scatter point positions
        zone_ax.imshow(
            zone_indices.reshape(1, -1), 
            aspect='auto', 
            cmap=zone_colors,
            extent=[-0.5, n_bins-0.5, 0, 1]  # Changed from [0, n_bins] to [-0.5, n_bins-0.5]
        )
        
        # Configure zone axis
        zone_ax.set_xlim(-0.5, n_bins-0.5)  # Match the scatter point range
        zone_ax.set_ylim(0, 1)
        zone_ax.set_xticks([])
        zone_ax.set_yticks([])
        
        # Add zone labels
        zone_positions = []
        zone_labels = []
        for zone in unique_zone:
            zone_mask = zone_assignments == zone
            if np.any(zone_mask):
                # Find center position of this zone
                indices = np.where(zone_mask)[0]
                center_pos = (indices[0] + indices[-1]) / 2
                zone_positions.append(center_pos)
                zone_labels.append(zone)
        
        # Add text labels for zone
        for pos, label in zip(zone_positions, zone_labels):
            zone_ax.text(pos, 0.5, label, ha='center', va='center', 
                            fontsize=8, fontweight='bold')
        
        # Remove x-axis label from main plot
        ax.set_xlabel(r'Pseudotime (root $\rightarrow$ tumor bins)')
        ax.set_xticks(np.arange(0, n_bins, n_bins//5))
        
        # Add colorbar title
        zone_ax.set_title('', pad=5)
        
        # Match x-axis limits between main plot and colorbar
        ax.set_xlim(-0.5, n_bins-0.5)
        
    else:
        ax.set_xlabel(r'Pseudotime (root $\rightarrow$ tumor bins)', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def test_dynamic_differences(
    dynamic_dfs, labels, feature, alpha=0.05
):
    r"""
    Perform paired t-test between two dynamics for a given feature.
    """
    assert len(dynamic_dfs) == 2, "Exactly two dynamics required for paired t-test."
    
    data1 = dynamic_dfs[0][feature]
    data2 = dynamic_dfs[1][feature]
    
    t_stat_greater, p_value_greater = ttest_rel(data1, data2, alternative='greater')
    t_stat_less, p_value_less = ttest_rel(data1, data2, alternative='less')

    if p_value_greater < alpha:
        print(f"{feature}: {labels[0]} > {labels[1]} (p={p_value_greater:.4e})")
    elif p_value_less < alpha:
        print(f"{feature}: {labels[0]} < {labels[1]} (p={p_value_less:.4e})")
    else:
        print(f"""No significant directional differences""")
    

# %%
# Compute smoothed trajectory seg assignments
n_bins = 50
smoothed_t = utils.get_binned_expr(
    pd.DataFrame(adata.obs['t'].sort_values()).T,
    n_bins=n_bins
).values.flatten()
t_threshold = adata[adata.obs['zone'] == 'root'].obs['t'].max()
zone_assignments = np.where(smoothed_t < t_threshold, 'root', 'tumor')
del smoothed_t, t_threshold

# Cell-type dynamics towards Invasive path
adata_invasive = adata_norm.copy()
adata_invasive.obs_names = adata_invasive.obs_names.astype(str)
adata_invasive = adata_invasive[adata_invasive.obs['zone'].isin(['root', 'invasive'])].copy()
invasive_dynamic_df = utils.get_celltype_dynamics(adata_invasive, adata_invasive.obs['cell_type'], n_bins=n_bins)

# Cell-type dynamics towards DCIS
adata_dcis = adata_norm.copy()
adata_dcis.obs_names = adata_dcis.obs_names.astype(str)
adata_dcis = adata_dcis[adata_dcis.obs['zone'].isin(['root', 'dcis'])].copy()
dcis_dynamic_df = utils.get_celltype_dynamics(adata_dcis, adata_dcis.obs['cell_type'], n_bins=n_bins)

cluster_labels = adata.obs[cluster_key].cat.categories.to_list()
for label in cluster_labels:
    if label in dcis_dynamic_df.columns and label in invasive_dynamic_df.columns:
        disp_tree_dynamics(
            dynamic_dfs=[dcis_dynamic_df, invasive_dynamic_df],
            labels=['DCIS_trajectory', 'Invasive_trajectory'],
            ylabel='Proportion', colors=['mediumblue', 'coral'],
            feature=label
        )
        test_dynamic_differences(
            dynamic_dfs=[dcis_dynamic_df, invasive_dynamic_df],
            labels=['DCIS_trajectory', 'Invasive_trajectory'],
            feature=label
        )

del label
gc.collect()

# %%
def plot_stacked_dynamics(df, title=None, figsize=(8, 4)):
    """
    Plots a stacked bar chart of cell-type proportions across bins.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Plot stacked bars
    df.plot(
        kind='bar', 
        stacked=True, 
        width=1.0,       # Remove gaps between bars for a continuous look
        edgecolor='black', # Add edge color
        linewidth=0.2,     # Thin edge line
        ax=ax,
        legend=False     # Hide legend initially to keep it clean
    )

    # Minimal styling
    ax.set_xlabel(r'Pseudotime bins (root $\rightarrow$ tumor)')
    ax.set_ylabel('Proportion')
    ax.set_xticks([])    # Hide x-ticks for cleaner look
    ax.set_xlim(-0.5, len(df)-0.5)
    ax.set_ylim(0, 1)
    ax.grid(False)       # Ensure grid is off
    
    # Add legend outside
    ax.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0,
        frameon=False,
        fontsize='small'
    )
    
    if title:
        ax.set_title(title)
        
    plt.tight_layout()
    return fig, ax


# %%
fig, ax = plot_stacked_dynamics(
    dcis_dynamic_df, 
    title='Cell-type Dynamics (DCIS trajectory)'
)
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_dcis_stacked_dynamics.pdf'), bbox_inches='tight')

fig, ax = plot_stacked_dynamics(
    invasive_dynamic_df, 
    title='Cell-type Dynamics (Invasive trajectory)'
)
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_invasive_stacked_dynamics.pdf'), bbox_inches='tight')



# %% [markdown]
# Gene expression dynamics along DCIS vs Invasive trajectories

# %%
# Example visualizations
fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_dynamic_df, invasive_dynamic_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    ylabel='Proportion', colors=['mediumblue', 'coral'],
    feature='CD8+_T_Cells', zone_assignments=zone_assignments, 
    dpi=300
)
plt.show()
test_dynamic_differences(
    dynamic_dfs=[dcis_dynamic_df, invasive_dynamic_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='CD8+_T_Cells'
)
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_cd8_dynamics.pdf'), bbox_inches='tight')

fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_dynamic_df, invasive_dynamic_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    ylabel='Proportion', colors=['mediumblue', 'coral'],
    feature='Macrophages_2', zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
test_dynamic_differences(
    dynamic_dfs=[dcis_dynamic_df, invasive_dynamic_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='Macrophages_2'
)
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_m2_dynamics.pdf'), bbox_inches='tight')

gc.collect()

# %%
# Gene expression dynamics along DCIS vs Invasive trajectories
n_bins = 50
indices = np.argsort(adata_dcis.obs['t'].values)

dcis_gexp_df, dcis_gexp_std_df = utils.get_binned_expr(
    adata_dcis.to_df().iloc[indices].T,
    n_bins=n_bins,
    std=True
)

indices = np.argsort(adata_invasive.obs['t'].values)
invasive_gexp_df, invasive_gexp_std_df = utils.get_binned_expr(
    adata_invasive.to_df().iloc[indices].T,
    n_bins=n_bins,
    std=True
)

# Visualization
fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    # std_dfs=[dcis_gexp_std_df, invasive_gexp_std_df],
    feature='GJB2', spline_factor=.1,
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    colors=['mediumblue', 'coral'],
    zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_gjb2_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='GJB2'
)

fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    # std_dfs=[dcis_gexp_std_df, invasive_gexp_std_df],
    feature='SFRP4',spline_factor=.1,
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    colors=['mediumblue', 'coral'],
    zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_sfrp4_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='SFRP4'
)

fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    # std_dfs=[dcis_gexp_std_df, invasive_gexp_std_df],
    feature='CXCL12', spline_factor=.1,
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    colors=['mediumblue', 'coral'],
    zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_cxcl12_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='CXCL12'
)

fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    # std_dfs=[dcis_gexp_std_df, invasive_gexp_std_df],
    feature='CXCR4', spline_factor=.1,
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    colors=['mediumblue', 'coral'],
    zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_cxcr4_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='CXCR4'
)

gc.collect()

# %%
# Signature comparisons
myCAF_markers = [
    "APOD",
    "DCN",
    "PTGDS",
    "CFD",
    "LUM",
    "C1S",
    "CXCL12",
    "C3",
    "SFRP2",
    "CXCL14",
    "CCDC80",
    "MFAP4",
    "FBLN1",
    "GSN",
    "CTSK",
    "SERPINF1",
    "RARRES2",
    "SFRP4",
    "C1R",
    "IGF2",
    "CYR61",
    "CTGF",
    "SERPING1",
    "IGFBP6",
    "MMP2",
    "IGFBP4",
    "COL6A2",
    "MEG3",
    "IGF1",
    "SRPX",
    "COL3A1",
    "AEBP1"
]

iCAF_markers = [
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "LUM",
    "SFRP2",
    "POSTN",
    "MMP11",
    "CTHRC1",
    "FN1",
    "SPARC",
    "DCN",
    "COL6A3",
    "BGN",
    "COL6A2",
    "COL6A1",
    "CTGF",
    "AEBP1",
    "COL5A2",
    "VCAN",
    "CTSK",
    "RARRES2",
    "TIMP1",
    "CCDC80",
    "MMP2",
    "SFRP4",
    "CXCL14",
    "ASPN",
    "THY1",
    "MFAP2",
    "C1S",
    "SERPINF1"
]
CAF_markers = list(set(myCAF_markers + iCAF_markers))
CAF_markers = [gene for gene in CAF_markers if gene in adata.var_names]
del myCAF_markers, iCAF_markers


PVL_diff_markers = [
    "ACTA2",
    "TAGLN",
    "MYL9",
    "TPM2",
    "NDUFA4L2",
    "SOD3",
    "ADIRF",
    "MYH11",
    "RGS5",
    "RERGL",
    "IGFBP7",
    "CALD1",
    "SPARCL1",
    "MT1M",
    "C11orf96",
    "PPP1R14A",
    "MFGE8",
    "PLAC9",
    "DSTN",
    "PTP4A3",
    "MCAM",
    "SORBS2",
    "COL18A1",
    "TINAGL1",
    "CAV1",
    "TPM1",
    "MT1E",
    "PLN",
    "CSRP2",
    "PRKCDBP",
    "MYLK"
]

PVL_immature_markers = [
    "CCL19",
    "RGS5",
    "IGFBP7",
    "NDUFA4L2",
    "CCL2",
    "CCL21",
    "COL18A1",
    "CALD1",
    "LHFP",
    "THY1",
    "CPE",
    "MYL9",
    "SPARC",
    "COL4A1",
    "TAGLN",
    "STEAP4",
    "ACTA2",
    "COL4A2",
    "TIMP1",
    "IGFBP5",
    "NOTCH3",
    "PDGFRB",
    "BGN",
    "SERPING1",
    "TIMP3",
    "HIGD1B",
    "COL6A2",
    "COX4I2",
    "TPM2",
    "MT1M",
    "GGT5"    
]
PVL_markers = list(set(PVL_diff_markers + PVL_immature_markers))
PVL_markers = [gene for gene in PVL_markers if gene in adata.var_names]
del PVL_diff_markers, PVL_immature_markers


EMT_markers = [
    "ABI3BP","ACTA2","ADAM12","ANPEP","APLP1","AREG","BASP1","BDNF","BGN","BMP1",
    "CADM1","CALD1","CALU","CAP2","CAPG","CCN1","CCN2","CD44","CD59","CDH11","CDH2",
    "CDH6","COL11A1","COL12A1","COL16A1","COL1A1","COL1A2","COL3A1","COL4A1","COL4A2",
    "COL5A1","COL5A2","COL5A3","COL6A2","COL6A3","COL7A1","COL8A2","COLGALT1","COMP",
    "COPA","CRLF1","CTHRC1","CXCL1","CXCL12","CXCL6","CXCL8","DAB2","DCN","DKK1",
    "DPYSL3","DST","ECM1","ECM2","EDIL3","EFEMP2","ELN","EMP3","ENO2","FAP","FAS",
    "FBLN1","FBLN2","FBLN5","FBN1","FBN2","FERMT2","FGF2","FLNA","FMOD","FN1","FOXC2",
    "FSTL1","FSTL3","FUCA1","FZD8","GADD45A","GADD45B","GAS1","GEM","GJA1","GLIPR1",
    "GPC1","GPX7","GREM1","HTRA1","ID2","IGFBP2","IGFBP3","IGFBP4","IL15","IL32",
    "IL6","INHBA","ITGA2","ITGA5","ITGAV","ITGB1","ITGB3","ITGB5","JUN","LAMA1",
    "LAMA2","LAMA3","LAMC1","LAMC2","LGALS1","LOX","LOXL1","LOXL2","LRP1","LRRC15",
    "LUM","MAGEE1","MATN2","MATN3","MCM7","MEST","MFAP5","MGP","MMP1","MMP14","MMP2",
    "MMP3","MSX1","MXRA5","MYL9","MYLK","NID2","NNMT","NOTCH2","NT5E","NTM","OXTR",
    "P3H1","PCOLCE","PCOLCE2","PDGFRB","PDLIM4","PFN2","PLAUR","PLOD1","PLOD2",
    "PLOD3","PMEPA1","PMP22","POSTN","PPIB","PRRX1","PRSS2","PTHLH","PTX3","PVR",
    "QSOX1","RGS4","RHOB","SAT1","SCG2","SDC1","SDC4","SERPINE1","SERPINE2","SERPINH1",
    "SFRP1","SFRP4","SGCB","SGCD","SGCG","SLC6A8","SLIT2","SLIT3","SNAI2","SNTB1",
    "SPARC","SPOCK1","SPP1","TAGLN","TFPI2","TGFB1","TGFBI","TGFBR3","TGM2","THBS1",
    "THBS2","THY1","TIMP1","TIMP3","TNC","TNFAIP3","TNFRSF11B","TNFRSF12A","TPM1",
    "TPM2","TPM4","VCAM1","VCAN","VEGFA","VEGFC","VIM","WIPF1","WNT5A"
]
EMT_markers = [gene for gene in EMT_markers if gene in adata.var_names]


hypoxia_markers = [
    "ACKR3","ADM","ADORA2B","AK4","AKAP12","ALDOA","ALDOB","ALDOC","AMPD3",
    "ANGPTL4","ANKZF1","ANXA2","ATF3","ATP7A","B3GALT6","B4GALNT2","BCAN",
    "BCL2","BGN","BHLHE40","BNIP3L","BRS3","BTG1","CA12","CASP6","CAV1","CAVIN1",
    "CAVIN3","CCN1","CCN2","CCN5","CCNG2","CDKN1A","CDKN1B","CDKN1C","CHST2","CHST3",
    "CITED2","COL5A1","CP","CSRP2","CXCR4","DCN","DDIT3","DDIT4","DPYSL4","DTNA",
    "DUSP1","EDN2","EFNA1","EFNA3","EGFR","ENO1","ENO2","ENO3","ERO1A","ERRFI1",
    "ETS1","EXT1","F3","FAM162A","FBP1","FOS","FOSL2","FOXO3","GAA","GALK1",
    "GAPDH","GAPDHS","GBE1","GCK","GCNT2","GLRX","GPC1","GPC3","GPC4","GPI",
    "GRHPR","GYS1","HAS1","HDLBP","HEXA","HK1","HK2","HMOX1","HOXB9","HS3ST1",
    "HSPA5","IDS","IER3","IGFBP1","IGFBP3","IL6","ILVBL","INHA","IRS2","ISG20",
    "JMJD6","JUN","KDELR3","KDM3A","KIF5A","KLF6","KLF7","KLHL24","LALBA","LARGE1",
    "LDHA","LDHC","LOX","LXN","MAFF","MAP3K1","MIF","MT1E","MT2A","MXI1","MYH9",
    "NAGK","NCAN","NDRG1","NDST1","NDST2","NEDD4L","NFIL3","NOCT","NR3C1","P4HA1",
    "P4HA2","PAM","PCK1","PDGFB","PDK1","PDK3","PFKFB3","PFKL","PFKP","PGAM2","PGF",
    "PGK1","PGM1","PGM2","PHKG1","PIM1","PKLR","PKP1","PLAC8","PLAUR","PLIN2","PNRC1",
    "PPARGC1A","PPFIA4","PPP1R15A","PPP1R3C","PRDX5","PRKCA","PYGM","RBPJ","RORA",
    "RRAGD","S100A4","SAP30","SCARB1","SDC2","SDC3","SDC4","SELENBP1","SERPINE1",
    "SIAH2","SLC25A1","SLC2A1","SLC2A3","SLC2A5","SLC37A4","SLC6A6","SRPX","STBD1",
    "STC1","STC2","SULT2B1","TES","TGFB3","TGFBI","TGM2","TIPARP","TKTL1","TMEM45A",
    "TNFAIP3","TPBG","TPD52","TPI1","TPST2","UGP2","VEGFA","VHL","VLDLR","WSB1",
    "XPNPEP1","ZFP36","ZNF292"
]
hypoxia_markers = [gene for gene in hypoxia_markers if gene in adata.var_names]


# Append PVL & CAF markers
PVL_signature = adata_norm.to_df()[PVL_markers].mean(axis=1).values
CAF_signature = adata_norm.to_df()[CAF_markers].mean(axis=1).values
EMT_signature = adata_norm.to_df()[EMT_markers].mean(axis=1).values
hypoxia_signature = adata_norm.to_df()[hypoxia_markers].mean(axis=1).values

signature_df = pd.DataFrame({
        'PVL_signature': PVL_signature,
        'CAF_signature': CAF_signature,
        'EMT_signature': EMT_signature,
        'hypoxia_signature': hypoxia_signature
    }, 
    index=adata_norm.obs_names
)


adata.obs['PVL_signature'] = PVL_signature
adata.obs['CAF_signature'] = CAF_signature
adata.obs['EMT_signature'] = EMT_signature
adata.obs['hypoxia_signature'] = hypoxia_signature


sc.pl.pca(adata, color=['PVL_signature', 'CAF_signature', 'EMT_signature', 'hypoxia_signature'], cmap='seismic')

del adata.obs['PVL_signature']
del adata.obs['CAF_signature']
del adata.obs['EMT_signature']
del adata.obs['hypoxia_signature']
gc.collect()

# %%
n_bins = 50
indices = np.argsort(adata_dcis.obs['t'].values)
dcis_sig_df, dcis_sig_std_df = utils.get_binned_expr(
    signature_df.loc[adata_dcis.obs_names].iloc[indices].T,
    n_bins=n_bins,
    std=True
)

indices = np.argsort(adata_invasive.obs['t'].values)
invasive_sig_df, invasive_sig_std_df = utils.get_binned_expr(
    signature_df.loc[adata_invasive.obs_names].iloc[indices].T,
    n_bins=n_bins,
    std=True
)

fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_sig_df, invasive_sig_df],
    # std_dfs=[dcis_sig_std_df, invasive_sig_std_df],
    feature='CAF_signature', spline_factor=1, 
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    colors=['mediumblue', 'coral'],
    zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_CAF_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_sig_df, invasive_sig_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='CAF_signature'
)


fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_sig_df, invasive_sig_df],
    # std_dfs=[dcis_sig_std_df, invasive_sig_std_df],
    feature='PVL_signature', spline_factor=1, 
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    colors=['mediumblue', 'coral'],
    zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_PVL_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_sig_df, invasive_sig_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='PVL_signature'
)

fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_sig_df, invasive_sig_df],
    # std_dfs=[dcis_sig_std_df, invasive_sig_std_df],
    feature='EMT_signature', spline_factor=1, 
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    colors=['mediumblue', 'coral'],
    zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_EMT_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_sig_df, invasive_sig_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='EMT_signature'
)


fig, ax = disp_tree_dynamics(
    dynamic_dfs=[dcis_sig_df, invasive_sig_df],
    # std_dfs=[dcis_sig_std_df, invasive_sig_std_df],
    feature='hypoxia_signature', spline_factor=1, 
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    colors=['mediumblue', 'coral'],
    zone_assignments=zone_assignments,
    dpi=300
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_hypoxia_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_sig_df, invasive_sig_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='hypoxia_signature'
)


gc.collect()



# %% [markdown]
# (2). cell-cell interaction analysis
# TODO: slightly modify architecture first

# %%
cluster_labels = adata.obs['cell_type'].cat.categories

rcParams["axes.grid"] = False
cci_df = plot.summarize_cell_interaction(
    adata, 
    cluster_key=cluster_key, 
    title='Summary of cell-cell interaction\n(Overall)',
    show_plot=True
)
cci_df, _ = test_assoc.test_cci(adata, cci_df, cluster_labels, cluster_key=cluster_key)
plot.disp_heatmap(
    cci_df,
    title='Summary of cell-cell interaction\n(Overall)'
)


rcParams["axes.grid"] = False
adata_dcis = adata[adata.obs['zone'] == 'dcis'].copy()
dcis_cci_df = plot.summarize_cell_interaction(
    adata_dcis,
    cluster_key=cluster_key, 
    cluster_labels=cluster_labels,
    title='Summary of cell-cell interaction\n(DCIS hub)',
    show_plot=True
)
dcis_cci_df, _ = test_assoc.test_cci(adata_dcis, dcis_cci_df, cluster_labels, cluster_key=cluster_key)
plot.disp_heatmap(
    dcis_cci_df,
    title='Summary of cell-cell interaction\n(DCIS hub)'
)

rcParams["axes.grid"] = False
adata_invasive = adata[adata.obs['zone'] == 'invasive'].copy()
invasive_cci_df = plot.summarize_cell_interaction(
    adata_invasive,
    cluster_key=cluster_key, 
    cluster_labels=cluster_labels,
    title='Summary of cell-cell interaction\n(Invasive Tumor hub)',
    show_plot=True
)
invasive_cci_df, _ = test_assoc.test_cci(adata_invasive, invasive_cci_df, cluster_labels, cluster_key=cluster_key)
plot.disp_heatmap(
    invasive_cci_df,
    title='Summary of cell-cell interaction\n(Invasive Tumor hub)'
)

# %%
fig, ax = plot.netVisual_circle(
    dcis_cci_df, figsize=(18, 18),
    title="Summary of cell-cell interaction\n (DCIS hub)", 
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_dcis_cci.pdf'), bbox_inches='tight')

fig, ax = plot.netVisual_circle(
    invasive_cci_df, figsize=(18, 18),
    title="Summary of cell-cell interaction\n (Invasive Tumor hub)", 
)
plt.show()
# fig.savefig(os.path.join(outdir, 'LYNX_Fig4_invasive_cci.pdf'), bbox_inches='tight')

# %%
del adata_dcis, adata_invasive
gc.collect()

# %%