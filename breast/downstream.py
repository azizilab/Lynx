# %%
import os
import gc
import sys

import numpy as np
import scanpy as sc
import pandas as pd
import squidpy as sq


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
data_path = '../data/breast/'
outdir = '../figures/'
cluster_key = 'cell_type'
adata = sc.read_h5ad(os.path.join(data_path, 'LYNX_xenium.h5ad'))

# %% [markdown]
# (1). Trajectonry inference

# %%
principal_graph = trajectory.get_tree(
    adata,
    use_rep='X_z',
    n_nodes=50,
    plot_graph=True
)

# %%
# Visualize principal graph
import scFates as as scf
sc.set_figure_params(scanpy=True, dpi_save=300, fontsize=10)

rcParams.update({'font.size': 12})
scf.pl.graph(
    adata, tips=False, forks=False, basis='pca', 
    title='Principal graph',
    save='LYNX_Fig4_pc_tree.pdf'
)


# %%
# From the principal tree visualization, we assign the root node as 22
trajectory.compute_pseudotime(adata, principal_graph, source=22)



# %% [markdown]
# We can now further extract principal tree segments w.r.t the branching points:
# - (1). root to branching point (fork)
# - (2). fork to leaves

# %%
root_path = trajectory.sort_nodes(
    adata, root_node=22, term_node=6
)

dcis_path = trajectory.sort_nodes(
    adata, root_node=6, term_node=1
)[1:]   # Avoid repeating the branching node

invasive_path = trajectory.sort_nodes(
    adata, root_node=6, term_node=11
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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_hub.pdf'), bbox_inches='tight')

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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_pseudotime.pdf'), bbox_inches='tight')

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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_spatial_pseudotime.pdf'), bbox_inches='tight')
# %%
# 3D UMAP w/' principal tree
# sc.tl.umap(adata, n_components=3)
# scf.pl.trajectory_3d(adata, basis='umap', color='milestones')

# %%
# sc.set_figure_params(scanpy=True, fontsize=10)
fig, ax = plt.subplots(dpi=300)
sq.pl.spatial_scatter(
    adata, color='cell_type',
    img=False, size=20, ax=ax, return_ax=True,
    title=''
)
plt.show()
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_spatial.pdf'), bbox_inches='tight')

sc.set_figure_params(scanpy=True, fontsize=10)
fig, ax = plt.subplots(dpi=300)
sc.pl.pca(
    adata, 
    color='cell_type', 
    groups=['Stromal', 'DCIS', 'Invasive_Tumor'],
    na_in_legend=False,
    legend_loc="on data",
    legend_fontsize=6,
    ax=ax, title='', show=False)
ax.set_title('Stromal & tumor cell distributions\n'+'LYNX latent embedding', fontsize=12)
plt.show()
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_tumor.pdf'), bbox_inches='tight')

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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_immune.pdf'), bbox_inches='tight')


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
plt.show()
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_pc_stromal_state.pdf'), bbox_inches='tight').savefig(os.path.join(outdir, 'LYNX_Fig4_pc_stromal_state.pdf'), bbox_inches='tight')

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
    adata_stromal, groupby="stromal_state", # values_to_plot='logfoldchanges',
    dendrogram=False, n_genes=5, cmap='RdBu_r',
    ax=ax, show=False
)
plt.show()
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_stromal_marker_heatmap.pdf'), bbox_inches='tight')

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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_cd8_dynamics.pdf'), bbox_inches='tight')

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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_m2_dynamics.pdf'), bbox_inches='tight')

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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_gjb2_dynamics.pdf'), bbox_inches='tight')
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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_sfrp4_dynamics.pdf'), bbox_inches='tight')
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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_cxcl12_dynamics.pdf'), bbox_inches='tight')
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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_cxcr4_dynamics.pdf'), bbox_inches='tight')
test_dynamic_differences(
    dynamic_dfs=[dcis_gexp_df, invasive_gexp_df],
    labels=['DCIS_trajectory', 'Invasive_trajectory'],
    feature='CXCR4'
)

gc.collect()
gc.collect()

# %% [markdown]
# (2). cell-cell interaction analysis

# %%
cluster_labels = adata.obs['cell_type'].cat.categories

rcParams["axes.grid"] = False
cci_df = plot.summarize_cell_interaction(
    adata, 
    cluster_key=cluster_key, 
    title='Summary of cell-cell interaction\n(Overall)',
    show_plot=True
)
cci_df = test_assoc.test_cci(adata, cci_df, cluster_labels, cluster_key=cluster_key)
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
dcis_cci_df = test_assoc.test_cci(adata_dcis, dcis_cci_df, cluster_labels, cluster_key=cluster_key)
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
invasive_cci_df = test_assoc.test_cci(adata_invasive, invasive_cci_df, cluster_labels, cluster_key=cluster_key)
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
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_dcis_cci.pdf'), bbox_inches='tight')

fig, ax = plot.netVisual_circle(
    invasive_cci_df, figsize=(18, 18),
    title="Summary of cell-cell interaction\n (Invasive Tumor hub)", 
)
plt.show()
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_invasive_cci.pdf'), bbox_inches='tight')

# %%
del adata_dcis, adata_invasive
gc.collect()

# %%