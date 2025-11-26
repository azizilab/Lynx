# %%
import os
import gc
import sys

import numpy as np
import scanpy as sc
import pandas as pd
import squidpy as sq


from torch_geometric.loader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import rcParams

sns.set_context('paper')
rcParams.update({'font.family': 'Arial'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')
import IO, plot, utils, test_assoc, trajectory

from importlib import reload
%matplotlib inline
%load_ext autoreload
%autoreload 2


# %% Load data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5_proseg'

adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=False)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
cluster_key = 'subtype'


# %%
# load saved adata w/ all parameters
adata_xenium = sc.read_h5ad('../results/liver/LYNX_xenium_6_debug.h5ad')
adata_desi.obsm['X_z'] = np.load(
    '../results/liver/LYNX_desi_6_debug.npy'
).astype(np.float32)

# %%
# (i). Trajectory Inference
# Xenium gradient 
curve = trajectory.get_curve(adata_xenium, epg_lambda=0.01)
trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu_r', size=25, img=False,
    title=r'Inferred spatial Gradient $(t)$'+'\nLYNX'
)

plot.disp_trajectory(
    adata_xenium, 
    cmap='RdBu_r',
    title='Spatial Gradient \n LYNX (Xenium)'
)

# DESI gradient
curve = trajectory.get_curve(adata_desi, epg_lambda=0.01)
trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine ')

# sq.pl.spatial_scatter(
#     adata_desi, color='t', 
#     cmap='RdBu_r', size=1, img=False,
#     title=r'Spatial Gradient $(t)$'+'\nLYNX (DESI)'
# )

# plot.disp_trajectory(
#     adata_desi, 
#     cmap='RdBu_r',
#     title='Spatial Gradients\n LYNX (DESI)'
# )

# %%
if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
    sc.pp.normalize_total(adata_xenium)
    sc.pp.log1p(adata_xenium)

utils.get_zonation_features(    
    adata_xenium, adata_desi,
    n_zones=3, sample_id=sample_id,
    abundance_test=True,
    show=True
)
sq.pl.spatial_scatter(
    adata_xenium, color='zone',
    size=25, img=False,
)

# %%
adata_xenium.obs['zone']


# %%
# Cell-type dynamics along the gradient
from scipy.interpolate import UnivariateSpline

def disp_dynamics(
    df, feature, color='blue',
    std_df=None, ylabel='Expression', 
    dpi=100, figsize=(6, 3),
    milestone_assignments=None, milestone_cmap='Set3'
):
    r"""
    Plot curve dynamics with optional milestone colorbar.
    """    
    n_bins = df.shape[0]

    # Adjust figure layout if milestones are provided
    if milestone_assignments is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create main plot with space for milestone colorbar
        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=8)
        milestone_ax = plt.subplot2grid((10, 1), (9, 0), rowspan=1)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot trajectories
    x = np.arange(n_bins)
    y = df[feature]
    

    if std_df is None:
        # Spline regression
        spline = UnivariateSpline(x, y, s=len(x)*1e-3) 
        xx = np.linspace(x.min(), x.max(), 500)
        yy = spline(xx)
        
        # Compute residuals and standard deviation for uncertainty
        y_pred = spline(x)
        residuals = y - y_pred
        std_residual = np.std(residuals)
        
        # Plot with uncertainty bands
        ax.scatter(x, y, s=5, c=color, alpha=0.7)
        ax.plot(xx, yy, linewidth=1, c=color)
        ax.fill_between(xx, yy - std_residual, yy + std_residual, 
                color=color, alpha=0.3)
    else:
        ax.plot(x, y, linewidth=2, color=color, linestyle='-.')
        ax.fill_between(x, y-std_df[feature], y+std_df[feature], 
                        color=color, alpha=0.3)

    ax.grid(False)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(feature, fontsize=15)
    
    # Add milestone colorbar if provided
    if milestone_assignments is not None:
        # Create milestone colorbar
        unique_milestones = np.unique(milestone_assignments)
        n_milestones = len(unique_milestones)
        
        # Create colormap and normalization
        milestone_colors = plt.cm.get_cmap(milestone_cmap, n_milestones)
        milestone_to_idx = {milestone: i for i, milestone in enumerate(unique_milestones)}
        
        # Create array for colorbar
        milestone_indices = np.array([milestone_to_idx[m] for m in milestone_assignments])
        
        # Plot milestone assignments as image - align with scatter point positions
        milestone_ax.imshow(
            milestone_indices.reshape(1, -1), 
            aspect='auto', 
            cmap=milestone_colors,
            extent=[-0.5, n_bins-0.5, 0, 1]  # Changed from [0, n_bins] to [-0.5, n_bins-0.5]
        )
        
        # Configure milestone axis
        milestone_ax.set_xlim(-0.5, n_bins-0.5)  # Match the scatter point range
        milestone_ax.set_ylim(0, 1)
        milestone_ax.set_xticks([])
        milestone_ax.set_yticks([])
        
        # Add milestone labels
        milestone_positions = []
        milestone_labels = []
        for milestone in unique_milestones:
            milestone_mask = milestone_assignments == milestone
            if np.any(milestone_mask):
                # Find center position of this milestone
                indices = np.where(milestone_mask)[0]
                center_pos = (indices[0] + indices[-1]) / 2
                milestone_positions.append(center_pos)
                milestone_labels.append(milestone)
        
        # Add text labels for milestones
        for pos, label in zip(milestone_positions, milestone_labels):
            milestone_ax.text(pos, 0.5, label, ha='center', va='center', 
                            fontsize=8, fontweight='bold')
        
        # Remove x-axis label from main plot
        ax.set_xlabel(r'Pseudotime ($t$) (PV $\rightarrow$ CV bins)', fontsize=12)
        ax.set_xticks(np.arange(0, n_bins, n_bins//5))
        
        # Add colorbar title
        milestone_ax.set_title('', pad=5)
        
        # Match x-axis limits between main plot and colorbar
        ax.set_xlim(-0.5, n_bins-0.5)
        
    else:
        ax.set_xlabel(r'Pseudotime ($t$) (PV $\rightarrow$ CV bins)', fontsize=12)
    
    return fig, ax

# %%
n_bins = 50
celltype_dynamic_df = utils.get_celltype_dynamics(
    adata_xenium, adata_xenium.obs[cluster_key], n_bins=n_bins
)

cluster_labels = adata_xenium.obs[cluster_key].cat.categories.to_list()
for label in cluster_labels:
    disp_dynamics(
        celltype_dynamic_df,
        ylabel='Proportion', color='mediumblue',
        feature=label
    )

del label

# %%
# Compute binned zones
gamma = utils.get_binned_expr(
    pd.DataFrame(adata_xenium.obs['t'].sort_values()).T,
    n_bins=n_bins
).values.flatten()

zone_thresholds = [
    adata_xenium[adata_xenium.obs['zone'] == str(i)].obs['t'].max()
    for i in np.unique(adata_xenium.obs['zone'])
]

zone_assignments = []
for val in gamma:
    if val <= zone_thresholds[0]:
        zone_assignments.append('Zone 1')
    elif val <= zone_thresholds[1]:
        zone_assignments.append('Zone 2')
    else:
        zone_assignments.append('Zone 3')
zone_assignments = np.array(zone_assignments)
del val


# %%
fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300,
    ylabel='Proportion', color='mediumblue',
    feature='Endothelial', milestone_assignments=zone_assignments
)
fig.savefig('../figures/LYNX_Fig2_endothelial.pdf', bbox_inches='tight')


fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300,
    ylabel='Proportion', color='mediumblue',
    feature='LSECs', milestone_assignments=zone_assignments
)
fig.savefig('../figures/LYNX_Fig2_lsecs.pdf', bbox_inches='tight')

fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300,
    ylabel='Proportion', color='mediumblue',
    feature='Myeloid', milestone_assignments=zone_assignments
)
fig.savefig('../figures/LYNX_Fig2_myeloid.pdf', bbox_inches='tight')


# %%
cluster_labels



# %%
# (ii). Evaluate cell-cell interaction represented by cell-to-cell edge features
# (2.1) Retrieve overview summary of cell-cell interaction (apriori)
adata_xenium.obs[cluster_key] = adata_xenium.obs[cluster_key].astype('category')
cluster_labels=adata_xenium.obs[cluster_key].cat.categories
cci_df = plot.summarize_cell_interaction(
    adata_xenium, 
    cluster_key=cluster_key, 
    cluster_labels=cluster_labels,
    title='Overall Interaction',
    show_fig=True
)

# %%
# Visualize spatial cell-type distribution
sq.pl.spatial_scatter(
    adata_xenium, color='subtype',
    groups=['Progenitor+Cholangiocytes', 'PC-Hep', 'PP-Hep'],
    size=25, img=False,
)

# %%
# (2.2) Visualize spatial interaction within a local niche
# E.g. Visualize T-cell interaction patterns along the gradient
adata_subset = adata_xenium.copy()
adata_subset.obs.reset_index(inplace=True, drop=True)
cell_boundaries_filename = os.path.join(xenium_path, sample_id, 'cell_boundaries.parquet')
for idx in adata_subset.obs[adata_subset.obs[cluster_key] == 'SMCs'].sort_values('t').index[:5]:
    plot.disp_spatial_interaction(
        adata_xenium,
        target_idx=idx,
        cell_boundaries_parquet=cell_boundaries_filename,
        cluster_key=cluster_key,
    )
del idx, adata_subset


# %% 
cell_boundaries_filename = os.path.join(xenium_path, sample_id, 'cell_boundaries.parquet')
rand_indices= np.random.choice(adata_xenium.n_obs, size=5, replace=False)
for idx in rand_indices:
    subgraph_dict = plot.disp_spatial_interaction(
        adata_xenium,
        target_idx=idx,
        cell_boundaries_parquet=cell_boundaries_filename,
        cluster_key=cluster_key,
        return_subgraph=True
    )
    print(subgraph_dict['omega'])
    print(subgraph_dict['omega'].sum())
del idx


# %%
# (2.3) Statistical test vs. abundance
cluster_labels = adata_xenium.obs[cluster_key].cat.categories

cci_df = test_assoc.test_cci(adata_xenium, cci_df, cluster_labels, cluster_key=cluster_key)
plot.disp_heatmap(
    cci_df, 
    title='Significant cell-cell interaction (Overall)',
)

cci_dfs = []
for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_cci_df = plot.summarize_cell_interaction(
        adata_sub, 
        cluster_key=cluster_key,
        cluster_labels=cluster_labels,
        show_fig=False
    )
    
    zone_cci_df = test_assoc.test_cci(adata_sub, cluster_labels, zone_cci_df, cluster_key=cluster_key)
    cci_dfs.append(zone_cci_df)
    plot.disp_heatmap(
        zone_cci_df, 
        title=f'Significant cell-cell interaction (Zone {int(cluster_id)})',
    )

    plot.netVisual_circle(
        zone_cci_df, min_threshold=0.05, vertex_size_max=20, figsize=(15, 15),
        title=f'Summary of cell-cell interaction\n (Zone {int(cluster_id)})' 
    )

del zone_cci_df
gc.collect()

# %%
fig, ax = plot.netVisual_circle(
    cci_dfs[1], min_threshold=0.05, vertex_size_max=20, figsize=(15, 15),
    title=f'Summary of cell-cell interaction\n (Zone 2)'
)
fig.savefig('../figures/LYNX_fig2_cci_zone2.pdf', bbox_inches='tight')

# %%
