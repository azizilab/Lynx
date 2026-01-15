# %%
import os
import gc
import sys

import numpy as np
import scanpy as sc
import spatialdata as sd
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
cluster_key = 'cell_type'

# Update w/ DESI annotations
metabolite_annots_df = pd.read_csv('../data/DESI_annotation.csv', header=0)
metabolite_dict = {
    k: v for k, v in zip(metabolite_annots_df.iloc[:, 0], metabolite_annots_df.iloc[:, 1])
    if not pd.isna(v)
}
del metabolite_annots_df
adata_desi.var_names = [
    metabolite_dict[c] if c in metabolite_dict else c
    for c in adata_desi.var_names
]
adata_desi.var_names_make_unique()

# %%
# load saved adata w/ all parameters
adata_xenium = sc.read_h5ad('../results/liver/LYNX_xenium_6_cluster.h5ad')
adata_desi.obsm['X_z'] = np.load(
    '../results/liver/LYNX_desi_6_cluster.npy'
).astype(np.float32)

# %%
# (i). Trajectory Inference
# Xenium gradient 
# curve = trajectory.get_curve(adata_xenium, epg_lambda=0.01, trim_radius_ratio=0.5)
# trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

# sq.pl.spatial_scatter(
#     adata_xenium, color='t', 
#     cmap='RdBu_r', size=25, img=False,
#     title='Inferred spatial Gradient\nLYNX'
# )

# plot.disp_trajectory(
#     adata_xenium, 
#     cmap='RdBu_r',
#     title='Inferred Spatial Gradient\nLYNX embedding'
# )

# DESI gradient
curve = trajectory.get_curve(adata_desi, epg_lambda=0.01, trim_radius_ratio=0.5)
trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine')

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
# Normalize Xenium data for DEG calculation
if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
    sc.pp.normalize_total(adata_xenium, target_sum=1e4)
    sc.pp.log1p(adata_xenium)

utils.get_zonation_features(    
    adata_xenium, 
    adata_desi,
    n_zones=5, sample_id=sample_id,
    abundance_test=True,
    show=True
)
sq.pl.spatial_scatter(
    adata_xenium, color='zone',
    size=25, img=False,
)

# %%
# Helper functions 
from scipy.interpolate import UnivariateSpline

def smooth_zone_assignments(adata, n_bins):
    r"""Smooth discrete zone assignments"""
    assert 't' in adata.obs.keys() and 'zone' in adata.obs.keys(), \
        "Please run trajectory & zonation inference first"

    df = pd.DataFrame(adata.obs['t'].sort_values()).T
    smoothed_t = utils.get_binned_expr(df,n_bins=n_bins).values.flatten()
    zone_cutoffs = [
        adata[adata.obs['zone'] == i].obs['t'].max()
        for i in np.unique(adata.obs['zone'])
    ]
    smoothed_zones = np.digitize(smoothed_t, zone_cutoffs[:-1])

    return np.array([
        'Zone '+str(z+1) for z in smoothed_zones
    ])

def disp_dynamics(
    df, feature, color='blue',
    std_df=None, ylabel='Expression', 
    dpi=100, figsize=(6, 3),
    zone_assignments=None, zone_cmap='Set3'
):
    r"""
    Plot curve dynamics with optional zone colorbar.
    """    
    n_bins = df.shape[0]

    # Adjust figure layout if zones are provided
    if zone_assignments is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create main plot with space for zone colorbar
        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=8)
        zone_ax = plt.subplot2grid((10, 1), (9, 0), rowspan=1)
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
    ax.set_title(feature, fontsize=16)
    
    # Add zone colorbar if provided
    if zone_assignments is not None:
        # Create zone colorbar
        unique_zones = np.unique(zone_assignments)
        n_zones = len(unique_zones)
        
        # Create colormap and normalization
        zone_colors = plt.cm.get_cmap(zone_cmap, n_zones)
        zone_to_idx = {zone: i for i, zone in enumerate(unique_zones)}
        
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
        for zone in unique_zones:
            zone_mask = zone_assignments == zone
            if np.any(zone_mask):
                # Find center position of this zone
                indices = np.where(zone_mask)[0]
                center_pos = (indices[0] + indices[-1]) / 2
                zone_positions.append(center_pos)
                zone_labels.append(zone)
        
        # Add text labels for zones
        for pos, label in zip(zone_positions, zone_labels):
            zone_ax.text(pos, 0.5, label, ha='center', va='center', 
                            fontsize=8, fontweight='bold')
        
        # Remove x-axis label from main plot
        ax.set_xlabel(r'Pseudotime ($t$) (PV $\rightarrow$ CV bins)', fontsize=12)
        ax.set_xticks(np.arange(0, n_bins, n_bins//5))
        
        # Add colorbar title
        zone_ax.set_title('', pad=5)
        
        # Match x-axis limits between main plot and colorbar
        ax.set_xlim(-0.5, n_bins-0.5)
        
    else:
        ax.set_xlabel(r'Pseudotime ($t$) (PV $\rightarrow$ CV bins)', fontsize=12)
    
    return fig, ax

def plot_stacked_dynamics(df, title=None, figsize=(8, 4), zone_assignments=None, zone_cmap='Set3'):
    if zone_assignments is not None:
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=9)
        zone_ax = plt.subplot2grid((12, 1), (10, 0), rowspan=1)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    df.plot(
        kind='bar', 
        stacked=True, 
        width=1.0,
        edgecolor='black',
        linewidth=0.2,
        ax=ax,
        legend=False
    )

    ax.set_xlabel(r'Pseudotime ($t$) (PV $\rightarrow$ CV bins)')
    ax.set_ylabel('Proportion')
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(df)-0.5)
    ax.set_ylim(0, 1)
    ax.grid(False)
    
    ax.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0,
        frameon=False,
        fontsize='small'
    )
    
    if title:
        ax.set_title(title, fontsize=15)
    
    if zone_assignments is not None:
        unique_zones = np.unique(zone_assignments)
        n_zones = len(unique_zones)
        zone_colors = plt.cm.get_cmap(zone_cmap, n_zones)
        zone_to_idx = {zone: i for i, zone in enumerate(unique_zones)}
        zone_indices = np.array([zone_to_idx[m] for m in zone_assignments])
        
        zone_ax.imshow(
            zone_indices.reshape(1, -1), 
            aspect='auto', 
            cmap=zone_colors,
            extent=[-0.5, len(df)-0.5, 0, 1]
        )
        
        zone_ax.set_xlim(-0.5, len(df)-0.5)
        zone_ax.set_ylim(0, 1)
        zone_ax.set_xticks([])
        zone_ax.set_yticks([])
        
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
    else:
        plt.tight_layout()
        
    return fig, ax

# %%
n_bins = 50
cluster_labels = adata_xenium.obs[cluster_key].cat.categories.to_list()
smoothed_zones = smooth_zone_assignments(adata_xenium, n_bins=n_bins)

celltype_dynamic_df = utils.get_celltype_dynamics(
    adata_xenium, adata_xenium.obs[cluster_key], n_bins=n_bins
)

for label in cluster_labels:
    disp_dynamics(
        celltype_dynamic_df, figsize=(7, 3), 
        ylabel='Proportion', color='mediumblue',
        feature=label, zone_assignments=smoothed_zones
    )

del label

# %%
n_bins = 50
smoothed_zones = smooth_zone_assignments(adata_xenium, n_bins=n_bins)

fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300,
    ylabel='Proportion', color='mediumblue',
    feature='Endothelial', zone_assignments=smoothed_zones
)
fig.savefig('../figures/LYNX_Fig2_endothelial.pdf', bbox_inches='tight')


fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300,
    ylabel='Proportion', color='mediumblue',
    feature='LSECs', zone_assignments=smoothed_zones
)
fig.savefig('../figures/LYNX_Fig2_lsecs.pdf', bbox_inches='tight')

fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300,
    ylabel='Proportion', color='mediumblue',
    feature='Myeloid', zone_assignments=smoothed_zones
)
fig.savefig('../figures/LYNX_Fig2_myeloid.pdf', bbox_inches='tight')

# %%
fig, ax = plot_stacked_dynamics(
    celltype_dynamic_df, 
    zone_assignments=smoothed_zones,
    figsize=(6, 3),
    title='Cell-type Dynamics'
)
fig.savefig('../figures/LYNX_Fig2_celltype_dynamics.pdf', bbox_inches='tight')

# %%
# (ii). Evaluate cell-cell interaction represented by cell-to-cell edge features
# (2.1) Visualize spatial interaction within a local niche

cell_boundaries_filename = os.path.join(xenium_path, sample_id, 'cell_boundaries.parquet')

# # Visualize spatial cell-type distribution
# sq.pl.spatial_scatter(
#     adata_xenium, color='subtype',
#     groups=['Progenitor+Cholangiocytes', 'PC-Hep', 'PP-Hep'],
#     size=25, img=False,
# )


# %% 
# E.g. random cells
rand_indices= np.random.choice(adata_xenium.n_obs, size=5, replace=False)
for idx in rand_indices:
    subgraph_dict = plot.disp_spatial_interaction(
        adata_xenium,
        target_idx=idx,
        cell_boundaries_parquet=cell_boundaries_filename,
        cluster_key=cluster_key,
        return_subgraph=True
    )
    print(subgraph_dict['omega'].sum())
del idx

# %%
# E.g. SMCs
adata_subset = adata_xenium.copy()
adata_subset.obs.reset_index(inplace=True, drop=True)
for idx in adata_subset.obs[adata_subset.obs[cluster_key] == 'SMCs'].sort_values('t').index[:5]:
    plot.disp_spatial_interaction(
        adata_xenium,
        target_idx=idx,
        cell_boundaries_parquet=cell_boundaries_filename,
        cluster_key=cluster_key,
    )
del idx, adata_subset


# %%
# (2.2) Statistical test vs. abundance
# (a). Retrieve overview summary of cell-cell interaction (apriori to abundance test)
adata_xenium.obs[cluster_key] = adata_xenium.obs[cluster_key].astype('category')
cluster_labels=adata_xenium.obs[cluster_key].cat.categories
cci_df = plot.summarize_cell_interaction(
    adata_xenium, 
    cluster_key=cluster_key, 
    cluster_labels=cluster_labels,
    title='Summary of cell-cell interaction (Overall)\n w/o abundance-test',
    show_plot=False
)

cci_df, qval_df = test_assoc.test_cci(
    adata_xenium, cci_df, 
    cluster_key=cluster_key,
    cluster_labels=cluster_labels    
)

plot.disp_heatmap(
    cci_df, 
    title='Cell-cell interaction strength',
)

plot.disp_heatmap(
    qval_df, 
    title='Cell-cell interaction significance (-log10 q-val)',
)

# %%
# (b). Zone-specific cell-cell interaction
for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_cci_df = plot.summarize_cell_interaction(
        adata_sub, 
        cluster_key=cluster_key,
        cluster_labels=cluster_labels,
        show_plot=False
    )

    plot.netVisual_circle(
        zone_cci_df, vertex_size_max=20, figsize=(15, 15),
        title=f'Interaction strength (pre-filter)\n (Zone {int(cluster_id)})' 
    )  
    
    zone_cci_df, zone_qval_df = test_assoc.test_cci(
        adata_sub, zone_cci_df, 
        cluster_key=cluster_key,
        cluster_labels=cluster_labels,
    )
    
    plot.disp_heatmap(
        zone_cci_df, 
        title=f'Summary of cell-cell interaction (Zone {int(cluster_id)})',
    )

    plot.disp_heatmap(
        zone_qval_df, 
        title=f'Significance of cell-cell interaction (Zone {int(cluster_id)})',
    )

    plot.netVisual_circle(
        zone_cci_df, vertex_size_max=20, figsize=(15, 15),
        title=f'Interaction strength\n (Zone {int(cluster_id)})' 
    )   

    plot.netVisual_circle(
        zone_qval_df, vertex_size_max=20, figsize=(15, 15),
        title=f'Interaction significance (-log10 q-val)\n (Zone {int(cluster_id)})' 
    )  

# del zone_cci_df, zone_qval_df
gc.collect()


# %%
# fig, ax = plot.netVisual_circle(
#     cci_dfs[2], vertex_size_max=20, figsize=(15, 15),
#     title=f'Summary of cell-cell interaction strength\n (Zone 3)'
# )
# fig.savefig('../figures/LYNX_Fig2_cci_zone3.pdf', bbox_inches='tight')

# %%
