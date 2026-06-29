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
rcParams.update({'figure.dpi': 300})
rcParams.update({'savefig.dpi': 300})

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')
import IO, plot, utils, test_assoc, trajectory

from importlib import reload
%matplotlib inline
%load_ext autoreload
%autoreload 2


# %%
# Helper functions 
from scipy.interpolate import UnivariateSpline
from skimage.filters import threshold_otsu

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
        'Zone '+str(z) for z in smoothed_zones
    ])

def disp_dynamics(
    df, feature, smooth_multiplier=1e-3, 
    std_df=None, ylabel='Expression',
    dpi=100, figsize=(6, 3), show=True,
    color='blue', zone_assignments=None, zone_cmap='Set3',
    ax=None, zone_ax=None, add_zone_bar=True
):
    r"""Plot feature dynamics on a provided axis (or create a new figure)."""
    n_bins = df.shape[0]
    created_fig = False

    # Create axes only if not provided
    if ax is None:
        created_fig = True
        if zone_assignments is not None and add_zone_bar:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = plt.subplot2grid((120, 1), (0, 0), rowspan=80)
            zone_ax = plt.subplot2grid((120, 1), (85, 0), rowspan=12)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        if zone_assignments is not None and add_zone_bar and zone_ax is None:
            # Inset zone strip for subplot mode (bottom fixed at -0.22, height 1.1x)
            zone_ax = ax.inset_axes([0.0, -0.22, 1.0, 0.132], transform=ax.transAxes)

    x = np.arange(n_bins)
    y = df[feature].values

    if std_df is None:
        spline = UnivariateSpline(x, y, s=len(x) * smooth_multiplier)
        xx = np.linspace(x.min(), x.max(), 500)
        yy = spline(xx)
        y_pred = spline(x)
        std_residual = np.std(y - y_pred)

        ax.scatter(x, y, s=5, c=color, alpha=0.7)
        ax.plot(xx, yy, linewidth=1, c=color)
        ax.fill_between(xx, yy - std_residual, yy + std_residual, color=color, alpha=0.3)
    else:
        ax.plot(x, y, linewidth=2, color=color, linestyle='-.')
        ax.fill_between(
            x,
            y - std_df[feature].values,
            y + std_df[feature].values,
            color=color,
            alpha=0.3
        )

    ax.grid(False)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(feature, fontsize=16)
    ax.set_xlabel(r'Gradient coordinate ($t$) (PV $\rightarrow$ CV bins)', fontsize=12, labelpad=15)
    ax.set_xticks(np.arange(0, n_bins, max(1, n_bins // 5)))
    ax.set_xlim(-0.5, n_bins - 0.5)

    if zone_assignments is not None and add_zone_bar and zone_ax is not None:
        unique_zones = pd.unique(zone_assignments)
        n_zones = len(unique_zones)

        if hasattr(zone_cmap, "__call__"):  # colormap object
            cmap = zone_cmap
        else:  # string
            cmap = plt.cm.get_cmap(zone_cmap, n_zones)

        zone_to_idx = {z: i for i, z in enumerate(unique_zones)}
        zone_indices = np.array([zone_to_idx[z] for z in zone_assignments])

        zone_ax.imshow(
            zone_indices.reshape(1, -1),
            aspect='auto',
            cmap=cmap,
            extent=[-0.5, n_bins - 0.5, 0, 1]
        )
        zone_ax.set_xlim(-0.5, n_bins - 0.5)
        zone_ax.set_ylim(0, 1)
        zone_ax.set_xticks([])
        zone_ax.set_yticks([])

        for zone in unique_zones:
            idx = np.where(zone_assignments == zone)[0]
            if len(idx) > 0:
                center = (idx[0] + idx[-1]) / 2
                zone_ax.text(center, 0.5, str(zone), ha='center', va='center', fontsize=7, fontweight='bold')

    if show and created_fig:
        plt.show()

    return fig, ax


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
adata_xenium = sc.read_h5ad('../results/liver/LYNX_xenium_6_0512.h5ad')
adata_desi.obsm['X_z'] = np.load(
    '../results/liver/LYNX_desi_6_0512.npy'
).astype(np.float32)

if 'cell_type' in adata_xenium.obs.keys():
    adata_xenium.obs.drop('cell_type', axis=1, inplace=True) # Remove archived cell-type annotation if exists


# %%
# (i). Trajectory Inference
# Xenium gradient 
curve = trajectory.get_curve(adata_xenium, epg_lambda=0.01, trim_radius_ratio=0.25)
trajectory.compute_pseudotime(adata_xenium, curve, root_marker='DPT')

sq.pl.spatial_scatter(
    adata_xenium, color='t', 
    cmap='RdBu_r', size=25, img=False,
    title='Inferred spatial Gradient\nLYNX'
)

plot.disp_trajectory(
    adata_xenium, 
    cmap='RdBu_r',
    title='Inferred Spatial Gradient\nLYNX embedding'
)

# DESI gradient
curve = trajectory.get_curve(adata_desi, epg_lambda=0.01, trim_radius_ratio=0.25)
trajectory.compute_pseudotime(adata_desi, curve, root_marker='Taurine [M-H]-')

sq.pl.spatial_scatter(
    adata_desi, color='t', 
    cmap='RdBu_r', size=1, img=False,
    title=r'Spatial Gradient $(t)$'+'\nLYNX (DESI)'
)

plot.disp_trajectory(
    adata_desi, 
    cmap='RdBu_r',
    title='Spatial Gradients\n LYNX (DESI)'
)

# %%
# Assign zones from continuous gradients, compute zone-specific features
n_zones = 4
if adata_xenium.X.toarray()[adata_xenium.X.toarray() > 0].min() == 1.0:
    sc.pp.normalize_total(adata_xenium, target_sum=1e4)
    sc.pp.log1p(adata_xenium)

utils.get_zonation_features(    
    adata_xenium, 
    adata_desi,
    n_zones=n_zones, sample_id=sample_id,
    abundance_test=True,
    show=False
)

# %%
set3_cmap = plt.cm.get_cmap('Set3', n_zones+1)
zone_colors = [set3_cmap(i) for i in range(n_zones)]
zone_cmap = plt.cm.colors.ListedColormap(zone_colors)
adata_xenium.uns['zone_colors'] = zone_colors
sq.pl.spatial_scatter(
    adata_xenium, color='zone', title='LYNX inferred zones',
    size=25, img=False,
)

# %%
# Novae inferred zones (benchmark comparison)
novae_labels = np.load('../results/liver/Novae_xenium_seg_labels.npy')
assert len(novae_labels) == adata_xenium.n_obs, \
    "Novae labels do not match adata_xenium cells"
novae_labels = pd.Series(novae_labels).replace('nan', np.nan)
adata_xenium.obs['novae_zone'] = pd.Categorical(novae_labels)

n_novae_zones = adata_xenium.obs['novae_zone'].cat.categories.size
novae_set3 = plt.cm.get_cmap('Set3', n_novae_zones + 1)
adata_xenium.uns['novae_zone_colors'] = [
    novae_set3(i) for i in range(n_novae_zones)
]
sq.pl.spatial_scatter(
    adata_xenium, color='novae_zone', title='Novae inferred zones',
    size=25, img=False,
)

# %%
# "DEG" & "DEM" summary per zone
fig, ax = plot.disp_joint_logfc(
    adata_xenium, 
    adata_desi,  
    zones=adata_xenium.obs['zone'].cat.categories.astype('str'),
    title='Representative zone features',
    show=False
)
plt.show()
fig.savefig('../figures/LYNX_Fig2_zone_features.pdf', bbox_inches='tight')


# %%
# Multi-panel compilation of cell-type dynamics plots
n_bins = 50
cluster_labels = adata_xenium.obs[cluster_key].cat.categories.to_list()
smoothed_zones = smooth_zone_assignments(adata_xenium, n_bins=n_bins)

celltype_dynamic_df = utils.get_celltype_dynamics(
    adata_xenium, adata_xenium.obs[cluster_key], n_bins=n_bins
)

n_panels = len(cluster_labels)
n_cols = 4
n_rows = int(np.ceil(n_panels / n_cols))

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5 * n_cols, 2.5 * n_rows),
    dpi=500,
    squeeze=False
)

axes_flat = axes.ravel()
for i, label in enumerate(cluster_labels):
    disp_dynamics(
        celltype_dynamic_df,
        feature=label,
        ylabel='Proportion',
        color='mediumblue',
        zone_assignments=smoothed_zones,
        zone_cmap=zone_cmap,
        ax=axes_flat[i],
        add_zone_bar=True, 
        show=False,
    )

for ax in axes_flat[n_panels:]:
    ax.axis('off')

fig.tight_layout()
fig.savefig('../figures/LYNX_Fig2_celltype_dynamics_all.svg', bbox_inches='tight')

# %%
# Validate Kupffer cell marker (CD68 & MARCO)
n_bins = 50
adata_kupffer = adata_xenium[adata_xenium.obs[cluster_key] == 'Kupffer'].copy()
indices = np.argsort(adata_kupffer.obs['t'].values)
marker_gexp_df, _ = utils.get_binned_expr(
    adata_kupffer.to_df().iloc[indices].T,
    n_bins=n_bins, 
    std=True
)

disp_dynamics(
    marker_gexp_df, smooth_multiplier=1e-1, dpi=300,
    ylabel='Proportion', color='mediumblue',
    zone_assignments=smoothed_zones,
    zone_cmap=zone_cmap,
    feature='CD68', figsize=(6, 2.5)
)

disp_dynamics(
    marker_gexp_df, smooth_multiplier=1e-1, dpi=300,
    ylabel='Proportion', color='mediumblue',
    zone_assignments=smoothed_zones,
    zone_cmap=zone_cmap,
    feature='MARCO', figsize=(6, 2.5)
)
del marker_gexp_df, indices,

# %%
features = ['MARCO', 'CD68']
cmaps = ['Reds', 'Reds']
titles = ['MARCO', 'CD68']

fig, axes = plt.subplots(1, 2, figsize=(10, 7), dpi=200)

for i, (feat, cmap, title) in enumerate(zip(features, cmaps, titles)):
    sq.pl.spatial_scatter(
        adata_kupffer,
        color=feat,
        cmap=cmap,
        size=100,
        img=False,
        colorbar=False,
        ax=axes[i],
        alpha=0.8,
        return_ax=False,
        title=title,
    )

    # Add per-panel colorbar
    if len(axes[i].collections) > 0:
        sm = axes[i].collections[-1]
        plt.colorbar(sm, ax=axes[i], shrink=0.3, aspect=20)

plt.show()
# fig.savefig('../figures/LYNX_Fig2_kupffer_markers.png', bbox_inches='tight')
del feat, cmap, title, features, cmaps, titles

# %%
n_bins = 50
celltype_dynamic_df = utils.get_celltype_dynamics(
    adata_xenium, adata_xenium.obs[cluster_key], n_bins=n_bins
)
smoothed_zones = smooth_zone_assignments(adata_xenium, n_bins=n_bins)

# %%
fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300, figsize=(6, 3),
    ylabel='Proportion', color='mediumblue',
    feature='Vascular Endothelial', 
    zone_assignments=smoothed_zones,
    zone_cmap=zone_cmap,
)
fig.savefig('../figures/LYNX_Fig2_endothelial.pdf', bbox_inches='tight')


fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300, figsize=(6, 3),
    ylabel='Proportion', color='mediumblue',
    feature='LSECs', 
    zone_assignments=smoothed_zones,
    zone_cmap=zone_cmap,
)
fig.savefig('../figures/LYNX_Fig2_lsecs.pdf', bbox_inches='tight')

fig, ax = disp_dynamics(
    celltype_dynamic_df, dpi=300, figsize=(6, 3),
    ylabel='Proportion', color='mediumblue',
    feature='Kupffer', 
    zone_assignments=smoothed_zones,
    zone_cmap=zone_cmap
)
fig.savefig('../figures/LYNX_Fig2_kupffer.pdf', bbox_inches='tight')

# %%
sq.pl.spatial_scatter(
    adata_xenium, color=cluster_key, palette='tab20',
    size=25, img=False,
)

fig, ax = plot.disp_stacked_dynamics(
    celltype_dynamic_df, 
    zone_assignments=smoothed_zones,
    zone_cmap=zone_cmap,
    colors=adata_xenium.uns['subtype_colors'],
    figsize=(6, 3.3),
    xlabel_desc=r' (PV $\rightarrow$ CV bins)',
    title='Cell-type Dynamics'
)
plt.show()
fig.savefig('../figures/LYNX_Fig2_celltype_dynamics.pdf', bbox_inches='tight')

# %%
# (ii). Evaluate cell-cell interaction represented by cell-to-cell edge features
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

# %%
# (b). Zone-specific cell-cell interaction
cci_dfs = []
qval_dfs = []
for cluster_id in sorted(adata_xenium.obs['zone'].unique()):
    adata_sub = adata_xenium[adata_xenium.obs['zone'] == cluster_id].copy()
    zone_cci_df = plot.summarize_cell_interaction(
        adata_sub, 
        cluster_key=cluster_key,
        cluster_labels=cluster_labels,
        show_plot=False
    )

    _, zone_qval_df = test_assoc.test_cci(
        adata_sub, zone_cci_df, 
        cluster_key=cluster_key,
        cluster_labels=cluster_labels,
    )

    plot.netVisual_circle(
        zone_cci_df, vertex_size_max=15,
        colors=adata_xenium.uns['subtype_colors'], figsize=(26, 26),
        show_celltype_legend=True,
        title=f'Interaction strength (Zone {int(cluster_id)})' 
    )   

    # plot.netVisual_circle(
    #     zone_qval_df, vertex_size_max=15, 
    #     colors=adata_xenium.uns['subtype_colors'], figsize=(23, 21),
    #     show_celltype_legend=False,
    #     edge_legend_label=r'$-\log_{10}$(p-val)',
    #     title=f'Interaction significance (Zone {int(cluster_id)})' 
    # )   

    cci_dfs.append(zone_cci_df)
    qval_dfs.append(zone_qval_df)

    break

del zone_cci_df, zone_qval_df
gc.collect()


# %%
# Note: saving plots with 0-index for zones (0,1,2,...)
fig, ax = plot.netVisual_circle(
    qval_dfs[1], vertex_size_max=15, 
    colors=adata_xenium.uns['subtype_colors'], figsize=(23, 23),
    edge_legend_label=r'$-\log_{10}$(p-val)',
    title=f'Interaction significance (Zone 1)' 
)
fig.savefig('../figures/LYNX_Fig2_cci_zone1.pdf', bbox_inches='tight')

fig, ax = plot.netVisual_circle(
    qval_dfs[2], vertex_size_max=15, 
    colors=adata_xenium.uns['subtype_colors'], figsize=(23, 23),
    edge_legend_label=r'$-\log_{10}$(p-val)',
    title=f'Interaction significance (Zone 2)' 
)
fig.savefig('../figures/LYNX_Fig2_cci_zone2.pdf', bbox_inches='tight')

fig, ax = plot.netVisual_circle(
    qval_dfs[3], vertex_size_max=20, 
    colors=adata_xenium.uns['subtype_colors'], figsize=(23, 23),
    edge_legend_label=r'$-\log_{10}$(p-val)',
    title=f'Interaction significance (Zone 3)' 
)
fig.savefig('../figures/LYNX_Fig2_cci_zone3.pdf', bbox_inches='tight')

# %%
