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
import scanpy as sc
import squidpy as sq

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')
import IO, plot, utils, test_assoc, trajectory

# %%
from IPython.display import display
from matplotlib import rcParams
from matplotlib.axes import Axes
sns.set_context('paper')
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 300})
rcParams.update({'savefig.dpi': 300})

import warnings
from scipy.interpolate import UnivariateSpline
import numpy as np
warnings.filterwarnings('ignore')

%load_ext autoreload
%autoreload 2


# %%
# Load data & processed latent embeddings
data_path = '../data/thymus/'
outdir = '../figures/'
n_latent = 6

sample_ids = sorted([
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
])
sample_id = sample_ids[0]
adata_rna_raw = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
adata_rna = sc.read_h5ad('../results/thymus/lynx_rna_6_Mouse_Thymus1.h5ad')
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))
adata_protein.var_names_make_unique()

# adata_rna.obsm['X_z'] = np.load('../results/thymus/lynx_rna_{0}_{1}.npy'.format(n_latent, sample_id))
adata_protein.obsm['X_z'] = adata_rna.obsm['X_z'].copy()  # paired data, use the primary embedding

# %%
# Suppl plots: comparison of raw RNA / protein expressions vs. LYNX reconstructions
sc.pp.normalize_total(adata_rna_raw, target_sum=1e4)
sc.pp.log1p(adata_rna_raw)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)

# %%
sq.pl.spatial_scatter(
    adata_rna_raw, color=['Cd5', 'Cd44'],
    size=100, img=False, cmap='magma', wspace=-0.1
)

sq.pl.spatial_scatter(
    adata_protein, color=['CD5', 'CD44'],
    size=100, img=False, cmap='magma', wspace=-0.1
)

sq.pl.spatial_scatter(
    adata_rna, color=['Cd5', 'Cd44'],
    size=100, img=False, cmap='magma', wspace=-0.1
)

# %%
# (1). CMA trajectory inference
curve = trajectory.get_curve(adata_rna, trim_radius_ratio=0.25)
trajectory.compute_pseudotime(adata_rna, curve, root_marker='Dcn')
adata_protein.obs['t'] = adata_rna.obs['t'].values 

sq.pl.spatial_scatter(
    adata_rna, color='t',
    size=100, img=False, cmap='RdBu_r',
    title='Inferred Spatial Gradient - LYNX'
)

plot.disp_trajectory(
    adata_rna, cmap='RdBu_r',
    title='Inferred Spatial Gradient\nLYNX embedding'
)

# %%
# (2). Discrete zonation analysis
if 'milestones_colors' in adata_rna.uns_keys():
    adata_rna.uns.pop('milestones_colors')

utils.get_zonation_features(    
    adata_rna, adata_protein,
    n_zones=4, sample_id=sample_id,
    abundance_test=True, show=True, 
)

sq.pl.spatial_scatter(
    adata_rna, color='zone',
    size=100, img=False, 
    title='Inferred spatial zones - LYNX'
)

# %%
# (3). Spatial dynamics of cell-type markers & chemokines along the trajectory
# Markers of interest (Yayon + Liao)

# (a). Thymic epithelial cells (TEC)
tec_markers = [
    'Psmb11', 'Ly75', 'Ccl25',  # Pan-cTEC 
    'Tbata', 'Tp53aip1', 'Dll4', # cTEC subtypes
    'Dlk2', 'Igfbp5', 'Igfbp6', 'Ccn2', 'Ccl2', 'Krt15', 'Itga6', 'Mki67',  # mcTEC subtypes
    'Epcam', # Pan-mTEC
    'Ascl1', 'Ccl21a',   # mTECI 
    'Aire', 'Fezf2', 'Crip1',  # mTECII
    'Slpi', 'Ivl', 'Krt10', 'Cdkn2a'  # mTEC subtypes
]
ctec_markers = [
    'Psmb11', 'Ly75', 'Ccl25',  # Pan-cTEC 
    'Tbata', 'Tp53aip1', 'Dll4', # cTEC subtypes
]
mctec_markers = ['Dlk2', 'Igfbp5', 'Igfbp6', 'Ccn2', 'Ccl2', 'Krt15', 'Itga6', 'Mki67']
mtec_markers = [
    'Epcam', # Pan-mTEC
    'Ascl1', 'Ccl21a',   # mTECI 
    'Aire', 'Fezf2', 'Crip1',  # mTECII
    'Slpi', 'Ivl', 'Krt10', 'Cdkn2a'  # mTEC subtypes
]

tec_markers = [m for m in tec_markers if m in adata_rna.var_names]
ctec_markers = [m for m in ctec_markers if m in adata_rna.var_names]
mctec_markers = [m for m in mctec_markers if m in adata_rna.var_names]
mtec_markers = [m for m in mtec_markers if m in adata_rna.var_names]


# (b). Macrophages
macrophage_markers = [
    'Cd68', 'Cd163', 'Cd11b', 'Cd11c',
    'Timd4', 'Hpgd', 'Serpinb6a', 'Slc40a1', 'Cd81',  # Cortex-enriched Timd4+ markers
    'Cx3cr1', 'Ctsz', 'Cd63', 'Pmepa1', 'Zmynd15', # Medulla-enriched Cx3cr1+ markers
]
cmacro_markers = [
    'Timd4', 'Hpgd', 'Serpinb6a', 'Slc40a1', 'Cd81',  
]
mmacro_markers = [
    'Cx3cr1', 'Ctsz', 'Cd63', 'Pmepa1', 'Zmynd15',
]

macrophage_markers = [m for m in macrophage_markers if m in adata_rna.var_names]
cmacro_markers = [m for m in cmacro_markers if m in adata_rna.var_names]
mmacro_markers = [m for m in mmacro_markers if m in adata_rna.var_names]

# (c). General immune markers
immune_markers = [
    'Cd3d', 'Cd3e', 'Cd4', 'Cd8a', 'Cd8b1',  # T cells
    'Cd19', 'Ptprc', 'Ighd',  # B cells
    'Cd5', 'Cd27', 'Cd44'   # General thymocytes
]
immune_markers = [m for m in immune_markers if m in adata_rna.var_names]


# %% [markdown]
# Compute feature dynamics along the CMA trajectory

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


def disp_matrixplot(
    expr_df, features, 
    title='Expression Dynamics Heatmap', 
    figsize=(8, 6), cmap='RdBu_r', dpi=100,
    vmin=None, vmax=None,
    milestone_assignments=None, milestone_cmap='Set3'
):
    r"""
    Create a matrix plot of expression dynamics with features sorted by peak pseudotime.
    """
    from scipy.stats import zscore
    
    # Filter features that exist in the dataframe
    available_features = [f for f in features if f in expr_df.columns]

    # Get expression data for available features
    # Z-score normalize each feature (row-wise)
    feature_data = expr_df[available_features].T.copy()
    feature_data += 1e-8
    feature_data = feature_data.apply(zscore, axis=1)
    n_bins = feature_data.shape[1]
    
    # Calculate peak positions (argmax) for each feature
    peak_positions = {}
    for feature in available_features:
        peak_idx = np.argmax(feature_data.loc[feature].values)
        peak_positions[feature] = peak_idx
    
    # Sort features by peak position (early peaks at top, late peaks at bottom)
    sorted_features = sorted(available_features, key=lambda x: peak_positions[x])
    sorted_data = feature_data.loc[sorted_features]
    
    # Create figure layout
    if milestone_assignments is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create main heatmap with space for milestone colorbar
        # Use gridspec to control spacing more precisely
        gs = fig.add_gridspec(12, 20, hspace=0.1, wspace=0.3)
        ax = fig.add_subplot(gs[0:10, 0:17])  # Heatmap takes columns 0-16
        cbar_ax = fig.add_subplot(gs[0:10, 18])  # Colorbar takes column 18 (thinner)
        milestone_ax = fig.add_subplot(gs[11, 0:17])  # Milestone bar matches heatmap width exactly
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create heatmap
    im = ax.imshow(
        sorted_data, 
        cmap=cmap, aspect='auto',
        vmin=vmin, vmax=vmax,
        extent=[0, n_bins, 0, len(sorted_features)]
    )
    
    # Add colorbar in the designated space
    if milestone_assignments is not None:
        cbar = plt.colorbar(im, cax=cbar_ax)
    else:
        cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expression (z-score)', fontsize=14)
    
    # Set y-axis (features)
    ax.set_yticks(np.arange(0.5, len(sorted_features), 1))
    ax.set_yticklabels(sorted_features[::-1])
    
    # Set x-axis
    n_ticks = 6
    tick_positions = np.linspace(0, n_bins, n_ticks)
    tick_labels = np.arange(0, n_bins+1, n_bins//5)  # [0, 10, 20, 30, 40, 50]
    ax.set_xticks(tick_positions)
    
    # Add milestone colorbar if provided
    if milestone_assignments is not None:
        unique_milestones = np.unique(milestone_assignments)
        milestone_indices = np.array([np.where(unique_milestones == m)[0][0] 
                                    for m in milestone_assignments])
        
        # Create milestone colorbar with exact same width as heatmap
        milestone_ax.imshow(
            milestone_indices.reshape(1, -1), 
            aspect='auto', 
            cmap=milestone_cmap, 
            extent=[0, n_bins, 0, 1]  # Match heatmap extent exactly
        )
        
        # Match axes limits exactly
        milestone_ax.set_xlim(0, n_bins)
        milestone_ax.set_ylim(0, 1)
        milestone_ax.set_xticks([])
        milestone_ax.set_yticks([])
        
        # Add milestone labels at centers
        for milestone in unique_milestones:
            mask = milestone_assignments == milestone
            if np.any(mask):
                # Calculate center position based on bin indices
                indices = np.where(mask)[0]
                center_pos = (indices[0] + indices[-1]) / 2 + 0.5  # Add 0.5 for center of bin
                milestone_ax.text(center_pos, 0.5, milestone, ha='center', va='center', 
                                fontsize=8, fontweight='bold')
        
        milestone_ax.set_xlabel(r'CMA Gradient coordinate ($t$) (Cortex $\rightarrow$ Medulla bins)', fontsize=14)
        ax.set_xlim(0, n_bins)
        
    else:
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(r'CMA Gradient coordinate ($t$) (Cortex $\rightarrow$ Medulla bins)', fontsize=14)
    
    ax.set_ylabel('Features', fontsize=14)
    ax.set_title(title, fontsize=18)

    plt.tight_layout()
    return fig, ax


def disp_dynamics(
    df_list, feature_list, 
    std_df_list=None, colors=None, labels=None,
    title='',
    ylabel='Expression', 
    dpi=100, figsize=(6, 3),
    spline_ratio=1e-3,
    milestone_assignments=None, 
    milestone_cmap='Set3'
):
    r"""
    Plot curve dynamics with optional milestone colorbar.
    """
    
    # Convert to lists if needed
    if not isinstance(df_list, list):
        df_list = [df_list]
    if not isinstance(feature_list, list):
        feature_list = [feature_list]
    if std_df_list is not None and not isinstance(std_df_list, list):
        std_df_list = [std_df_list]

    # Repeat df_list if multiple features 
    if len(df_list) == 1 and len(feature_list) > 1:
        df_list = df_list * len(feature_list)
        if std_df_list is not None:
            std_df_list = std_df_list * len(feature_list)

    n_curves = len(df_list)
    n_bins = df_list[0].shape[0]
    
    # Set defaults
    colors = colors or plt.cm.tab10(np.linspace(0, 1, 10))[:n_curves]
    labels = labels or feature_list
    
    # Create figure layout
    if milestone_assignments is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=8)
        milestone_ax = plt.subplot2grid((10, 1), (9, 0), rowspan=1)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot curves
    x = np.arange(n_bins)
    for i in range(n_curves):
        y = df_list[i][feature_list[i]]
        color = colors[i]
        
        if std_df_list is None:
            # Spline with uncertainty
            spline = UnivariateSpline(x, y, s=len(x)*spline_ratio) 
            xx = np.linspace(0, n_bins-1, 500)
            yy = spline(xx)
            std_residual = np.std(y - spline(x))
            
            ax.scatter(x, y, s=5, c=color, alpha=0.7)
            ax.plot(xx, yy, linewidth=1, c=color, label=labels[i])
            ax.fill_between(xx, yy - std_residual, yy + std_residual, 
                          color=color, alpha=0.3)
        else:
            # Direct plot with std
            ax.plot(x, y, linewidth=2, linestyle='-.',  color=color, label=labels[i])
            ax.fill_between(x, y - std_df_list[i][feature_list[i]], 
                          y + std_df_list[i][feature_list[i]], 
                          color=color, alpha=0.3)
    
    # Configure main plot
    ax.grid(False)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)

    if title == '':
        ax.set_title(feature_list[0] if n_curves == 1 else 'Expression Dynamics', fontsize=15)
    else:
        ax.set_title(title, fontsize=15)
    
    if n_curves > 1:
        ax.legend(handlelength=0.5, handletextpad=1.)
    
    # Add milestone colorbar
    if milestone_assignments is not None:
        unique_milestones = np.unique(milestone_assignments)
        #milestone_colors = plt.cm.get_cmap(milestone_cmap, len(unique_milestones))
        milestone_indices = np.array([np.where(unique_milestones == m)[0][0] 
                                    for m in milestone_assignments])
        
        milestone_ax.imshow(milestone_indices.reshape(1, -1), aspect='auto', 
                          cmap=milestone_cmap, extent=[-0.5, n_bins-0.5, 0, 1])
        milestone_ax.set_xlim(-0.5, n_bins-0.5)
        milestone_ax.set_ylim(0, 1)
        milestone_ax.set_xticks([])
        milestone_ax.set_yticks([])
        
        # Add milestone labels at centers
        for milestone in unique_milestones:
            mask = milestone_assignments == milestone
            if np.any(mask):
                center_pos = (np.where(mask)[0][[0, -1]]).mean()
                milestone_ax.text(center_pos, 0.5, milestone, ha='center', va='center', 
                                fontsize=8, fontweight='bold')
        
        ax.set_xlabel(r'CMA Gradient coordinate ($t$) (Cortex $\rightarrow$ Medulla bins)', fontsize=12)
        ax.set_xticks(np.arange(0, n_bins, n_bins//5))
        ax.set_xlim(-0.5, n_bins-0.5)
    else:
        ax.set_xlabel(r'CMA Gradient coordinate ($t$) (Cortex $\rightarrow$ Medulla bins)', fontsize=12)
    
    if n_curves > 1:
        plt.tight_layout()
        
    return fig, ax

# %%
n_bins = 50
smoothed_zones = smooth_zone_assignments(adata_rna, n_bins=n_bins)

indices = np.argsort(adata_rna.obs['t']).values
gexp_df, gexp_std_df = utils.get_binned_expr(
    adata_rna.to_df().iloc[indices].T,
    n_bins=n_bins, std=True,
)

mexp_df, mexp_std_df = utils.get_binned_expr(
    adata_protein.to_df().iloc[indices].T,
    n_bins=n_bins, std=True
)

# %%
fig, ax = disp_matrixplot(
    gexp_df, tec_markers, figsize=(7, 6), cmap='seismic', dpi=300,
    milestone_assignments=smoothed_zones,
    title='Stereo-seq TEC Expression Dynamics'
)
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_TEC_heatmap.pdf'), bbox_inches='tight')    

# %%
fig, ax = disp_matrixplot(
    gexp_df, macrophage_markers, figsize=(7, 6), cmap='seismic', dpi=300,
    milestone_assignments=smoothed_zones,
    title='Stereo-seq Macrophage Expression Dynamics'
)
fig.savefig(os.path.join(outdir, 'Suppl4_Macrophage_heatmap.pdf'), bbox_inches='tight')    

fig, ax = disp_matrixplot(
    gexp_df, immune_markers, figsize=(7, 6), cmap='seismic', dpi=300,
    milestone_assignments=smoothed_zones,
    title='Stereo-seq Immune Expression Dynamics'
)
fig.savefig(os.path.join(outdir, 'Suppl4_Immune_heatmap.pdf'), bbox_inches='tight')    

# %%
sq.pl.spatial_scatter(
    adata_rna, color=['Psmb11', 'Mki67', 'Ighg2c', 'Serpinb2'],
    wspace=-0.1, size=100, img=False, cmap='magma'
)

fig, ax = disp_dynamics(
    df_list=gexp_df,
    feature_list=['Psmb11', 'Mki67', 'Ighg2c', 'Serpinb2'],
    ylabel='Expression', spline_ratio=5e-3,
    milestone_assignments=smoothed_zones,
    figsize=(6, 3), colors=['mediumblue', 'coral', 'red', 'green'], dpi=300,
    title='CITE-seq Expression Dynamics'
)


# %%
# Name correction
mexp_df['F4/80'] = mexp_df['F480'].copy()
mexp_std_df['F4/80'] = mexp_std_df['F480'].copy()

fig, ax = disp_dynamics(
    df_list=mexp_df,
    feature_list=['F4/80', 'CD169'],
    # std_df_list=mexp_std_df,
    ylabel='Expression', spline_ratio=5e-3,
    milestone_assignments=smoothed_zones,
    figsize=(6, 3), colors=['mediumblue', 'coral'], dpi=300,
    title='Stereo-seq Expression Dynamics'
)
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_CITE_dynamics1.pdf'), bbox_inches='tight')

fig, ax = disp_dynamics(
    df_list=mexp_df,
    feature_list=['CD4', 'CD8a'],
    # std_df_list=mexp_std_df,
    ylabel='Expression', spline_ratio=5e-3,
    milestone_assignments=smoothed_zones,
    figsize=(6, 3), colors=['red', 'green'], dpi=300,
    title='CITE-seq Expression Dynamics'
)
fig.savefig(os.path.join(outdir, 'LYNX_Fig4_CITE_dynamics2.pdf'), bbox_inches='tight')

# %%