import os
import sys
import re
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import squidpy as sq
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.stats import pearsonr, spearmanr
from scipy.special import comb
from typing import Dict, List
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.filters import threshold_otsu

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import get_binned_expr

# Set font family
from matplotlib import rcParams
import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Liberation Sans' in available_fonts:
    rcParams['font.family'] = 'Liberation Sans'
elif 'Helvetica' in available_fonts:
    rcParams['font.family'] = 'Helvetica'
elif 'Arial' in available_fonts:
    rcParams['font.family'] = 'Arial'


def generate_random_colors(n):
    random_colors = []
    for _ in range(n):
        # Generate random RGB values
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)

        # Convert RGB values to hexadecimal color code
        color_code = "#{:02x}{:02x}{:02x}".format(r, g, b)
        random_colors.append(color_code)
    return random_colors


def disp_heatmap(df, xlabel='Receiver', ylabel='Sender', title=''):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, cmap="magma", linecolor='gray', linewidth=0.5)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=20)
    plt.show()


def disp_gradient(
    feature_means, feature_stds,
    figsize=(10, 3), dpi=200,
    vmin=0, vmax=1,
    title=None
):
    r"""
    Display expressions of a single feature along the trajectory
    """
    xx = np.linspace(0, 1, len(feature_means))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(
        xx, feature_means, 
        'b-.', marker='o', linewidth=0.5, markersize=0.7, label='mean'
    )
    ax.fill_between(
        xx, feature_means-feature_stds, feature_means+feature_stds, 
        alpha=0.2, label='Uncertainty'
    )
    ax.legend()
    ax.set_title(title)

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylim([vmin, vmax])

    ax.set_xlabel('Trajectory')
    ax.set_ylabel('Smoothed expression')

    plt.show()


def disp_factor_corr(z):
    z_corr = np.corrcoef(z.T)
    z_score = np.abs(np.tril(z_corr, k=-1)).sum() / comb(z_corr.shape[0], 2)

    g = sns.clustermap(z_corr, cmap='RdBu_r')
    g.figure.suptitle(
        'q(z)\n Correlation score: {}'.format(np.round(z_score, 3)), 
        fontsize=30, y=1.05
    )
    plt.show()


def disp_spatial_latents(adata, latent, ncols=3, cmap='turbo', vmax=None):
    assert adata.shape[0] == latent.shape[0], \
        "Inconsistent # samples btw inference & dataset"
    labels = ['Z'+str(i) for i in range(latent.shape[1])]
    for label, z_k in zip(labels, latent.T):
        adata.obs[label] = z_k
    sq.pl.spatial_scatter(
        adata, color=labels, vmax=vmax, cmap=cmap, 
        size=20, img=False, ncols=ncols
    )
    adata.obs.drop(labels, axis=1, inplace=True)
    
    return None


def disp_trajectory(
    adata, 
    use_rep=None,
    figsize=(5, 4),
    cmap='RdBu_r',
    title=None
):
    if use_rep is None:
        use_rep = 'X_z'
    else:
        assert use_rep in adata.obsm.keys()

    principal_repr = adata.uns['graph']['F'][
        adata.uns['graph']['pnode_indices']
    ]
    n_nodes = principal_repr.shape[0]
    adata_repr = sc.AnnData(
        np.vstack([adata.obsm[use_rep], principal_repr])
    )
    sc.pp.neighbors(adata_repr)
    sc.pp.pca(adata_repr, n_comps=adata_repr.shape[1]-1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.scatter(
        adata_repr.obsm['X_pca'][:-n_nodes, 0],
        adata_repr.obsm['X_pca'][:-n_nodes, 1],
        c=adata.obs['t'], s=0.1, edgecolors=None, cmap=cmap
    )
    ax.plot(
        adata_repr.obsm['X_pca'][-n_nodes:, 0],
        adata_repr.obsm['X_pca'][-n_nodes:, 1],
        '.-', color='gray', lw=.5, ms=2, mfc='yellow'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    cb = plt.gcf().axes[-1]
    cb.set_ylabel(r'Pseudotime $(t)$', fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.show()


def disp_celltype_dynamics(dynamics_df, ncols=4, title='', savedir=None):
    """
    Display cell-type dynamics along the zonation trajectory
    """
    n_bins, n_cell_types = dynamics_df.shape
    nrows = n_cell_types // ncols
    if n_cell_types % ncols != 0:
        nrows += 1

    idx = 0
    x = np.linspace(0, 1, n_bins)
    f = lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*2), dpi=300)
    for row in range(nrows):
        for col in range(ncols):
            if idx >= n_cell_types:
                axes[row, col].axis('off')
                continue

            y = dynamics_df.iloc[:, idx]
            xx = np.linspace(x.min(), x.max(), 500)
            a, b, c, d, e = np.polyfit(x, y, 4)
            yy = f(xx, a, b, c, d, e)
            
            axes[row, col].scatter(
              x, dynamics_df.iloc[:, idx], 
              s=2, c='k', alpha=0.5
            )
            axes[row, col].plot(xx, yy, color='b', linewidth=2, alpha=0.5)
            axes[row, col].set_title(dynamics_df.columns[idx], fontsize=12)
            axes[row, col].set_xlabel('Spatial gradient\n'+ title, fontsize=10)
            axes[row, col].set_ylabel('Proportions')
            axes[row, col].spines[['right', 'top']].set_visible(False)
            idx += 1
    
    plt.tight_layout()
    plt.show()

    if savedir:
        fig.savefig(savedir, bbox_inches="tight", dpi=300)


def disp_stacked_dynamics(
    df, 
    zone_assignments=None, 
    zone_cmap='Set3',
    colors=None,
    title=None, 
    figsize=(8, 4),
):
    if zone_assignments is not None:
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=9)
        zone_ax = plt.subplot2grid((12, 1), (10, 0), rowspan=1)
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    if colors is not None:
        df.plot(
            kind='bar', 
            stacked=True, 
            width=1.0,
            edgecolor='black',
            linewidth=0.2,
            ax=ax,
            color=colors,
            legend=False
        )
    else:
        df.plot(
            kind='bar', 
            stacked=True, 
            width=1.0,
            edgecolor='black',
            linewidth=0.2,
            ax=ax,
            cmap='tab20',
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


def disp_kde_scatter(
    x_true: np.ndarray, 
    x_pred: np.ndarray, 
    indices: List[int] = None,
    logscale=True,
    subset_ratio : float = 0.01,
    size=1., 
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    show_plot: bool = True,

):
    r"""Reconstruction plot w/ density"""
    # Subsample data points for faster KDE visualization
    if indices is None:
        indices = np.random.choice(
            np.arange(len(x_true)), int(subset_ratio*len(x_true)), replace=False
        )

    if logscale:
        x_true = np.log1p(x_true)
        x_pred = np.log1p(x_pred)  

    v_stacked = np.vstack([x_true[indices], x_pred[indices]])
    density = gaussian_kde(v_stacked)(v_stacked)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.scatter(x_true[indices], x_pred[indices], 
               s=size, c=density, cmap='turbo')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15)

    text_xloc = 0.05*(ax.get_xlim()[1]-ax.get_xlim()[0])
    text_yloc = 0.95*ax.get_ylim()[1]
    if logscale:
        corr = pearsonr(x_true, x_pred)[0]
        ax.annotate(r"$r$ = {:.3f}".format(corr),
            (text_xloc, text_yloc), fontsize=12
        )
    else:
        corr = spearmanr(x_true, x_pred)[0]
        ax.annotate(r"$r_s$ = {:.3f}".format(corr),
            (text_xloc, text_yloc), fontsize=12
        )

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.grid(False)
    
    if show_plot:
        plt.show() 
    else:
        return fig, ax


def disp_feature_dynamics(
    expr_df, 
    feature, 
    std_df=None,
    figsize=(6, 2.5)
):
    r"""Plot feature dynamics across the zonation trajectory"""
    n_bins = expr_df.shape[1]
    x = np.arange(n_bins)
    y = expr_df.loc[feature]

    plt.figure(figsize=figsize)
    if std_df is None:
        xx = np.linspace(x.min(), x.max(), 500)
        f = lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e
        a, b, c, d, e = np.polyfit(x, y, 4)
        yy = f(xx, a, b, c, d, e)
        plt.scatter(x, expr_df.loc[feature], s=2, c='k', alpha=0.5)
        plt.plot(xx, yy, color='b', linewidth=2, alpha=0.5)

    else:
        plt.plot(x, y, linewidth='.5', c='k', linestyle='-.')
        plt.fill_between(x, y-std_df.loc[feature], y+std_df.loc[feature], color='blue', alpha=.1)

    plt.xlabel(r"PV $\rightarrow$ CV bins", fontsize=12)
    plt.ylabel('Expression', fontsize=12)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title(feature, fontsize=15)
    plt.show()


def disp_sex_feature_dynamics(
    df: pd.DataFrame, 
    feature: str,
    show: bool = True,
    ylabel='Expression',
    ax: Axes = None
):
    """Plot sex-specific feature dynamics across multiple samples"""  
    expr_df = df.T.stack()[feature].reset_index()
    expr_df.columns = ['Bin', 'Expression']
    expr_df['Sex'] = df['sex'].values.copy()

    if ax is None:
        _, ax = plt.subplots()
    ax = sns.lineplot(
        data=expr_df, x='Bin', y='Expression', hue='Sex',
        color='k', linestyle='-.',
        err_kws={'alpha': .1},
        ax=ax, 
    )

    ax.set_xlabel(r"PV $\rightarrow$ CV bins", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.set_title(feature, fontsize=12)
    if show:
        plt.show()
        return None
    else: 
        return ax


def disp_joint_logfc(
    adata_st,
    adata_sm,
    zones,
    st_key='zones',
    sm_key='zones',
    st_name_col='gene',
    sm_name_col='m/z',
    fc_col='logFC',
    top_n=5,
    figsize=(12, 8),
    show=True,
    title="Joint Upregulated Features",
    logfc_threshold=0.5,
    pval_threshold=0.05
):
    """
    Plot joint upregulated genes (positive Y) & metabolites (negative Y) across zones.
    Reference: https://spatialmeta.readthedocs.io/en/latest/api.html#spatialmeta.pl.plot_marker_gene_metabolite
    """
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=figsize)

    # Hide spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 1. Prepare Data
    st_data_list = []
    sm_data_list = []

    # Use default palette if none provided
    if 'zone_colors' in adata_st.uns:
        colors = adata_st.uns['zone_colors']
        palette = {z: colors[i] for i, z in enumerate(zones)}
    else:
        colors = sns.color_palette("Set3", len(zones))
        palette = {z: colors[i] for i, z in enumerate(zones)}

    # Collect ST Data (Genes)
    for z in zones:
        if z in adata_st.uns[st_key]:
            df = adata_st.uns[st_key][z].copy()
            # Keep all positive logFC
            df = df[df[fc_col] > 0]
            if df.empty:
                continue
            df['zone'] = z
            df['type'] = 'Gene'
            df['name'] = df[st_name_col]
            df['logFC_plot'] = df[fc_col] # Positive for genes

            # Determine significance
            df['is_sig'] = (df[fc_col] > logfc_threshold) & (df['pvals_adj'] < pval_threshold)
            st_data_list.append(df)

    # Collect SM Data (Metabolites)
    for z in zones:
        if z in adata_sm.uns[sm_key]:
            df = adata_sm.uns[sm_key][z].copy()
            # Keep all positive logFC (but plotted negatively)
            df = df[df[fc_col] > 0]
            if df.empty:
                continue
            df['zone'] = z
            df['type'] = 'Metabolite'
            # Clean names (remove adducts like [M+H]+)
            df['name'] = df[sm_name_col].apply(lambda x: re.sub(r'\[.*?\]', '', str(x)).strip())
            df['logFC_plot'] = -df[fc_col] # Negative for metabolites

            # Determine significance
            df['is_sig'] = (df[fc_col] > logfc_threshold) & (df['pvals_adj'] < pval_threshold)
            sm_data_list.append(df)

    if not st_data_list and not sm_data_list:
        print("No data found for plotting.")
        return fig, ax

    full_df = pd.concat(st_data_list + sm_data_list, ignore_index=True)

    # Add Jitter for X-axis
    # Map zones to integers 0, 1, 2...
    zone_map = {z: i for i, z in enumerate(zones)}
    full_df['zone_idx'] = full_df['zone'].map(zone_map)

    # Add random jitter
    np.random.seed(42)
    full_df['x_plot'] = full_df['zone_idx'] + np.random.uniform(-0.3, 0.3, len(full_df))

    # 2. Plotting

    # Plot Background (Non-Significant) - Grey, Small
    bg_df = full_df[~full_df['is_sig']]
    if not bg_df.empty:
        ax.scatter(
            bg_df['x_plot'],
            bg_df['logFC_plot'],
            c='lightgrey',
            s=10,
            alpha=0.5,
            edgecolors='none',
            label='Non-significant'
        )

    # Plot Significant - Colored by Zone, Larger
    sig_df = full_df[full_df['is_sig']]
    if not sig_df.empty:
        for z in zones:
            subset = sig_df[sig_df['zone'] == z]
            if subset.empty:
                continue
            ax.scatter(
                subset['x_plot'],
                subset['logFC_plot'],
                c=[palette[z]],
                s=25,
                alpha=0.8,
                edgecolors='none',
                label=z if z not in ax.get_legend_handles_labels()[1] else ""
            )

    # 3. Annotations (Top N Significant with Collision Avoidance)
    text_y_extents = []

    def _annotate_subset(subset_df, ax, color='k', is_top=True):
        """Helper to annotate subset (Genes or Metabolites) per zone"""
        if subset_df.empty:
            return

        for z in zones:
            z_df = subset_df[subset_df['zone'] == z]
            if z_df.empty:
                continue

            # Sort by logFC for consistent stacking
            if is_top: # Genes -> Up
                z_df = z_df.sort_values('logFC_plot', ascending=True)
            else: # Mets -> Down
                z_df = z_df.sort_values('logFC_plot', ascending=False)
            if is_top:
                top_features = z_df.tail(top_n)
            else:
                top_features = z_df.tail(top_n)

            if is_top:
                top_features = top_features.sort_values('logFC_plot', ascending=True)
            else:
                top_features = top_features.sort_values('logFC_plot', ascending=False)

            # Spacing logic
            adjusted_positions = [] # list of (feature, original_x, original_y, text_y)

            prev_text_y = -np.inf if is_top else np.inf

            y_range = full_df['logFC_plot'].max() - full_df['logFC_plot'].min()
            offset = y_range * 0.05 # 5% of range offset for base line
            min_spacing = y_range * 0.03 # 3% min spacing

            for _, row in top_features.iterrows():
                y = row['logFC_plot']
                x = row['x_plot']
                name = row['name']

                if is_top:
                    target_y = y + offset
                    text_y = max(target_y, prev_text_y + min_spacing)
                    prev_text_y = text_y
                else:
                    target_y = y - offset
                    text_y = min(target_y, prev_text_y - min_spacing)
                    prev_text_y = text_y

                adjusted_positions.append((name, x, y, text_y))
                text_y_extents.append(text_y)

            # Draw
            x_center = zone_map[z]
            for name, ox, oy, ty in adjusted_positions:
                # Line from point (ox, oy) to text (x_center, ty)
                ax.annotate(
                    "",
                    xy=(ox, oy),
                    xytext=(x_center, ty),
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.5, alpha=0.6)
                )

                ax.text(
                    x_center,
                    ty,
                    name,
                    color=color,
                    fontsize=8,
                    ha='center',
                    va='bottom' if is_top else 'top',
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none")
                )

    def _multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
        """
        Reference: https://stackoverflow.com/questions/33159134/matplotlib-y-axis-label-with-multiple-colors
        """
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

        # x-axis label
        if axis=='x' or axis=='both':
            boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                        for text,color in zip(list_of_strings,list_of_colors) ]
            xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
            anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,
                                            bbox_to_anchor=(0.2, -0.07),
                                            bbox_transform=ax.transAxes, borderpad=0.)
            ax.add_artist(anchored_xbox)

        # y-axis label
        if axis=='y' or axis=='both':
            boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                        for text,color in zip(list_of_strings[::-1],list_of_colors) ]
            ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, 
                                            bbox_to_anchor=(-0.07, 0.2), 
                                            bbox_transform=ax.transAxes, borderpad=0.)
            ax.add_artist(anchored_ybox)

    # Annotate Genes (Top N Significant)
    genes_sig = sig_df[sig_df['type'] == 'Gene']
    _annotate_subset(genes_sig, ax, color='navy', is_top=True)

    # Annotate Metabolites
    mets_sig = sig_df[sig_df['type'] == 'Metabolite']
    _annotate_subset(mets_sig, ax, color='darkmagenta', is_top=False)

    # Update Limits to fit annotations
    data_min = full_df['logFC_plot'].min()
    data_max = full_df['logFC_plot'].max()

    if text_y_extents:
        y_min = min(min(text_y_extents), data_min)
        y_max = max(max(text_y_extents), data_max)
    else:
        y_min, y_max = data_min, data_max

    y_range_total = y_max - y_min
    padding = y_range_total * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

    # 4. Styling
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels(['zone '+ zone_id for zone_id in zones], fontsize=15)
    ax.set_xlabel(r"Zones (PV $\rightarrow$ CV)", fontsize=15)
    # ax.set_ylabel("LogFC\n"+r"{\color{red}metabolites}  | {\color{blue}genes}", fontsize=15)
    
    _multicolor_ylabel(
        ax,
        ("LogFC  (", "metabolites", " |", "genes", ")"),
        ('k','navy','k','darkmagenta', 'k'),  # Note: color order (top -> bottom!)
        axis='y',size=15
    )
    ax.set_title(title, fontsize=20)

    # Fix Y-axis labels to be absolute
    ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.set_yticklabels([f"{abs(t):.1f}" for t in ticks])

    if show:
        plt.show()
        return None
    else:
        return fig, ax


# -----------------------------------
# Visualize cell-cell interactions
# -----------------------------------

def summarize_cell_interaction(
    adata,  
    ccc_rep='omega', 
    cluster_key='cell_type', 
    cluster_labels=None,
    title='', 
    show_plot=False
):
    r"""Compute cluster-wise summary of cell-cell interactions"""
    if cluster_labels is None:
        cluster_labels = adata.obs[cluster_key].cat.categories

    per_idx_labels = adata.obs[cluster_key].values
    n_clusters = len(cluster_labels)
    mat = np.zeros((n_clusters, n_clusters), dtype=np.float32)

    # Aggregate: for each receiver type, average over its cells
    for i, rtype in enumerate(cluster_labels):
        mask = (per_idx_labels == rtype)
        if mask.sum() > 0:
            mat[i] = adata.obsm[ccc_rep][mask].mean(axis=0)   # sender cell types

    # add omega as an extra sender column
    df = pd.DataFrame(
        mat.T,  # (sender, receiver)
        index=cluster_labels, 
        columns=list(cluster_labels)
    )
    # np.fill_diagonal(df.values, 0)

    # plot heatmap
    if show_plot:
        disp_heatmap(df, title=title)

    return df


def _smooth_polygon(xy, n_iter=2, corner_ratio=0.25):
    """Chaikin corner-cutting algorithm to smooth polygons."""
    for _ in range(n_iter):
        new_points = []
        for i in range(len(xy) - 1):
            p0, p1 = xy[i], xy[i + 1]
            Q = (1 - corner_ratio) * p0 + corner_ratio * p1
            R = corner_ratio * p0 + (1 - corner_ratio) * p1
            new_points.extend([Q, R])
        xy = np.vstack([new_points, new_points[0]])  # close polygon
    return xy


# Visualize spatial microenvironment of a few cells
def disp_spatial_interaction(
    adata, cell_boundaries_parquet, 
    cluster_key='cell_type', target_idx=None,
    edge_cmap='Purples', node_cmap='Set3',
    n_smooth_iter=2, title='', figsize=(10, 8),
    return_subgraph=False
):
    """
    Visualize spatial cell-cell interactions using XeniumRanger cell boundaries,
    with robust polygon smoothing (Chaikin corner cutting).

    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial data and interaction edges.
    cell_boundaries_parquet : str or Path
        Path to Xenium `cell_boundaries.parquet`.
        Columns: ['cell_id', 'vertex_x', 'vertex_y']
    cluster_key : str
        Column in `adata.obs` used for coloring.
    n_smooth_iter : int
        Number of smoothing iterations (0 = no smoothing).
    """

    # --- Load polygon dataframe ---
    df = pd.read_parquet(cell_boundaries_parquet)
    required_cols = {'cell_id', 'vertex_x', 'vertex_y'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
    
    if adata.obs_names.dtype == 'category':
        df['cell_id'] = df['cell_id'].astype('str').astype('category')

    # --- Select target cell ---
    if target_idx is None:
        target_idx = np.random.choice(adata.n_obs)
    if isinstance(target_idx, str):
        target_idx = adata.obs_names.get_loc(target_idx)

    # --- Extract edge information ---
    edge_index = adata.uns['edge_index']
    omega = adata.uns['omega']
    target_mask = edge_index[1] == target_idx
    source_indices = edge_index[0][target_mask]
    edge_weights = omega[target_mask]

    # --- Subgraph nodes ---
    all_nodes = np.concatenate([source_indices, [target_idx]])
    node_names = adata.obs_names[all_nodes]

    # --- Subset polygon dataframe ---
    poly_df = df[df['cell_id'].isin(node_names)].copy()

    # --- Group polygons + apply smoothing ---
    polygons = []
    poly_cell_ids = []
    for cid, g in poly_df.groupby('cell_id'):
        xy = g[['vertex_x', 'vertex_y']].to_numpy()
        if len(xy) < 3:
            continue
        # ensure closed polygon
        if not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[0]])
        if n_smooth_iter > 0:
            xy = _smooth_polygon(xy, n_iter=n_smooth_iter)
        polygons.append(xy)
        poly_cell_ids.append(cid)

    # --- Color by cell type ---
    cell_types = adata.obs.loc[poly_cell_ids, cluster_key]
    unique_types = cell_types.unique()
    colors = plt.cm.get_cmap(node_cmap, len(unique_types))
    type_to_color = dict(zip(unique_types, colors(np.arange(len(unique_types)))))
    face_colors = [type_to_color[ct] for ct in cell_types]

    # --- Plot polygons ---
    fig, ax = plt.subplots(figsize=figsize)
    coll = PolyCollection(polygons, facecolors=face_colors, edgecolors='k', linewidths=0.4, alpha=0.8)
    ax.add_collection(coll)

    # --- Draw weighted edges ---
    spatial_coords = adata.obsm['spatial']
    edge_color_values = edge_weights / 0.2
    edge_colors = plt.cm.get_cmap(edge_cmap)(edge_color_values)
    target_coord = spatial_coords[target_idx]
    for src, w, color in zip(source_indices, edge_weights, edge_colors):
        if w > 0:
            src_coord = spatial_coords[src]
            ax.plot(
                [src_coord[0], target_coord[0]],
                [src_coord[1], target_coord[1]],
                color=color,
                linewidth=1.2 + 5 * (w / 0.2),
                alpha=0.8,
            )

    # --- Highlight target polygon ---
    tgt_id = adata.obs_names[target_idx]
    tgt_label = adata.obs.loc[tgt_id, cluster_key]
    tgt_poly = poly_df[poly_df['cell_id'] == tgt_id][['vertex_x', 'vertex_y']].to_numpy()
    ax.plot(tgt_poly[:, 0], tgt_poly[:, 1], color='black', linewidth=2.2)

    # --- Legend & colorbar ---
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=type_to_color[ct], markersize=8, label=ct)
        for ct in unique_types
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
            ncol=min(5, len(unique_types)), frameon=False, fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=0, vmax=edge_weights.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(r'Interaction Strength ($\omega{ij}$)', rotation=90, labelpad=15, fontsize=12)

    ax.set_title(f'Spatial Interaction (Target: {tgt_label})'+ '\n' + title, fontsize=20)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    if return_subgraph:
        return {'source': source_indices, 'target': target_idx, 'omega': edge_weights}


    
# Functions for cluster-level interaction summary:
# Circular network visualization
# Reference: https://github.com/Starlitnightly/omicverse
def _draw_self_loop(ax, pos, weight, max_weight, color, edge_width_max):
    """
    Draw self-loops (connections from cell type to itself)
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object
    pos : tuple
        Position (x, y)
    weight : float
        Edge weight
    max_weight : float
        Maximum weight for normalization
    color : str or tuple
        Edge color
    edge_width_max : float
        Maximum edge width
    """
    import matplotlib.patches as patches
    
    x, y = pos
    width = (weight / max_weight) * edge_width_max
    
    # Create a small circle as self-loop
    radius = 0.15
    circle = patches.Circle((x + radius, y), radius, fill=False, 
                            edgecolor=color, linewidth=width, alpha=0.7)
    ax.add_patch(circle)
    
    # Add small arrow
    arrow_x = x + radius + radius * 0.7
    arrow_y = y
    arrow = patches.FancyArrowPatch((arrow_x - 0.05, arrow_y), (arrow_x, arrow_y),
                                    arrowstyle='->', mutation_scale=10, 
                                    color=color, alpha=0.8)
    ax.add_patch(arrow)


def _draw_curved_arrow(
    ax, start_pos, end_pos, weight, max_weight, color, 
    edge_width_max=10, curve_strength=0.3, arrowsize=5
):
    """
    Draw curved arrows, mimicking CellChat's rotated blooming effect
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.patches import ConnectionPatch
    import matplotlib.patches as patches
    
    # Calculate arrow width
    width = (weight / max_weight) * edge_width_max
    
    # Calculate vector from start to end
    start_x, start_y = start_pos
    end_x, end_y = end_pos
    
    dx = end_x - start_x
    dy = end_y - start_y
    
    # Calculate distance and normalize
    distance = np.sqrt(dx**2 + dy**2)
    if distance == 0:
        return
    
    # Shorten the arrow to avoid overlap with nodes
    # Adjust these values based on your node sizes
    start_offset = 0.07  
    end_offset = 0.07
    
    # Calculate shortened start and end positions
    unit_dx = dx / distance
    unit_dy = dy / distance
    
    shortened_start_x = start_x + unit_dx * start_offset
    shortened_start_y = start_y + unit_dy * start_offset
    shortened_end_x = end_x - unit_dx * end_offset
    shortened_end_y = end_y - unit_dy * end_offset
    
    # Calculate midpoint and control point for curve
    mid_x = (shortened_start_x + shortened_end_x) / 2
    mid_y = (shortened_start_y + shortened_end_y) / 2
    
    # Calculate perpendicular vector (for curvature)
    shortened_distance = np.sqrt((shortened_end_x - shortened_start_x)**2 + 
                                (shortened_end_y - shortened_start_y)**2)
    
    if shortened_distance > 0:
        # Normalize perpendicular vector
        perp_x = -(shortened_end_y - shortened_start_y) / shortened_distance
        perp_y = (shortened_end_x - shortened_start_x) / shortened_distance
        
        # Add curvature offset
        curve_offset = curve_strength * shortened_distance
        control_x = mid_x + perp_x * curve_offset
        control_y = mid_y + perp_y * curve_offset
        
        # Create curved path
        from matplotlib.path import Path
        import matplotlib.patches as patches
        
        # Define Bezier curve path
        verts = [
            (shortened_start_x, shortened_start_y),  # Shortened start point
            (control_x, control_y),                   # Control point
            (shortened_end_x, shortened_end_y),      # Shortened end point
        ]
        
        codes = [
            Path.MOVETO,  # Move to start point
            Path.CURVE3,  # Quadratic Bezier curve
            Path.CURVE3,  # Quadratic Bezier curve
        ]
        
        path = Path(verts, codes)
        
        # Draw curved line
        patch = patches.PathPatch(path, facecolor='none', edgecolor=color, 
                                linewidth=width, alpha=0.85)
        ax.add_patch(patch)
        
        # Add arrow head at the shortened end position
        # Calculate arrow direction from control point to end
        arrow_dx = shortened_end_x - control_x
        arrow_dy = shortened_end_y - control_y
        arrow_length = np.sqrt(arrow_dx**2 + arrow_dy**2)
        
        if arrow_length > 0:
            # Normalize direction vector
            arrow_dx /= arrow_length
            arrow_dy /= arrow_length
            
            # Arrow head size (reduced from your original values)
            head_length = arrowsize * 0.008 
            head_width = arrowsize * 0.005 
            
            # Calculate three points of arrow head
            # Arrow tip at shortened end position
            tip_x = shortened_end_x
            tip_y = shortened_end_y
            
            # Two base points of arrow
            base_x = tip_x - arrow_dx * head_length
            base_y = tip_y - arrow_dy * head_length
            
            left_x = base_x - arrow_dy * head_width
            left_y = base_y + arrow_dx * head_width
            right_x = base_x + arrow_dy * head_width
            right_y = base_y - arrow_dx * head_width
            
            # Draw arrow head
            triangle = plt.Polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], 
                                color=color, alpha=0.85)
            ax.add_patch(triangle)


def netVisual_circle(
    matrix_df, min_threshold=None,
    edge_width_max=10, vertex_size_max=50, 
    show_labels=True, edge_color="#606060", colors=None,
    figsize=(10, 10), use_sender_colors=True,
    curve_strength=0.15, adjust_text=False,
    title="Cell-Cell Communication Network",
    edge_legend_label='Interaction\nstrength',
    n_edge_legend_levels=5,
):
    """
    # Reference: 
    https://github.com/Starlitnightly/omicverse

    Circular network visualization (similar to CellChat's circle plot)
    Uses sender cell type colors as edge gradient colors
    
    Parameters:
    -----------
    matrix_df : pd.DataFrame
        Interaction matrix (rows: sender type, columns: receiver type)
    title : str
        Plot title
    edge_width_max : float
        Maximum edge width
    vertex_size_max : float
        Maximum vertex size
    show_labels : bool
        Whether to show cell type labels
    edge_color : str
        Edge color (used when use_sender_colors=False)
    figsize : tuple
        Figure size
    use_sender_colors : bool
        Whether to use different colors for different sender cell types (default: True)
    curve_strength : float
        Strength of the curve (0 = straight, higher = more curved)
    adjust_text : bool
        Whether to use adjust_text library to prevent label overlapping (default: False)
    edge_legend_label : str
        Title for the edge width legend
    n_edge_legend_levels : int
        Number of discrete levels in the edge width legend
    """
    n_cell_types = len(matrix_df)
    cell_types = matrix_df.index.tolist()
    matrix = matrix_df.values.copy()
    
    if min_threshold is None:
        min_threshold = threshold_otsu(matrix_df.values.flatten())

    matrix[matrix < min_threshold] = 0.  # min-threshold for visualization

    # Generate colors for cell types
    if colors is None:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(cell_types))]
    palette = dict(zip(cell_types, colors))

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Create circular layout
    angles = np.linspace(0, 2*np.pi, n_cell_types, endpoint=False)
    pos = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
    
    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n_cell_types))
    
    # Add edges with weights
    max_weight = matrix.max()
    if max_weight == 0:
        max_weight = 1  # Avoid division by zero
        
    for i in range(n_cell_types):
        for j in range(n_cell_types):
            if matrix[i, j] > 0:
                G.add_edge(i, j, weight=matrix[i, j], sender_idx=i)
    
    # Compute min weight of displayed edges
    displayed_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    min_weight = min(displayed_weights) if displayed_weights else 0

    # Draw nodes
    node_sizes = matrix.sum(axis=1) + matrix.sum(axis=0)
    if node_sizes.max() > 0:
        node_sizes = (node_sizes / node_sizes.max() * vertex_size_max * 100) + 200
    else:
        node_sizes = np.full(n_cell_types, 200)
    
    # Get cell type colors for nodes
    node_colors = [palette.get(cell_types[i], '#1f77b4') for i in range(n_cell_types)]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                           node_color=node_colors, 
                           ax=ax, alpha=0.8, edgecolors='black', linewidths=1)
    
    # Draw edges with curved arrows
    if use_sender_colors:
        # Group edges by sender
        edges_by_sender = {}
        for u, v, data in G.edges(data=True):
            sender_idx = data['sender_idx']
            if sender_idx not in edges_by_sender:
                edges_by_sender[sender_idx] = []
            edges_by_sender[sender_idx].append((u, v, data['weight']))
        
        # Draw curved edges for each sender with its specific color
        for sender_idx, edges in edges_by_sender.items():
            sender_cell_type = cell_types[sender_idx]
            sender_color = palette.get(sender_cell_type, '#1f77b4')
            
            for u, v, weight in edges:
                start_pos = pos[u]
                end_pos = pos[v]
                # Handle self-loops
                if u == v:
                    _draw_self_loop(ax, start_pos, weight, max_weight, 
                                        sender_color, edge_width_max)
                else:
                    _draw_curved_arrow(ax, start_pos, end_pos, weight, max_weight, 
                                       sender_color, edge_width_max, curve_strength)
    else:
        # Use traditional single colormap
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            start_pos = pos[u]
            end_pos = pos[v]
            
            if u == v:
                _draw_self_loop(ax, start_pos, weight, max_weight, 
                                    edge_color, edge_width_max)
            else:
                _draw_curved_arrow(ax, start_pos, end_pos, weight, max_weight, 
                                   edge_color, edge_width_max, curve_strength)
    
    # Add labels
    if show_labels:
        label_pos = {i: (1.2*np.cos(angle), 1.2*np.sin(angle)) 
                    for i, angle in enumerate(angles)}
        labels = {i: cell_types[i] for i in range(n_cell_types)}
        
        if adjust_text:
            try:
                from adjustText import adjust_text
                
                texts = []
                for i in range(n_cell_types):
                    x, y = label_pos[i]
                    text = ax.text(
                        x, y, cell_types[i], 
                        fontsize=16, ha='center', va='center',
                    )
                    texts.append(text)
                
                # Adjust text positions to avoid overlapping
                adjust_text(texts, ax=ax,
                            expand_points=(1.2, 1.2),
                            expand_text=(1.2, 1.2),
                            force_points=0.5,
                            force_text=0.5,
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7, lw=0.5))
                
            except ImportError:
                import warnings
                warnings.warn("adjustText library not found. Using default nx.draw_networkx_labels instead.")
                nx.draw_networkx_labels(
                    G, label_pos, labels, font_size=16, ax=ax, 
                    font_family=rcParams['font.family'], font_weight='bold'
                )
        else:
            # Use traditional networkx labels
            nx.draw_networkx_labels(
                G, label_pos, labels, font_size=16, ax=ax, 
                font_family=rcParams['font.family'], font_weight='bold'
            )
    
    ax.set_title(title, fontsize=30, y=0.9, pad=20, fontweight='bold')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    
    # Add legend for node colors (cell types)
    legend_elements = []
    for i, cell_type in enumerate(cell_types):
        color = palette.get(cell_type, '#1f77b4')
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                           edgecolor='black', linewidth=0.5,
                                           label=cell_type))
    ncol = min(5, len(legend_elements))
    legend1 = ax.legend(
        handles=legend_elements, loc='lower center', 
        bbox_to_anchor=(0.5, 0.01), ncol=ncol, fontsize=15,
        frameon=True, fancybox=True, shadow=True
    )
    ax.add_artist(legend1)
    
    # Add legend fo edge widths
    # Compute discrete weight levels evenly spaced between min and max
    weight_levels = np.linspace(min_weight, max_weight, n_edge_legend_levels)
    # Remove duplicates and sort
    weight_levels = sorted(set(np.round(weight_levels, 2)))

    size_legend_elements = []
    for w in weight_levels:
        # Map weight to line width (same mapping as edges)
        lw = (w / max_weight) * edge_width_max
        # Scale marker size proportionally to line width
        ms = max(lw * 2.5, 2)
        size_legend_elements.append(
            plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor='grey', markeredgecolor='black',
                markeredgewidth=0.5,
                markersize=ms,
                label=f'{w:.2f}',
                linestyle='None'
            )
        )

    ax.legend(
        handles=size_legend_elements,
        loc='center right',
        bbox_to_anchor=(1.1, 0.5),
        fontsize=18,
        title=edge_legend_label,
        title_fontsize=18,
        frameon=True,
        fancybox=True,
        shadow=True,
        labelspacing=1.5,
        handletextpad=1.0,
    )    
    fig.tight_layout()
    return fig, ax

