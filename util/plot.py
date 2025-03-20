import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import squidpy as sq
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from scipy.special import comb
from typing import Dict, List
from matplotlib.axes import Axes

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import get_binned_expr


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


def disp_chans(img, title=None, ncols=4, cmap='magma'):
    r"""Display single-channel aligned images"""
    depth = len(img)
    nrows = depth // ncols if depth % ncols == 0 else depth // ncols + 1
    
    idx = 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3.2*nrows))
    for r in range(nrows):
        for c in range(ncols):
            if idx >= depth:
                axes[r, c].axis('off')
                continue
            axes[r, c].imshow(img[idx], cmap=cmap)
            idx += 1
            
    fig.tight_layout()
    fig.suptitle(title, y=1.01)
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

    principal_repr = adata.uns['graph']['F'].T[
        adata.uns['graph']['pnode_indices']
    ]
    n_nodes = principal_repr.shape[0]
    adata_repr = sc.AnnData(
        np.vstack([adata.obsm[use_rep], principal_repr])
    )
    sc.pp.neighbors(adata_repr)
    sc.tl.umap(adata_repr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        adata_repr.obsm['X_umap'][:-n_nodes, 0],
        adata_repr.obsm['X_umap'][:-n_nodes, 1],
        c=adata.obs['t'], s=0.1, edgecolors=None, cmap=cmap
    )
    ax.plot(
        adata_repr.obsm['X_umap'][-n_nodes:, 0],
        adata_repr.obsm['X_umap'][-n_nodes:, 1],
        '.-', color='gray', lw=1, ms=10, mfc='yellow'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('UMAP1', fontsize=8)
    ax.set_ylabel('UMAP2', fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.show()


def disp_fitted_expr(
    expr_df, 
    n_bins=500,
    figsize=(5, 8),
    display=False,
    return_expr=False,
    savedir=None,
):
    """
    Display interpolated cell / pixel expressions along the trajectory
    """
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Norm per-feature expressions, take avg. pooling into K bins
    binned_expr_df = get_binned_expr(
        expr_df.T,
        n_bins=n_bins
    )

    if display:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(binned_expr_df, ax=ax, cmap='RdBu_r')
        fig.show()

    heatmap = go.Heatmap(
        z=binned_expr_df.values,
        y=binned_expr_df.index,
        colorscale='RdBu_r'
    )
    fig = go.Figure(data=heatmap)
    fig.update_layout(
        height=700, width=500,
        xaxis=dict(title='PV-CV bins'),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white',
    )
        
    if savedir is not None:
        fig.write_html(savedir+'.html')
    if return_expr:
        return binned_expr_df
    else:
        return fig
    

def disp_celltype_dynamics(dynamics_df, ncols=4, savedir=None):
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
            
            axes[row, col].scatter(x, dynamics_df.iloc[:, idx], 
                               s=1, c='k', alpha=0.5)
            axes[row, col].plot(xx, yy, color='b', linewidth=2, alpha=0.2)
            axes[row, col].set_title(dynamics_df.columns[idx], fontsize=12)
            axes[row, col].set_xlabel('PV -> CV\n (sliding windows)', fontsize=10)
            axes[row, col].set_ylabel('Proportions')
            axes[row, col].spines[['right', 'top']].set_visible(False)
            idx += 1
    
    
    plt.tight_layout()
    plt.show()

    if savedir:
        fig.savefig(savedir, bbox_inches="tight", dpi=300)


def disp_kde_scatter(
    x_true, 
    x_pred, 
    xlabel=None,
    ylabel=None,
    title=None
):
    r"""Reconstruction plot w/ density"""
    v_stacked = np.vstack([x_true, x_pred])
    density = gaussian_kde(v_stacked)(v_stacked)


    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    text_xloc = np.quantile(x_true, .05)
    text_yloc = np.quantile(x_pred, .95)
    
    ax.scatter(x_true, x_pred, s=.2, c=density, cmap='turbo')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.annotate(r"$r$ = {:.3f}".format(
        pearsonr(x_true, x_pred)[0]), (text_xloc, text_yloc), fontsize=12
    )

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

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
