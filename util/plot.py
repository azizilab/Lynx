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
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def disp_fitted_expr(
    expr_df, 
    n_bins=500,
    figsize=(5, 8),
    show_pot=False,
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

    if show_plot:
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
    ax.annotate(r"$PearsonR$ = {:.3f}".format(
        pearsonr(x_true, x_pred)[0]), (text_xloc, text_yloc), fontsize=12
    )

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
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
    np.fill_diagonal(df.values, 0)

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
    matrix_df, min_threshold=0,
    edge_width_max=10, vertex_size_max=50, show_labels=True,
    edge_color="#606060", palette=None,
    figsize=(10, 10), use_sender_colors=True,
    curve_strength=0.15, adjust_text=False,
    title="Cell-Cell Communication Network", 
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
    cmap : str
        Colormap for edges (used when use_sender_colors=False)
    figsize : tuple
        Figure size
    use_sender_colors : bool
        Whether to use different colors for different sender cell types (default: True)
    curve_strength : float
        Strength of the curve (0 = straight, higher = more curved)
    adjust_text : bool
        Whether to use adjust_text library to prevent label overlapping (default: False)
        If True, uses plt.text instead of nx.draw_networkx_labels
    """
    n_cell_types = len(matrix_df)
    cell_types = matrix_df.index.tolist()
    matrix = matrix_df.values.copy()
    matrix[matrix < min_threshold] = 0.  # min-threshold for visualization

    # Generate colors for cell types
    if palette is None:
        # Use matplotlib's default color cycle for discrete categories
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        
        # Repeat colors if we have more cell types than default colors
        colors = [default_colors[i % len(default_colors)] for i in range(len(cell_types))]
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
                    # Draw self-loop
                    _draw_self_loop(ax, start_pos, weight, max_weight, 
                                        sender_color, edge_width_max)
                else:
                    # Draw curved arrow
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
            # Use plt.text with adjust_text to prevent overlapping
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
                warnings.warn("adjustText library not found. Using default nx.+networkx_labels instead.")
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
    # TODO: two legends not showing simultaneously
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
        
    if not use_sender_colors:
        edges = list(G.edges())
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            # Create legend with dots of different sizes representing edge widths
            legend_elements = []
            weight_levels = [np.percentile(weights, p) for p in [20, 40, 60, 80, 100]]
            weight_levels = sorted(list(set(weight_levels)))  # Remove duplicates and sort
            
            for weight in weight_levels:
                width = (weight / max_weight) * edge_width_max
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=edge_color, 
                                markersize=width*2, 
                                label=f'{weight:.1f}',
                                linestyle='None'))
            
            ax.legend(handles=legend_elements, loc='lower center', 
                     bbox_to_anchor=(1.1, 0.4), fontsize=15,
                     title='Strength', title_fontsize=18)
    
    fig.tight_layout()
    return fig, ax

