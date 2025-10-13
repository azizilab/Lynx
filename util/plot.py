import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import squidpy as sq
import seaborn as sns
import holoviews as hv
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from scipy.special import comb
from typing import Dict, List
from matplotlib.axes import Axes

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import get_binned_expr

hv.extension('bokeh')


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
        '.-', color='gray', lw=.5, ms=2, mfc='yellow'
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
    x_true: np.ndarray, 
    x_pred: np.ndarray, 
    indices: List[int] = None,
    logscale=True,
    subset_ratio : float = 0.01,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None
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
    text_xloc = np.quantile(x_true, .01)
    text_yloc = np.quantile(x_pred, .99)
    
    ax.scatter(x_true[indices], x_pred[indices], s=.2, c=density, cmap='turbo')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.annotate(r"$PearsonR$ = {:.3f}".format(
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

# -----------------------------------
# Visualize cell-cell interactions
# -----------------------------------

def summarize_cell_interaction(
    adata,  
    ccc_rep='omega', 
    cluster_key='cell_type', 
    cluster_labels=None,
    title='', 
    show_fig=False
):
    r"""Compute cluster-wise summary of cell-cell interactions"""
    if cluster_labels is None:
        cluster_labels = adata.obs[cluster_key].cat.categories
    per_idx_labels = adata.obs['cell_type'].values
    n_clusters = len(cluster_labels)
    mat = np.zeros((n_clusters, n_clusters), dtype=np.float32)

    # Aggregate: for each receiver type, average over its cells
    for i, rtype in enumerate(cluster_labels):
        mask = (per_idx_labels == rtype)
        if mask.sum() > 0:
            mat[i] = adata.obsm[ccc_rep][mask].mean(axis=0)   # sender cell types

    # add omega as an extra sender column
    df = pd.DataFrame(
        mat,
        index=cluster_labels, 
        columns=list(cluster_labels)
    )

    # plot heatmap
    if show_fig:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, cmap="magma", linecolor='gray', linewidth=0.5)
        plt.xlabel("Sender", fontsize=10)
        plt.ylabel("Receiver", fontsize=10)
        plt.title(title, fontsize=20)
        plt.show()

    return df


def interactive_cell_interaction(attn_df, amplitude=1):
    assert np.array_equal(attn_df.index, attn_df.columns)
    attn_score = attn_df.values
    cell_types = attn_df.columns

    graph = hv.Graph([
        (cell_types[i], cell_types[j], attn_score[i, j])
        for i in range(len(cell_types)-1) for j in range(i+1, len(cell_types))
    ], vdims=['weight'])
    labels = hv.Labels(graph.nodes, ['x', 'y'], 'index')

    graph = graph.opts(
        node_color='index', edge_color=hv.dim('weight')*amplitude, cmap='Category10',
        edge_cmap='Reds', edge_line_width=hv.dim('weight')*amplitude,
    )
    graph = (graph * labels.opts(text_font_size='10pt', text_color='black'))

    return graph



# Visualize spatial microenvironment of a few cells
def disp_spatial_interaction(
    adata, 
    cluster_key='cell_type', 
    target_idx=None, 
    figsize=(10, 8),
    return_subgraph=False
):
    """Visualize spatial cell-cell interaction weights for a target cell"""
    
    # Sample random target if not provided
    if target_idx is None:
        target_idx = np.random.choice(adata.shape[0])
    
    # Extract edge information
    edge_index = adata.uns['edge_index']
    omega = adata.uns['omega']
    
    # Find edges pointing to target
    target_mask = edge_index[1] == target_idx
    source_indices = edge_index[0][target_mask]
    edge_weights = omega[target_mask]
    
    # Create subgraph with target and its sources
    all_nodes = np.concatenate([source_indices, [target_idx]])
    spatial_coords = adata.obsm['spatial'][all_nodes]
    cell_types = adata.obs[cluster_key].iloc[all_nodes]
    
    # Create NetworkX graph
    G = nx.Graph()
    pos = {}
    
    # Add nodes with positions
    for i, node_idx in enumerate(all_nodes):
        G.add_node(node_idx)
        pos[node_idx] = spatial_coords[i]
    
    # Add edges with weights
    for i, (source_idx, weight) in enumerate(zip(source_indices, edge_weights)):
        G.add_edge(source_idx, target_idx, weight=weight)
    
    plt.figure(figsize=figsize)
    
    # Node colors by cell type
    unique_types = cell_types.unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    type_to_color = dict(zip(unique_types, colors))
    node_colors = [type_to_color[cell_types.iloc[i]] for i in range(len(all_nodes))]
    
    # Edge colors and widths by omega weights
    edge_colors = plt.cm.Purples(edge_weights / edge_weights.max())
    edge_widths = edge_weights * 10  # Fixed multiplier for edge width
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    
    # Highlight target node with black border
    target_color = type_to_color[cell_types.iloc[-1]]  # target is last in all_nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[target_idx], node_color=target_color, 
                          node_size=200, alpha=1.0, edgecolors='black', linewidths=2)
    
    # Add colorbar for omega values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Purples, 
                              norm=plt.Normalize(vmin=0, vmax=edge_weights.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6, aspect=20)
    cbar.set_label('Omega (Interaction Weight)', rotation=270, labelpad=15)
    
    # Add legend for cell types (max 6 columns per row)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=type_to_color[ct], markersize=8, label=ct)
                      for ct in unique_types]
    
    ncol = min(6, len(unique_types))
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
              ncol=ncol, frameon=False, fontsize=10)
    
    plt.title(f'Spatial Interaction Network\nTarget Cell: {target_idx}')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if return_subgraph:
        return {
            'source': source_indices,
            'target': target_idx,
            'omega': edge_weights
        }
    else:
        return None
    
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
    matrix_df, title="Cell-Cell Communication Network", 
    edge_width_max=10, vertex_size_max=50, show_labels=True,
    cmap='Blues', edge_color="#606060", palette=None,
    figsize=(10, 10), use_sender_colors=True,
    use_curved_arrows=True, 
    curve_strength=0.3, adjust_text=False
):
    """
    # Reference: 
    https://github.com/Starlitnightly/omicverse

    Circular network visualization (similar to CellChat's circle plot)
    Uses sender cell type colors as edge gradient colors
    
    Parameters:
    -----------
    matrix_df : pd.DataFrame
        Interaction matrix (count or weight)
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
    use_curved_arrows : bool
        Whether to use curved arrows like CellChat (default: True)
    curve_strength : float
        Strength of the curve (0 = straight, higher = more curved)
    adjust_text : bool
        Whether to use adjust_text library to prevent label overlapping (default: False)
        If True, uses plt.text instead of nx.draw_networkx_labels
    """
    n_cell_types = len(matrix_df)
    cell_types = matrix_df.index.tolist()
    matrix = matrix_df.values

    # Generate colors for cell types
    if palette is None:
        # Use matplotlib's default color cycle for discrete categories
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        
        # Repeat colors if we have more cell types than default colors
        colors = [default_colors[i % len(default_colors)] for i in range(len(cell_types))]
        palette = dict(zip(cell_types, colors))

    fig, ax = plt.subplots(figsize=figsize)
    
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
    
    ax.set_title(title, fontsize=24, y=0.9, pad=20, fontweight='bold')
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
        handles=legend_elements, loc='upper center', 
        bbox_to_anchor=(0.5, 0.1), ncol=ncol, fontsize=15,
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

