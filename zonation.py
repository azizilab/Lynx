import numpy as np
import networkx as nx

from scipy.sparse import linalg
from skimage.segmentation import find_boundaries


def create_graph(roi):
    """
    Convert combinatory graph (w/ 4-connected compoments adj(i) = {(i, j+1), (i, j-1), (i+1, j), (i-1, j)})
    from corresponding ROI pixels from the input image

    Code Referenced from: https://stackoverflow.com/questions/63653267/how-to-create-a-graph-with-an-images-pixel
    """
    # Horizontal edge
    hx, hy =  np.nonzero(np.logical_and(roi[1:] == 1, roi[:-1] == 1)) #horizontal edge start positions
    h_units = np.array([hx, hy]).T
    h_starts = [tuple(n) for n in h_units]
    h_ends = [tuple(n) for n in h_units + (1, 0)] #end positions = start positions shifted by vector (1,0)
    horizontal_edges = zip(h_starts, h_ends)

    # Vertical edge
    vx, vy = np.nonzero(np.logical_and(roi[:,1:] == 1,  roi[:,:-1] == 1)) #vertical edge start positions
    v_units = np.array([vx, vy]).T
    v_starts = [tuple(n) for n in v_units]
    v_ends = [tuple(n) for n in v_units + (0, 1)] #end positions = start positions shifted by vector (0,1)
    vertical_edges = zip(v_starts, v_ends)

    # Create graph
    G = nx.Graph()
    G.add_edges_from(horizontal_edges)
    G.add_edges_from(vertical_edges)

    return G


def add_graph_props(G, cv_nodes, pv_nodes):
    """
    Assign init. temp. & ROI boundary as graph properties
    """
    for n in G:
        if n in cv_nodes:
            G.nodes[n]['t'] = 1
            G.nodes[n]['bound'] = True
        elif n in pv_nodes:
            G.nodes[n]['t'] = -1
            G.nodes[n]['bound'] = True
        else:
            G.nodes[n]['t'] = 0
            if G.degree[n] < 4:
                G.nodes[n]['bound'] = True
    return None


def compute_interior_temp(G, debug=False):
    """
    Compute temperature of "interior" nodes based on Harmonic interpolation solution (Grady & Schwartz, 2003)
    """
    # Constructed permuted Laplacian Matrix L => {L_b, L_i, R, R^T}
    bound_nodes = [
        n for n, v in G.nodes.items()
        if 'bound' in v
    ]

    interior_nodes = [
        n for n, v in G.nodes.items()
        if 'bound' not in v
    ]

    n_bound = len(bound_nodes)
    perm_node_orders = bound_nodes + interior_nodes
    if debug:
        assert len(G) == len(perm_node_orders)

    L = nx.laplacian_matrix(G, nodelist=perm_node_orders)
    L_i = L[n_bound:, n_bound:]
    R = L[:n_bound, n_bound:]

    # Validate permuted nodes' in-degree have the correct order [d(bound), d(interior)]
    if debug:
        diag = np.diag(L)
        for i, n in enumerate(perm_node_orders):
            assert G.degree[n] == diag[i]

    # Compute interior temperature u(i) from L & u(b): 
    # Sol:  L_i @ u(i) = -R^T @ u(b)
    u_b = np.asarray([G.nodes[n]['t'] for n in bound_nodes])
    u_i = linalg.cg(A=L_i, b=-R.T@u_b) 

    if isinstance(u_i, tuple):
        u_i = u_i[0]
    
    return u_i, tuple(np.array(interior_nodes).T)  # 2 x N tuple


def assign_diffusion_temp(
    u_i, 
    interior_coords,
    cv_coords,
    pv_coords, 
    shape
):
    """
    Assign steady-state sol. of the diffused pixel values back to the image
    """
    assert len(interior_coords[0]) == len(u_i), 'Different coords & temperature lengths'
    u = np.zeros(shape, dtype=np.float64)
    u[interior_coords] = u_i
    u[cv_coords] = 1
    u[pv_coords] = -1
    return u


def discretize_temp(u, roi, cv_coords, n_layers=10, return_border=False, verbose=False):
    """
    Create discretized 1-indexed bins (1,2,...,n) as the zonation priors 
    from diffused gradient temperature `u`, keep CV & PV regions off from 
    `roi` as the min (PV) / max (CV) zones
    """
    assert n_layers > 3, "Invalid `n_layers`, please assign # lobule layers > 3"
    
    coords = np.nonzero(roi)
    coords_to_rm = np.nonzero(1-roi)
    qs = np.quantile(u[coords], np.linspace(0, 1, n_layers-1))

    if verbose:
        print('Quantile:', qs)
        
    zone = np.zeros_like(u, dtype=np.int32)
    for i, q in enumerate(qs[:-1]):
        zone[u >= q] = i+1
    zone[coords_to_rm] = 0

    zone[cv_coords] = zone.max() + 1
    zone += 1

    # Assign 1-pixel width border btw zones
    if return_border:
        border = find_boundaries(zone)
        zone[border] = 0

    return zone
