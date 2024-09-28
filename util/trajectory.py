import numpy as np
import pandas as pd
import igraph
import scanpy as sc
import scFates as scf

from scanpy.tools._dpt import DPT
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.interpolate import make_interp_spline


def get_pcurve_path(adata):
    """
    Compute trajectory ordering indices btw principal nodes in latent space
    """
    assert 'graph' in adata.uns.keys(), "Please run Principal Curve first"
    al = np.array(
        igraph.Graph.Adjacency(
            (adata.uns['graph']['B'] > 0).tolist(), 
            mode='undirected'
        ).get_edgelist()
    )

    root_node, term_node = adata.uns['graph']['tips']
    curr_node = root_node
    ypos, xpos = np.asarray(np.where(al == root_node)).T[0]  # coords of current node
    
    path = [curr_node]
    while term_node not in path:
        xpos = 1 - xpos  # adj. principle node
        curr_node = al[ypos, xpos]
        path.append(curr_node)
        if curr_node == term_node:
            break
        
        # Update `ypos`, `xpos` of `curr_node`:
        # each node except root & terminal appears twice 
        coords = np.asarray(np.where(al == curr_node)).T  # dim: [2, 2]
        if np.array_equal(coords[0], [ypos, xpos]):
            ypos, xpos = coords[1]
        else:
            ypos, xpos = coords[0]

    adata.uns['graph']['pnode_indices'] = path
    return path


def get_diffusion_dist(adata, root_expr, n_neighbors=50):
    """
    Compute diffusion distance against `root node`
    """
    # Append "dummy" principal node
    adata_root = sc.AnnData(np.expand_dims(root_expr, 0))
    adata_cat = sc.concat((adata, adata_root), axis=0)

    sc.pp.neighbors(adata_cat, n_neighbors=n_neighbors)
    
    dpt = DPT(adata_cat)
    dpt.compute_transitions()
    dpt.compute_eigen(n_comps=adata_cat.shape[1])
    dpt.iroot = adata_cat.shape[0] - 1
    dpt._set_pseudotime()
    
    return dpt.pseudotime[:-1]  # Drop "dummy" principal node


def get_geodesic_dist(pt1, pt2):
    """
    Compute geodesic distance along hypersphere
    """
    u = pt1.astype(np.float32)
    v = pt2.astype(np.float32)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    
    dot_product = np.dot(u, v).clip(-1.0, 1.0)
    return np.arccos(dot_product)


def dist_to_pnode(
    adata, 
    dist_metric='euclidean',
    use_rep=None,
    verbose=False
):
    """
    Compute distance(x, y) btw each cell (x) and 
    principal node (y) in latent space (Z \in R^K)
    """ 
    assert 'graph' in adata.uns.keys(), "Please run Principal Curve first"
    repr = adata.X if use_rep is None else adata.obsm[use_rep]
    
    pcurve_nodes = adata.uns['graph']['F'].T  # dim:[n_nodes, n_latent (K)]
    n_pts = adata.shape[0]
    n_nodes = pcurve_nodes.shape[0]
    dists = np.zeros((n_pts, n_nodes), dtype=np.float32)

    # Compute trajectory ordering of principal nodes
    pcurve_indices = get_pcurve_path(adata)
    if verbose:
        print('Principal Node ordering:', pcurve_indices)
    pcurve_nodes = pcurve_nodes[pcurve_indices, :]

    for i, node in enumerate(pcurve_nodes):
        if dist_metric == 'euclidean':
            dists[:, i] = cdist(repr, np.expand_dims(node, 0)).squeeze() 
        else:
            dists[:, i] = np.array([
                get_geodesic_dist(z, node) 
                for z in repr
            ])    
    indices = dists.argmin(1)

    return dists, indices


def dist_to_pcurve(
    adata,
    principal_curve,
    use_rep=None,
    dist_metric='euclidean'
):
    """
    Compute distance (x, y) btw each cell (x) and its closest
    principal curve point (y) in latent representation space
    """
    repr = adata.X if use_rep is None else adata.obsm[use_rep]

    if dist_metric == 'euclidean':
        tree = KDTree(principal_curve)
        dists, indices = tree.query(repr)
    else:
        dists = cdist(repr, principal_curve, metric=get_geodesic_dist)
        indices = dists.argmin(1)

    # Normalize indices as pseudotime assignment
    indices = np.array(indices)
    indices = (indices-indices.min()) / (indices.max()-indices.min())
    
    return dists, indices


def compute_trajectory(
    adata, 
    use_rep=None,
    dist_metric='euclidean',
    k=1,
    n_points=100,
    verbose=False,
):
    """
    Compute smooth trajectory \in [0, 1] based on 
    distance to the sorted principal nodes

    Parameters
    ----------
    adata : sc.AnnData
        AnnData of latent representation w/ computed elastic principal graph

    use_rep : str
        Use the indicated representation. 'X' or any key for .obsm is valid. 
        If None, the representation is chosen automatically
        
    dist_metric : str
        Distance metric to fit D(z_i, principal_curve)

    k : int
        degree of interpolation for principal curve

    n_points : int
        # points for discrete approx. of the principal curve
        
    Returns
    -------
    None. 
        Sets the following fields:
        
        `adata.obs['t']` : np.ndarray
            Smooth gradient / pseudotime
        `adata.obs['t_discrete]` : np.ndarray
            Discrete "zonation assignment" to the closest principal nodes
    """
    # TODO: [DEBUG], consider expression "velocity"/"density" ==> non-uniform \delta t

    distances, t_discrete = dist_to_pnode(
        adata,
        use_rep=use_rep,
        dist_metric=dist_metric, 
        verbose=verbose
    )
    
    principal_nodes = adata.uns['graph']['F'].T
    principal_nodes = principal_nodes[adata.uns['graph']['pnode_indices']]
    
    if k == 0:
        # Weighted distance to trajectory edges 
        # TODO: pseudotime ordering vs. actual pseudotime
        t = distances[:, 0] / (distances[:, 0]+distances[:, -1])
        # t_order = np.argsort(np.argsort(t))
        # t_order = (t_order-t_order.min()) / (t_order.max()-t_order.min())

    else:
        # Interpolation
        x = np.arange(len(principal_nodes))
        cs = make_interp_spline(x, principal_nodes, k=k)
        
        xs = np.linspace(x[0], x[-1], n_points)
        interpolants = cs(xs)
        _, t = dist_to_pcurve(
            adata, interpolants, 
            use_rep=use_rep,
            dist_metric=dist_metric
        )

    adata.obs['t'] = t
    adata.obs['milestones'] = t_discrete
    adata.obs['milestones'] = adata.obs['milestones'].astype('category')
   
    adata.obs['seg'] = '1'
    adata.obs['seg'] = adata.obs['seg'].astype('category')

    adata.uns["graph"]["milestones"] = dict(
        zip(
            adata.obs.milestones.cat.categories,
            adata.obs.milestones.cat.categories.astype(int),
        )
    )
    
    return None