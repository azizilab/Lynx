import numpy as np
import pandas as pd
import igraph
import scanpy as sc
import scFates as scf

from scanpy.tools._dpt import DPT
from collections import OrderedDict
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.interpolate import make_interp_spline

from utils import to_dense_array


def get_diffusion_dist(repr, root_repr, k=30):
    r"""Compute diffusion distance against the `root node`"""
    # Append "dummy" principal node
    n_features = repr.shape[-1]
    assert n_features == root_repr.shape[-1], \
        "latent repr & diffusion root repr have different dimensions"
    
    adata = sc.AnnData(np.vstack([repr, root_repr]))
    sc.pp.neighbors(adata, n_neighbors=k)
    
    dpt = DPT(adata)
    dpt.compute_transitions()
    dpt.compute_eigen(n_comps=n_features-1)
    
    dpt.iroot = adata.shape[0] - 1
    dpt._set_pseudotime()

    # Drop the "dummy" principal node
    return dpt.pseudotime[:-1]  


def get_knn_dist(repr, root_repr, k=30):
    r"""Compute kNN graph shortest-path length against the `root_node`"""
    # Append "dummy" principal node
    n_features = repr.shape[-1]
    assert n_features == len(root_repr), \
    "latent repr & diffusion root repr have different dimensions"

    from sklearn.utils.graph import single_source_shortest_path_length

    adata = sc.AnnData(np.vstack([repr, root_repr]))
    sc.pp.neighbors(adata, n_neighbors=k)
    knn_graph = adata.obsp['connectivities']
    dist_dict = single_source_shortest_path_length(knn_graph, adata.shape[0]-1)
    dist_dict = OrderedDict(sorted(dist_dict.items()))

    return [v for _, v in dist_dict.items()][:-1]


def sort_pnodes(adata):
    r"""
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


def dist_to_pnode(
    adata: sc.AnnData, 
    dist_metric: str = 'euclidean',
    use_rep: str = None,
    k: int = 30,
    verbose: str = False
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute distance(x, y) btw each cell (x) and 
    principal node (y) in latent space (Z \in R^K)
    """ 
    assert 'graph' in adata.uns.keys(), "Please run Principal Curve first"
    repr = adata.X if use_rep is None else adata.obsm[use_rep]
    
    # Get latent representations of principal nodes
    pcurve_repr = adata.uns['graph']['F'].T  # dim:[n_nodes, n_latent (K)]
    n_pts = adata.shape[0]
    n_nodes = pcurve_repr.shape[0]
    dists = np.zeros((n_pts, n_nodes), dtype=np.float32)

    # Compute trajectory ordering of principal nodes
    node_indices = sort_pnodes(adata)
    if verbose:
        print('Principal Node ordering:', node_indices)
    pcurve_repr = pcurve_repr[node_indices, :]

    for i, node in enumerate(pcurve_repr):
        if dist_metric == 'euclidean':
            dists[:, i] = cdist(repr, np.expand_dims(node, 0)).squeeze() 
        elif dist_metric == 'diffusion':
            dists[:, i] = get_diffusion_dist(repr=repr, root_repr=node, k=k)
        elif dist_metric == 'knn':
            dists[:, i] = get_knn_dist(repr=repr, root_repr=node, k=k)
        else:  
            raise NotImplementedError(dist_metric)

    indices = dists.argmin(1)
    return dists, indices


def dist_to_pcurve(
    adata: sc.AnnData,
    principal_curve,
    use_rep=None,
    dist_metric='euclidean'
):
    r"""
    Compute distance (x, y) btw each cell (x) and its closest
    principal curve point (y) in latent representation space
    """
    repr = adata.X if use_rep is None else adata.obsm[use_rep]

    if dist_metric == 'euclidean':
        tree = KDTree(principal_curve)
        dists, indices = tree.query(repr)
    elif dist_metric == 'diffusion':
        dists = cdist(repr, principal_curve, metric=get_diffusion_dist)
        indices = dists.argmin(1)
    elif dist_metric == 'knn':
        dists = cdist(repr, principal_curve, metric=get_knn_dist)
        indices = dists.argmin(1)

    # Normalize indices as pseudotime assignment
    indices = np.array(indices)
    indices = (indices-indices.min()) / (indices.max()-indices.min())
    return dists, indices


def compute_trajectory(
    adata: sc.AnnData, 
    root: str = None, 
    tip: str = None,
    root_marker: str = None,
    use_rep: str = None,
    n_nodes: int = 20,
    n_neighbors: int = 100,
    dist_metric: str = 'euclidean',
    degree: int = 0,
    n_points: int = 100,
    verbose=False,
):
    r"""
    Compute smooth trajectory \gamma(t) \in [0, 1] based on the distance to 
    the sorted principal nodes; Optional marker-based supervision to rotate 
    the direction of \gamma(t) origin.  

    Parameters
    ----------
    adata : sc.AnnData
        AnnData of latent representation w/ computed elastic principal graph
    root : str
        Annotation of the root feature
    tip : str
        Annotation of the tip feature
    root_marker : str
        Optional marker close to 'root' to rotate (+/-) of the trajectory
    use_rep : str
        Use the indicated representation. 'X' or any key for .obsm is valid. 
        If None, the representation is chosen automatically
    n_nodes : int
        # principal nodes to infer 
        Increase `n_nodes` get more localized principal manifold
    n_neighbors : int
        # nearest neighbors for graph-based distance metrics
    dist_metric : str
        Distance metric to fit D(z_i, principal_curve)
    degree : int
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
    assert use_rep in adata.obsm.keys(), \
        "Please run the model to obtain latent representation `z` first"
    
    if n_nodes is None:
        n_nodes = adata.obsm[use_rep].shape[-1]

    t_discrete = None 
    scf.tl.curve(
        adata,
        use_rep=use_rep,
        Nodes=n_nodes,
        epg_extend_leaves=True,
        ndims_rep=adata.obsm[use_rep].shape[-1],
        epg_mu=.05,
        epg_lambda=.1
    )

    if root is not None and tip is not None:
        # Supervised: compute diffusion dist. to `root`` & `tip`
        assert root in adata.var_names and tip in adata.var_names, \
            "Either `root` {0} or `tip` {1} annotation doesnt' exist".format(root, tip)
        
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm)
        sc.pp.log1p(adata_norm)

        root_idx = to_dense_array(adata_norm[:, root].X).argmax()
        tip_idx = to_dense_array(adata_norm[:, tip].X).argmax()

        rep = adata.obsm[use_rep]
        root_rep = rep[root_idx]
        tip_rep = rep[tip_idx]

        distances = np.array([
            get_diffusion_dist(rep, root_rep, k=n_neighbors),
            get_diffusion_dist(rep, tip_rep, k=n_neighbors)
        ]).T

    else:
        # Unsupervised: compute distances to manifold "roots"
        distances, t_discrete = dist_to_pnode(
            adata,
            use_rep=use_rep,
            dist_metric=dist_metric, 
            k=n_neighbors,
            verbose=verbose
        )
        
    if degree == 0:
        t = distances[:, 0] / (distances[:, 0]+distances[:, -1])
    else:
        # Interpolation
        principal_repr = adata.uns['graph']['F'].T
        principal_repr = principal_repr[adata.uns['graph']['pnode_indices']]
        
        x = np.arange(len(principal_repr))
        cs = make_interp_spline(x, principal_repr, k=degree)
        
        xs = np.linspace(x[0], x[-1], n_points)
        interpolants = cs(xs)
        _, t = dist_to_pcurve(
            adata, interpolants, 
            use_rep=use_rep,
            dist_metric=dist_metric
        )
  
    adata.obs['t'] = t
    adata.obs['seg'] = '1'
    adata.obs['seg'] = adata.obs['seg'].astype('category')  

    # [Optional]: Rotate trajectory direction based on marker
    if root_marker is not None:
        assert root_marker in adata.var_names, "Nonexisted marker {}".format(root_marker)
        
        root_expr = to_dense_array(
            adata[np.argsort(adata.obs['t'])[::-1][:n_neighbors], root_marker].X
        ).mean()
        tip_expr = to_dense_array(
            adata[np.argsort(adata.obs['t'])[:n_neighbors], root_marker].X
        ).mean()
        
        # Rotate axis s.t. `root_marker` enriched end --> 0
        if root_expr > tip_expr:
            adata.obs['t'] = 1-t

    return None
