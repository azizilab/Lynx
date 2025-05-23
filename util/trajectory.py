import numpy as np
import pandas as pd
import igraph
import scanpy as sc
import scFates as scf

from scanpy.tools._dpt import DPT
from collections import OrderedDict, defaultdict, deque
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.interpolate import make_interp_spline
from utils import to_dense_array
from typing import List


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


def prune_branches(edge_list: np.ndarray, tips: List[int]):
    """
    Given an acyclic undirected graph (tree) with edge_list representation
    Prune branches and return the path connected by tips with the longest distance
    """
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)

    def bfs(start: int):
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if nbr not in parent:
                    parent[nbr] = node
                    queue.append(nbr)
        # Farthest reachable tip from `start`
        farthest_tip = max((n for n in tips if n in parent), key=lambda n: _depth(n, parent))
        return farthest_tip, parent

    def _depth(n: int, parent: dict) -> int:
        d = 0
        while parent[n] is not None:
            n = parent[n]
            d += 1
        return d

    # Two-pass BFS: tip1 -> tip2 (diameter)
    tip1, _ = bfs(tips[0])
    tip2, parent = bfs(tip1)

    # Reconstruct path
    path = []
    node = tip2
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return tip1, tip2, path


def sort_pnodes(adata):
    r"""
    Compute trajectory ordering indices btw principal nodes in latent space
    """
    assert 'graph' in adata.uns.keys(), "Please run Principal Curve first"
    edge_list = np.array(
        igraph.Graph.Adjacency(
            (adata.uns['graph']['B'] > 0).tolist(), 
            mode='undirected'
        ).get_edgelist()
    )
    tips = adata.uns['graph']['tips']
    tip1, tip2, path = prune_branches(edge_list, tips)

    # Update pruned path & tips
    adata.uns['graph']['tips'] = [tip1, tip2]
    adata.uns['graph']['pnode_indices'] = path

    # root_node, term_node = adata.uns['graph']['tips']
    # curr_node = root_node
    # ypos, xpos = np.asarray(np.where(edge_list == root_node)).T[0]  # coords of current node
    
    # path = [curr_node]
    # while term_node not in path:
    #     xpos = 1 - xpos  # adj. principle node
    #     curr_node = edge_list[ypos, xpos]
    #     path.append(curr_node)
    #     if curr_node == term_node:
    #         break
        
    #     # Update `ypos`, `xpos` of `curr_node`:
    #     # each node except root & terminal appears twice 
    #     coords = np.asarray(np.where(edge_list == curr_node)).T  # dim: [2, 2]
    #     if np.array_equal(coords[0], [ypos, xpos]):
    #         ypos, xpos = coords[1]
    #     else:
    #         ypos, xpos = coords[0]
    # adata.uns['graph']['pnode_indices'] = path
    
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

    # Compute trajectory ordering of principal nodes
    node_indices = sort_pnodes(adata)
    if verbose:
        print('Principal Node ordering:', node_indices)

    # Filter out pruned "branching" principal nodes
    pcurve_repr = pcurve_repr[node_indices, :]
    n_pts = adata.shape[0]
    n_nodes = pcurve_repr.shape[0]
    dists = np.zeros((n_pts, n_nodes), dtype=np.float32)

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
    ppt_lambda: float = 1000.,
    ppt_sigma: float = .1,
    use_rep: str = 'X_z',
    n_nodes: int = 50,
    n_neighbors: int = 100,
    dist_metric: str = 'euclidean',
    seed: int = 42,
    verbose=False,
):
    r"""
    Compute a smooth trajectory \gamma(t): [0, 1] -> R^D projected from the 
    representation manifold with SimplePPT; Optional marker-based supervision 
    to rotate the direction of \gamma(t) source tip.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData of latent representation w/ computed elastic principal graph
    root_marker : str
        Optional marker close to 'root' to rotate (+/-) of the trajectory
    ppt_lambda : float
        Tree length penalty term for manifold smoothness (higher ->  simpler manifold)
    ppt_sigma : float
        Regularization term (higher -> less sentisitive to localized structure)
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
    seed : int
        Random seed for SimplePPT principal curve computation
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

    scf.tl.tree(
        adata,
        use_rep=use_rep,
        Nodes=n_nodes,
        ppt_lambda=ppt_lambda,
        ppt_sigma=ppt_sigma,
        seed=seed
    )

    # Note: if assuming "curve" manifold, prune branches & corresponding sub-paths
    edge_list = np.array(
        igraph.Graph.Adjacency(
            (adata.uns['graph']['B'] > 0).tolist(), 
            mode='undirected'
        ).get_edgelist()
    )
    tip1, tip2, path = prune_branches(edge_list, adata.uns['graph']['tips'])
    adata.uns['graph']['tips'] = [tip1, tip2]
    nodes_to_keep = np.sort(path)
    
    adata.uns['graph']['F'] = adata.uns['graph']['F'][:, nodes_to_keep]
    adata.uns['graph']['B'] = adata.uns['graph']['B'][np.ix_(nodes_to_keep, nodes_to_keep)]
    adata.obsm['X_R'] = adata.obsm['X_R'][:, nodes_to_keep]

    # Compute pseudotime, normalize to [0, 1]
    # Optionally rotate w.r.t. the root marker
    if root_marker is not None and root_marker in adata.var_names:
        scf.tl.root(adata, root_marker)
    else:
        scf.tl.root(adata, tip1)

    scf.tl.pseudotime(adata, n_jobs=20, seed=seed)
    adata.obs['t'] = (adata.obs['t'] - adata.obs['t'].min()) / (adata.obs['t'].max() - adata.obs['t'].min())

    # Dummy features
    adata.obs['seg'] = '1'
    adata.obs['seg'] = adata.obs['seg'].astype('category')  

    return None
