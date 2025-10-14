import numpy as np
import torch
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


def prune_branches(edge_list: np.ndarray, tips: List[int]):
    """
    Given an acyclic undirected graph (tree) with edge_list representation, prune 
    branches and return the sorted path connected by tips with the longest distance
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

    return path


def compute_trajectory(
    adata: sc.AnnData, 
    root: str = None, 
    tip: str = None,
    root_marker: str = None,
    use_rep: str = 'X_z',
    n_nodes: int = 20,
    n_neighbors: int = 100,
    dist_metric: str = 'euclidean',
    epg_mu: float = .05,
    epg_lambda: float = .1,
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
 
    scf.tl.curve(
        adata,
        use_rep=use_rep,
        Nodes=n_nodes,
        epg_extend_leaves=True,
        ndims_rep=adata.obsm[use_rep].shape[-1],
        epg_mu=epg_mu,
        epg_lambda=epg_lambda
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
        distances, _ = dist_to_pnode(
            adata,
            use_rep=use_rep,
            dist_metric=dist_metric, 
            k=n_neighbors,
            verbose=verbose
        )
        
    t = distances[:, 0] / (distances[:, 0]+distances[:, -1])  
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


# TODO: separate functions for curve vs. tree computation???
# def compute_trajectory(
#     adata: sc.AnnData, 
#     device: str = 'cpu',
#     root_marker: str = None,
#     ppt_lambda: float = 1000.,
#     ppt_sigma: float = .1,
#     ppt_niter: int = 200,
#     use_rep: str = 'X_z',
#     n_nodes: int = None,
#     learn_curve: bool = True,
#     seed: int = 42
# ):
#     r"""
#     Compute a smooth trajectory \gamma(t): [0, 1] -> R^D projected from the 
#     representation manifold with SimplePPT; Optional marker-based supervision 
#     to rotate the direction of \gamma(t) source tip.

#     Parameters
#     ----------
#     adata : sc.AnnData
#         AnnData of latent representation w/ computed elastic principal graph
#     root_marker : str
#         Optional marker close to 'root' to rotate (+/-) of the trajectory
#     ppt_lambda : float
#         Tree length penalty term for manifold smoothness (higher ->  simpler manifold)
#     ppt_sigma : float
#         Regularization term (higher -> less sentisitive to localized structure)
#     ppt_niter : int
#         # iterations for SimplePPT principal tree fitting
#     use_rep : str
#         Use the indicated representation. 'X' or any key for .obsm is valid. 
#         If None, the representation is chosen automatically
#     n_nodes : int
#         # principal nodes to infer 
#         Increase `n_nodes` get more localized principal manifold
#     learn_curve : bool
#         Whether to learn a single curve (True) or a branching tree (False)
#     Returns
#     -------
#     None. 
#         Sets the following fields:
        
#         `adata.obs['t']` : np.ndarray
#             Smooth gradient / pseudotime
#         `adata.obs['t_discrete]` : np.ndarray
#             Discrete "zonation assignment" to the closest principal nodes
#     """
#     assert use_rep in adata.obsm.keys(), \
#         "Please run the model to obtain latent representation `z` first"
    
#     # device = 'gpu' if torch.cuda.is_available() else 'cpu'
#     if n_nodes is None:
#         n_nodes = int(adata.shape[0] * 0.1)

#     scf.tl.tree(
#         adata,
#         use_rep=use_rep,
#         Nodes=n_nodes,
#         ppt_lambda=ppt_lambda,
#         ppt_sigma=ppt_sigma,
#         ppt_nsteps=ppt_niter,
#         seed=seed,
#         device=device
#     )

#     edge_list = np.array(
#         igraph.Graph.Adjacency(
#             (adata.uns['graph']['B'] > 0).tolist(), 
#             mode='undirected'
#         ).get_edgelist()
#     )
    
#     # Prune excessive roots to maintain the main "curve"
#     if learn_curve:
#         path = prune_branches(edge_list, adata.uns['graph']['tips'])
#         nodes_to_keep = np.sort(path)
#         node_indices = np.unique(path, return_inverse=True)[-1]

#         adata.uns['graph']['tips'] = [node_indices[0], node_indices[-1]]
#         adata.uns['graph']['pnode_indices'] = node_indices
#         adata.uns['graph']['F'] = adata.uns['graph']['F'][:, nodes_to_keep]
#         adata.uns['graph']['B'] = adata.uns['graph']['B'][np.ix_(nodes_to_keep, nodes_to_keep)]
#         adata.obsm['X_R'] = adata.obsm['X_R'][:, nodes_to_keep]

#         # Compute pseudotime, normalize to [0, 1], (optional) rotate w.r.t. the root marker
#         if root_marker is not None and root_marker in adata.var_names:
#             scf.tl.root(adata, root_marker)
#         else:
#             scf.tl.root(adata, adata.uns['graph']['tips'][0])

#         # TODO: check whether pseudotime computation is valid? We need projection of each points onto principal curve
#         scf.tl.pseudotime(adata, n_jobs=1, seed=seed)
#         adata.obs['t'] = (adata.obs['t'] - adata.obs['t'].min()) / (adata.obs['t'].max() - adata.obs['t'].min())

#         # Dummy features
#         adata.obs['seg'] = '1'
#         adata.obs['seg'] = adata.obs['seg'].astype('category')  

#     return None
