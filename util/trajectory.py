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
    root_marker: str = None,
    ppt_lambda: float = 1000.,
    ppt_sigma: float = .1,
    ppt_niter: int = 200,
    use_rep: str = 'X_z',
    n_nodes: int = None,
    seed: int = 42
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
    ppt_niter : int
        # iterations for SimplePPT principal tree fitting
    use_rep : str
        Use the indicated representation. 'X' or any key for .obsm is valid. 
        If None, the representation is chosen automatically
    n_nodes : int
        # principal nodes to infer 
        Increase `n_nodes` get more localized principal manifold
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
    
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if n_nodes is None:
        n_nodes = int(adata.shape[0] * 0.1)

    scf.tl.tree(
        adata,
        use_rep=use_rep,
        Nodes=n_nodes,
        ppt_lambda=ppt_lambda,
        ppt_sigma=ppt_sigma,
        ppt_nsteps=ppt_niter,
        seed=seed,
        device=device
    )

    edge_list = np.array(
        igraph.Graph.Adjacency(
            (adata.uns['graph']['B'] > 0).tolist(), 
            mode='undirected'
        ).get_edgelist()
    )
    
    # Prune excessive roots to maintain the main "curve"
    path = prune_branches(edge_list, adata.uns['graph']['tips'])
    nodes_to_keep = np.sort(path)
    node_indices = np.unique(path, return_inverse=True)[-1]

    adata.uns['graph']['tips'] = [node_indices[0], node_indices[-1]]
    adata.uns['graph']['pnode_indices'] = node_indices
    adata.uns['graph']['F'] = adata.uns['graph']['F'][:, nodes_to_keep]
    adata.uns['graph']['B'] = adata.uns['graph']['B'][np.ix_(nodes_to_keep, nodes_to_keep)]
    adata.obsm['X_R'] = adata.obsm['X_R'][:, nodes_to_keep]

    # Compute pseudotime, normalize to [0, 1], (optional) rotate w.r.t. the root marker
    if root_marker is not None and root_marker in adata.var_names:
        scf.tl.root(adata, root_marker)
    else:
        scf.tl.root(adata, adata.uns['graph']['tips'][0])

    scf.tl.pseudotime(adata, n_jobs=20, seed=seed)
    adata.obs['t'] = (adata.obs['t'] - adata.obs['t'].min()) / (adata.obs['t'].max() - adata.obs['t'].min())

    # Dummy features
    adata.obs['seg'] = '1'
    adata.obs['seg'] = adata.obs['seg'].astype('category')  

    return None
