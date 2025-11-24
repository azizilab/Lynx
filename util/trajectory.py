import igraph
import elpigraph
import numpy as np
import pandas as pd
import scanpy as sc

from collections import OrderedDict, defaultdict, deque
from scipy.spatial.distance import cdist
from utils import to_dense_array
from typing import List


# -------------------
#  Helper functions
# -------------------

def get_edges(path):
    """Get undirected edges from a given path"""
    return {(path[i], path[i+1]) for i in range(len(path)-1)} | \
           {(path[i+1], path[i]) for i in range(len(path)-1)}


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


def sort_nodes(adata, root_node: int = None, term_node: int = None) -> List[int]:
    r"""Compute principal node ordering from root to term"""
    assert 'graph' in adata.uns.keys(), "Please run Principal Curve first"
    al = np.array(
        igraph.Graph.Adjacency(
            (adata.uns['graph']['B'] > 0).tolist(), 
            mode='undirected'
        ).get_edgelist()
    )

    if root_node is None or term_node is None:
        root_node, term_node = adata.uns['graph']['tips'][:2]
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

    return path


def dist_to_principal_node(
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
    node_indices = sort_nodes(adata)
    pcurve_repr = pcurve_repr[node_indices, :]
    root_repr, term_repr = pcurve_repr[0], pcurve_repr[-1]

    if dist_metric == 'euclidean':
        dists[:, 0] = cdist(repr, np.expand_dims(root_repr, 0)).squeeze()
        dists[:, -1] = cdist(repr, np.expand_dims(term_repr, 0)).squeeze()
    elif dist_metric == 'knn':
        dists[:, 0] = get_knn_dist(repr=repr, root_repr=root_repr, k=k)
        dists[:, -1] = get_knn_dist(repr=repr, root_repr=term_repr, k=k)
    else:
        raise NotImplementedError(dist_metric)

    return dists


def get_cell_projections(adata, edge_projections, path_dict):
    """
    Assign each cell to the closest principal graph edges
    """
    assert 'graph' in adata.uns.keys(), "Please compute principal tree first"
    assignment = ['NA']*len(adata)
    edges = adata.uns['graph']['edges']
    for i, edge_idx in enumerate(edge_projections):
        edge = edges[edge_idx]
        for k in path_dict.keys():
            if tuple(edge) in path_dict[k]:
                assignment[i] = k
                break

    adata.obs['seg'] = assignment
    return assignment


# ---------------------------
#  Principal graph methods
# ---------------------------


def get_curve(
    adata: sc.AnnData, 
    use_rep: str = 'X_z',
    n_nodes: int = 20,
    epg_mu: float = 1.0,
    epg_lambda: float = 0.1,
    n_repeat: int = 5
):
    r"""
    Compute a smooth linear trajectory (t) \in [0, 1] via principal graph fitted
    onto the given manifold (adata.obsm[use_rep]); Use optional marker to rotate the 
    (+/-) sign of the trajectory s.t. the `root_marker` enriched end is close to t=0.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData of latent representation w/ computed elastic principal graph
    root_marker : str
        Optional marker close to 'root' to rotate (+/-) of the trajectory
    use_rep : str
        Use the indicated representation. 'X' or any key for .obsm is valid. 
        If None, the representation is chosen automatically
    n_nodes : int
        # principal nodes to infer 
        Increase `n_nodes` get more localized principal manifold    
    """
    assert use_rep in adata.obsm.keys(), \
        "Please run the LYNX model to get latent representation first"

    # Estimate elastic principal graph
    curve = elpigraph.computeElasticPrincipalCurve(
        adata.obsm[use_rep],
        NumNodes=n_nodes,
        Mu=epg_mu,
        Lambda=epg_lambda,
        nReps=n_repeat, 
        Do_PCA=False
    )[-1]
    curve = elpigraph.ExtendLeaves(adata.obsm[use_rep], curve, Mode='WeightedCentroid')

    # Extract principal graph properties
    graph = {}
    g = igraph.Graph(directed=False)
    g.add_vertices(np.unique(curve["Edges"][0].flatten().astype(int)))
    g.add_edges(pd.DataFrame(curve["Edges"][0]).astype(int).apply(tuple, axis=1).values)
    B = np.asarray(g.get_adjacency().data)
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    
    graph['B'] = B   # (n_nodes, n_nodes)
    graph['F'] = curve['NodePositions'].astype(np.float32)  # principal node manifold (n_nodes, K)
    
    graph['tips'] = np.argwhere(np.array(g.degree()) == 1).flatten()
    graph['forks'] = np.argwhere(np.array(g.degree()) > 2).flatten()
    graph['energy'] = curve['FinalReport']['ENERGY']
    graph['mse'] = curve['FinalReport']['MSE'] 

    adata.uns['graph'] = graph 
    adata.uns['graph']['pnode_indices'] = sort_nodes(adata)
    return curve


def get_tree(
    adata: sc.AnnData, 
    use_rep: str = 'X_z',
    n_nodes: int = 20,
    epg_mu: float = 1.0,
    epg_lambda: float = 0.1,
    n_repeat: int = 5,
    max_shift: int = 3,
    plot_graph: bool = True,
):
    r"""
    Compute a tree-like trajectory via elastic principal graph fitted
    onto the given manifold (adata.obsm[use_rep]).

    Parameters
    ----------
    adata : sc.AnnData
        AnnData of latent representation w/ computed elastic principal graph
    use_rep : str
        Use the indicated representation. 'X' or any key for .obsm is valid.            
    n_nodes : int
        # principal nodes to infer 
        Increase `n_nodes` get more localized principal manifold
    plot_graph : bool
        Whether to plot the fitted principal graph onto the latent space (PCA proj.)
    """
    assert use_rep in adata.obsm.keys(), \
        "Please run the LYNX model to get latent representation first"
    
    tree = elpigraph.computeElasticPrincipalTree(
        adata.obsm[use_rep],
        NumNodes=n_nodes,
        Mu=epg_mu,
        Lambda=epg_lambda,
        nReps=n_repeat,
        Do_PCA=False,
        ICOver='Density',
        DensityRadius=1.0

    )[-1]
    tree = elpigraph.ExtendLeaves(adata.obsm['X_z'], tree, Mode='WeightedCentroid')
    tree['NodePositions'] = tree['NodePositions'].astype(np.float32)

    # Adjust branching points based on density
    update_pg_dict = elpigraph.ShiftBranching(
        adata.obsm['X_z'], tree, MaxShift=max_shift,
        SelectionMode='NodeDensity', DensityRadius=1.0
    )
    principal_nodes = np.unique(update_pg_dict['Edges'])
    tree['Edges'] = [
        update_pg_dict['Edges'],
        0.5*np.ones(update_pg_dict['Edges'].shape[0], dtype=np.float32)
    ]
    tree['NodePositions'] = update_pg_dict['NodePositions'][:len(principal_nodes)]  # Remove dangling nodes
    tree['NodePositions'] = tree['NodePositions'].astype(np.float32)

    # Extract principal graph properties
    graph = {}
    g = igraph.Graph(directed=False)
    g.add_vertices(np.unique(tree["Edges"][0].flatten().astype(int)))
    g.add_edges(pd.DataFrame(tree["Edges"][0]).astype(int).apply(tuple, axis=1).values)    
    B = np.asarray(g.get_adjacency().data)
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode='undirected')

    graph['B'] = B  # adjacency matrix(n_nodes, n_nodes)
    graph['F'] = tree['NodePositions']  # principal node manifold(n_nodes, K)
    
    graph['edges'] = tree['Edges'][0]
    graph['pnode_indices'] = np.arange(tree['NodePositions'].shape[0])
    graph['tips'] = np.argwhere(np.array(g.degree()) == 1).flatten()
    graph['forks'] = np.argwhere(np.array(g.degree()) > 2).flatten()
    adata.uns['graph'] = graph 

    if plot_graph:
        elpigraph.plot.PlotPG(adata.obsm[use_rep], tree, Do_PCA=True) 
    return tree


def compute_pseudotime(
    adata: sc.AnnData,
    principal_graph: dict,
    n_nodes: int = 20,
    use_rep: str = 'X_z',
    source: int = None,
    root_marker: str = None
):
    r"""
    Compute pseudotime (t) \in [0, 1] along the given principal graph
    fitted onto the latent space (adata.obsm[use_rep]); Use optional marker to 
    rotate the (+/-) sign of the trajectory at which the `root_marker` enriches
    
    Parameters
    ----------
    adata : sc.AnnData
        AnnData of latent representation w/ computed elastic principal graph
    root_marker : str
        Optional marker close to 'root' to rotate (+/-) of the trajectory
    use_rep : str
        Use the indicated representation. 'X' or any key for .obsm is valid. 
        If None, the representation is chosen automatically
    n_nodes : int
        # principal nodes to infer 
        Increase `n_nodes` get more localized principal manifold  
    """
    assert 'graph' in adata.uns.keys(), \
        "Please run Principal Graph on LYNX latent first"
    if source is None:
        source = adata.uns['graph']['tips'][0]
    elpigraph.utils.getPseudotime(
        adata.obsm[use_rep], 
        principal_graph, 
        source=source
    )
    t = principal_graph['pseudotime']
    adata.obs['t'] = (t - t.min()) / (t.max() - t.min())

    # Rotate axis s.t. `root_marker` enriched end --> 0
    if root_marker in adata.var_names:
        n_neighbors = adata.shape[0] // n_nodes
        root_expr = to_dense_array(
            adata[np.argsort(adata.obs['t'])[::-1][:n_neighbors], root_marker].X
        ).mean()
        tip_expr = to_dense_array(
            adata[np.argsort(adata.obs['t'])[:n_neighbors], root_marker].X
        ).mean()

        if root_expr > tip_expr:
            adata.obs['t'] = 1 - adata.obs['t']

    return None


