import os
import igraph
import elpigraph
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import scFates as scf

from utils import to_dense_array
from typing import List


# -------------------
#  Helper functions
# -------------------

def get_edges(path):
    """Get undirected edges from a given path"""
    return {(path[i], path[i+1]) for i in range(len(path)-1)} | \
           {(path[i+1], path[i]) for i in range(len(path)-1)}


def sort_nodes(adata, root_node: int = None, term_node: int = None) -> List[int]:
    r"""Compute principal node ordering from root to term"""
    assert 'graph' in adata.uns.keys(), "Please run Principal Curve first"
    al = np.array(
        igraph.Graph.Adjacency(
            (adata.uns['graph']['B'] > 0).tolist(), 
            mode='undirected'
        ).get_edgelist()
    )
    def _traverse(al, root_node, term_node):
        curr_node = root_node
        ypos, xpos = np.asarray(np.where(al == root_node)).T[0]  # coords of current node
        
        path = [curr_node]
        while term_node not in path:
            xpos = 1 - xpos  # adj. principle node
            curr_node = al[ypos, xpos]
            path.append(curr_node)
            if curr_node == term_node:
                break
            
            # Each node except root & terminal appears twice 
            coords = np.asarray(np.where(al == curr_node)).T  # dim: [2, 2]
            if np.array_equal(coords[0], [ypos, xpos]):
                ypos, xpos = coords[1]
            else:
                ypos, xpos = coords[0]
        return path

    # DFS traverse from root to terminal
    if root_node is None or term_node is None:
        root_node, term_node = adata.uns['graph']['tips'][:2]
    try:
        path = _traverse(al, root_node, term_node)
    except IndexError:
        # reverse the traverse direction
        path = _traverse(al, term_node, root_node)[::-1]
    return path


def get_cell_projections(adata, edge_projections, path_dict):
    r"""
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


def get_branch_path(B, tip, forks):
    r"""
    Find all nodes from tip to nearest fork using igraph's shortest path.
    
    This follows the scFates implementation:
    https://github.com/LouisFaure/scFates/blob/master/scFates/tools/graph_operations.py
    
    Parameters
    ----------
    B : np.ndarray
        Adjacency matrix of the principal graph
    tip : int
        Tip node index to start from
    forks : np.ndarray
        Array of fork node indices (nodes with degree > 2)
        
    Returns
    -------
    branch_nodes : np.ndarray
        Array of node indices from tip to (but NOT including) the nearest fork
    """
    # Build igraph from adjacency matrix
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    
    # If no forks exist (linear graph), return just the tip
    if len(forks) == 0:
        return np.array([tip])
    
    # Get all shortest paths from tip to each fork
    all_paths = g.get_all_shortest_paths(tip, forks)
    
    if len(all_paths) == 0:
        return np.array([tip])
    
    # Find the shortest path (nearest fork)
    path_lengths = [len(p) for p in all_paths]
    idx_min = np.argmin(path_lengths)
    shortest_path = np.array(all_paths[idx_min])
    
    # Return all nodes EXCEPT the fork (last node in path)
    # This is what scFates does: g.get_shortest_paths(...)[0][:-1]
    branch_nodes = shortest_path[:-1]
    
    return branch_nodes


def get_branches_to_remove(adata, tips_to_keep, verbose=False):
    r"""
    Identify all branch nodes that will be removed when pruning unwanted tips.
    
    This function returns the branches WITHOUT modifying adata, allowing you to
    inspect what will be removed before calling prune_tree.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with computed principal tree in .uns['graph']
    tips_to_keep : list
        List of tip node indices to keep
    verbose : bool
        Print debug information
        
    Returns
    -------
    branches_info : dict
        Dictionary with:
        - 'tips_to_remove': list of tip indices to remove
        - 'branches': dict mapping tip -> array of nodes in that branch
        - 'all_nodes_to_remove': combined set of all nodes to be removed
        - 'node_map': dict mapping original node idx -> new idx after removal
    """
    graph = adata.uns['graph']
    B = graph['B'].copy()
    
    # Get current tips and forks from graph
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected")
    tips = np.argwhere(np.array(g.degree()) == 1).flatten()
    forks = np.argwhere(np.array(g.degree()) > 2).flatten()
    
    tips_to_remove = [t for t in tips if t not in tips_to_keep]
    
    if verbose:
        print(f"All tips: {tips}")
        print(f"Tips to keep: {tips_to_keep}")
        print(f"Tips to remove: {tips_to_remove}")
        print(f"Forks: {forks}")
    
    # Collect all branches to remove
    branches = {}
    all_nodes_to_remove = set()
    
    for tip in tips_to_remove:
        branch_nodes = get_branch_path(B, tip, forks)
        branches[tip] = branch_nodes
        all_nodes_to_remove.update(branch_nodes)
        
        if verbose:
            print(f"  Tip {tip} -> branch nodes: {branch_nodes}")
    
    all_nodes_to_remove = np.array(sorted(all_nodes_to_remove))
    
    # Compute node index mapping: original_idx -> new_idx after removal
    n_nodes = B.shape[0]
    node_map = {}
    new_idx = 0
    for orig_idx in range(n_nodes):
        if orig_idx not in all_nodes_to_remove:
            node_map[orig_idx] = new_idx
            new_idx += 1
        # Nodes to remove won't be in the map
    
    if verbose:
        print(f"Total nodes to remove: {len(all_nodes_to_remove)}")
        print(f"Nodes to remove: {all_nodes_to_remove}")
        print(f"Node map (original -> new): {node_map}")
    
    return {
        'tips_to_remove': tips_to_remove,
        'branches': branches,
        'all_nodes_to_remove': all_nodes_to_remove,
        'node_map': node_map
    }


def prune_tree(adata, tips_to_keep, verbose=False):
    r"""
    Remove unwanted tips by calling scf.tl.cleanup and return the node index mapping.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with computed principal tree in .uns['graph']
        This will be modified in-place.
    tips_to_keep : list
        List of tip node indices to keep
    verbose : bool
        Print debug information
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'branches_removed': dict mapping original tip -> branch nodes removed
        - 'all_nodes_removed': array of all removed node indices
        - 'node_map': dict mapping original node idx -> new idx after pruning
    """
    # First, get the branches that will be removed (before any modification)
    branches_info = get_branches_to_remove(adata, tips_to_keep, verbose=verbose)
    
    tips_to_remove = branches_info['tips_to_remove']
    
    if len(tips_to_remove) == 0:
        if verbose:
            print("No tips to remove.")
        return {
            'branches_removed': {},
            'all_nodes_removed': np.array([]),
            'node_map': {i: i for i in range(adata.uns['graph']['B'].shape[0])}
        }
    scf.tl.cleanup(adata, leaves=tips_to_remove)
    
    if verbose:
        print(f"Cleanup complete. New graph has {adata.uns['graph']['B'].shape[0]} nodes.")
        print(f"New tips: {adata.uns['graph']['tips']}")
        print(f"New forks: {adata.uns['graph']['forks']}")
    
    return None


# ---------------------------
#  Principal graph methods
# ---------------------------


def get_curve(
    adata: sc.AnnData, 
    use_rep: str = 'X_z',
    n_nodes: int = 20,
    epg_mu: float = 1.0,
    epg_lambda: float = 0.1,
    trim_radius_ratio: float = 0.1,
    n_repeat: int = 1
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
    trim_radius_ratio: float = 0.1
        Ratio to define trimming radius against embedding 
        value range for robust fitting
    """
    assert use_rep in adata.obsm.keys(), \
        "Please run the LYNX model to get latent representation first"
    
    # Define radius for robust fitting against outliers
    emb = adata.obsm[use_rep]
    trim_radius = trim_radius_ratio*(emb.max()-emb.min()) 

    # Estimate elastic principal graph
    curve = elpigraph.computeElasticPrincipalCurve(
        emb,
        NumNodes=n_nodes,
        Mu=epg_mu,
        Lambda=epg_lambda,
        TrimmingRadius=trim_radius,
        nReps=n_repeat, 
        Do_PCA=False
    )[-1]
    curve = elpigraph.ExtendLeaves(emb, curve, Mode='WeightedCentroid', TrimmingRadius=trim_radius)

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
    ppt_lambda: float = 10.0,
    seed: int = 42,
    plot_graph: bool = False
):
    r"""
    Compute a tree-like trajectory via elastic principal graph fitted
    onto the given manifold (adata.obsm[use_rep]).
    Principal graph params updated in-place: adata.uns['graph']
    
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
    
    Returns
    -------
    principal_graph : dict
        Fitted principal graph stored in adata.uns['graph']
    """
    assert use_rep in adata.obsm.keys(), \
        "Please run the LYNX model to get latent representation first"
    
    # Compute PC / UMAP for visualization
    if 'X_pca' not in adata.obsm.keys():
        adata_embed = sc.AnnData(adata.obsm[use_rep].copy())
        sc.pp.pca(adata_embed, n_comps=adata_embed.shape[1]-1)
        adata.obsm['X_pca'] = adata_embed.obsm['X_pca']
        del adata_embed

    if plot_graph:
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15)
        sc.tl.umap(adata)

    # Traverse through tree complexity regularizations (sigma)
    nsteps = 100
    sigma = scf.tl.explore_sigma(
        adata,
        Nodes=n_nodes,
        use_rep=use_rep,
        nsteps=nsteps,
        lambda_=ppt_lambda,
        sigmas=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
        seed=seed,
        plot=plot_graph
    )
    
    # Cleanup principal tree
    principal_graph = scf.tl.tree(
        adata,
        use_rep=use_rep,
        Nodes=n_nodes,
        ppt_nsteps=nsteps,
        ppt_lambda=ppt_lambda,
        ppt_sigma=sigma,
    )
    scf.tl.cleanup(adata, minbranchlength=10)
    
    if plot_graph:
        scf.pl.graph(
            adata, basis='pca', 
            title='Principal Tree\nPC space'
        )
    
    return adata.uns['graph'].copy()


def compute_pseudotime(
    adata: sc.AnnData,
    principal_graph: dict,
    n_nodes: int = 20,
    use_rep: str = 'X_z',
    source: int = None,
    seed: int = 42,
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

    # Determine whether it's a curve or tree,
    forks = adata.uns['graph']['forks']
    if len(forks) == 0:
        print("Computing pseudotime on principal curve...")
        elpigraph.utils.getPseudotime(
            adata.obsm[use_rep], 
            principal_graph, 
            source=source
        )
        t = principal_graph['pseudotime']
    else:
        print("Computing pseudotime on principal tree...")
        scf.tl.root(adata, source)
        scf.tl.pseudotime(adata, n_jobs=os.cpu_count()//2, seed=seed)
        t = adata.obs['t'].values

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

