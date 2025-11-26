import os
import igraph
import elpigraph
import numpy as np
import pandas as pd
import scanpy as sc
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
    # if 'X_pca' not in adata.obsm.keys():
    #     adata_embed = sc.AnnData(adata.obsm[use_rep].copy())
    #     sc.pp.pca(adata_embed, n_comps=adata_embed.shape[1]-1)
    #     adata.obsm['X_pca'] = adata_embed.obsm['X_pca']
    #     del adata_embed

    # if plot_graph:
    #     sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15)
    #     sc.tl.umap(adata)

    # # Traverse through tree complexity regularizations (sigma)
    # _ = scf.tl.explore_sigma(
    #     adata,
    #     Nodes=n_nodes,
    #     use_rep=use_rep,
    #     nsteps=50,
    #     lambda_=ppt_lambda,
    #     sigmas=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
    #     seed=seed,
    #     plot=plot_graph
    # )
    
    # Cleanup principal tree
    scf.tl.cleanup(adata, minbranchlength=int(0.1*n_nodes))
    
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

