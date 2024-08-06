import numpy as np
import pandas as pd
import igraph
import scanpy as sc
import scFates as scf

from scanpy.tools._dpt import DPT
from scipy.spatial.distance import cdist


def get_pcurve_path(adata):
    """
    Compute linear trajectory btw principal nodes in latent space
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
        xpos = 1 - xpos  # adjacenct principle node
        curr_node = al[ypos, xpos]
        path.append(curr_node)
        if curr_node == term_node:
            break
        
        # Update `ypos`, `xpos` of `curr_node`:
        # besides root & term node, every other node appears twice 
        coords = np.asarray(np.where(al == curr_node)).T  # dim: [2, 2]
        if np.array_equal(coords[0], [ypos, xpos]):
            ypos, xpos = coords[1]
        else:
            ypos, xpos = coords[0]
        
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


def dist_to_pcurve(
    adata, 
    dist_metric='euclidean',
    verbose=False
):
    """
    Compute diffusion distance (DPT: D(x, y)) btw each cell (x) 
    and latent representation vector of each principal node (y)
    """ 
    assert 'graph' in adata.uns.keys(), "Please run Principal Curve first"
    
    pcurve_reprs = adata.uns['graph']['F'].T  # dim:[n_nodes, n_latent (K)]
    n_pts = adata.shape[0]
    n_nodes, n_latent = pcurve_reprs.shape
    dists = np.zeros((n_pts, n_nodes), dtype=np.float32)

    # Compute trajectory ordering of principal nodes
    pcurve_indices = get_pcurve_path(adata)
    if verbose:
        print('Principal Node ordering:', pcurve_indices)
    pcurve_reprs = pcurve_reprs[pcurve_indices, :]

    for i, pcurve in enumerate(pcurve_reprs):
        if dist_metric == 'euclidean':
            dists[:, i] = cdist(adata.X, np.expand_dims(pcurve, 0)).squeeze() 
        elif dist_metric == 'geodesic':
            dists[:, i] = np.array([get_geodesic_dist(z, pcurve) 
                                    for z in adata.X])
        elif dist_metric == 'diffusion':
            dists[:, i] = get_diffusion_dist(adata, pcurve_repr[i], n_neighbors)  # Diffusion distance
        else:
            'Distance metric {} not implemented'.format(dist_metric)
    
    return dists