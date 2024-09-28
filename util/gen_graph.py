import os
import numpy as np
# import cupy as cp
import torch
import networkx as nx

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from torch_geometric.nn import TopKPooling
from torch_geometric import utils as pyg_utils
from scipy.sparse import csr_matrix


"""
def get_centroids(mask):
    mempool = cp.get_default_memory_pool()

    mask_cp = cp.array(mask)
    coords = []
    labels = cp.unique(mask_cp)[1:]
    for label in labels:
        pts_y, pts_x = cp.nonzero(mask_cp == label)
        cy = cp.round(pts_y.mean()).astype(cp.uint32)
        cx = cp.round(pts_x.mean()).astype(cp.uint32)
        coords.append([int(cy.get()), int(cx.get())])
        mempool.free_all_blocks()    

    return np.array(coords)
"""

def get_coords_from_graph(G, node_list=None):
    try:
        nodes = G.nodes if node_list is None else node_list
        return np.array([
            [G.nodes[n]['pos'][1], G.nodes[n]['pos'][0]]
             for n in nodes
        ])
    except KeyError as e:
        print(e, 'Please assign `pos` to graph nodes first')


def construct_graph(coords, k=5, r=np.inf, weighted=False):
    G = nx.Graph()
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
    distances, nn_indices = nbrs.kneighbors(coords)
    distances, nn_indices = distances[:, 1:], nn_indices[:, 1:]  # Skip self

    # Construct Graph w/ all nodes
    for i, (y, x) in enumerate(coords):
        G.add_node(i, pos=(x, y))  # XY-based coords

    # Add edges within `r` resolution
    for i in range(len(coords)):
        for j, distance in zip(nn_indices[i], distances[i]):
            if i != j and (r == np.inf or distance <= r):
                if weighted:
                    G.add_edge(i, j, weight=1/distance)
                else:
                    G.add_edge(i, j)

    return G


def sample_nodes(
    G, k=5, r=np.inf, 
    sample_ratio=0.1, res=0.5
):
    partition = nx.community.louvain_communities(G, resolution=res)
    sampled_nodes = []
    for nodes in partition:
        sample_size = np.ceil(len(nodes)*sample_ratio).astype(np.uint8)
        sampled_nodes.extend(
            np.random.choice(list(nodes), size=sample_size, replace=False)
        )
    coords = get_coords_from_graph(G, sampled_nodes)
    G_new = construct_graph(coords, k=k, r=r)
    return G_new


def construct_feature_matrix(img, coords, r=4):
    assert img.ndim == 3,\
        "Image dimension needs to be (C, Y, X)"
    
    n_nodes, n_chans = coords.shape[0], img.shape[0]
    ydim, xdim = img.shape[1:]
    features = np.zeros((n_nodes, n_chans))
    for i, (y, x) in enumerate(coords):
        
        indices = (slice(0, n_chans), 
                   slice(max(0, y-r), min(ydim-1, y+r)), 
                   slice(max(0, x-r), min(xdim-1, x+r)))
        features[i] = img[indices].sum((1, 2))

    return features


def pooled_edge_attrs_to_graph(
    edge_index, edge_weight, 
    G=None, perm=None, to_nx=False
):
    """
    Construct sparse adjacency matrix / nx.graph 
    from pooled (edge_index, edge_attrs)
    """
    if to_nx:
        assert G is not None and perm is not None, \
        "Requires original graph G & perm indices"

    adj_mat = pyg_utils.to_torch_coo_tensor(
        edge_index=edge_index, 
        edge_attr=edge_weight.detach()
    )
    
    if to_nx:
        adj_mat_scipy = csr_matrix((adj_mat.values(), adj_mat.indices()), shape=adj_mat.shape)
        G_new = nx.from_scipy_sparse_array(adj_mat_scipy)

        node_attrs = {n: G.nodes[k] for n, k in zip(G_new.nodes, perm)}
        nx.set_node_attributes(G_new, node_attrs)
        return adj_mat, G_new
    else:
        return adj_mat
    
