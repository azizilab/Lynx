import os
import sys
import logging
import torch
import numpy as np
import networkx as nx
import scanpy as sc

from sklearn.neighbors import KDTree
from torch.utils.data import ConcatDataset
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Dataset
from torch_geometric.data import ClusterData, HeteroData
from typing import List, Tuple, Union

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import logging

LOGGER = logging.getLogger()


class XeniumDataset(Dataset):
    r"""
    Load Xenium ST graph w/ auxiliary DESI modality

    Parameters
    ----------
    adatas : Union[sc.AnnData, List[sc.AnnData]]
        List of spatial data
    k : int
        Resolution for k-NN graph
    n_subgraphs : int
        Number of graph partition batches per data
    """
    def __init__(
        self,
        adatas : Union[sc.AnnData, List[sc.AnnData]],
        k : int = 30,
        n_subgraphs : int = 8,
        **kwargs
    ):
        super().__init__()

        self.adatas = [adatas] if isinstance(adatas, sc.AnnData) else adatas
        self.k = k
        self.n_subgraphs = n_subgraphs

        # Default graph parameters
        setattr(self, 'r', np.inf)                  # neighbor range (unit: pixel)
        setattr(self, 'is_weighted', False)         # weighted / unweighted k-NN graph
        setattr(self, 'is_hetero', False)           # homogeneous / heterogeneous graph

        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
                LOGGER.info('Update parameter {0} as {1}'.format(key, val))

        # Construct graphs
        self.batches = ConcatDataset(self.load_graphs())

    def load_graphs(self):
        data_list = []

        for i, adata in enumerate(self.adatas):
            LOGGER.info('Constructing graph partitions from data {}'.format(i+1))
            x = torch.tensor(
                adata.X if isinstance(adata.X, np.ndarray) else adata.X.A,
                dtype=torch.float
            )

            u = torch.tensor(
                adata.obsm['X_aux'] if 'X_aux' in adata.obsm.keys() else \
                    np.zeros_like(x),
                dtype=torch.float
            ) 

            coords = adata.obsm['spatial']
            distances, neighbors = self.query_neighbors(coords, coords, k=self.k+1)
            G = self._construct_graph(neighbors, distances)

            data = pyg_utils.from_networkx(G)
            data.x = x
            data.u = u 

            subgraph_data = ClusterData(data, num_parts=self.n_subgraphs, log=False) \
                if self.n_subgraphs > 1 else [data]
            data_list.append(subgraph_data)
        
        return data_list

    def len(self):
        return len(self.batches)
    
    def get(self, idx):
        return self.batches[idx]

    def query_neighbors(
        self,
        ref_coords: Union[np.ndarray, torch.tensor, list],
        query_coords: Union[np.ndarray, torch.tensor, list],
        k: int
    ):
        r"""
        Map k-nearest neighbors of `query_coords` to `ref_coords` using a KDTree
        """
        ref_coords = np.asarray(ref_coords)
        query_coords = np.asarray(query_coords)

        # Check if coordinate dimensions match
        if ref_coords.shape[1] != query_coords.shape[1]:
            raise ValueError("tree_coords must match dim of query_coords.")

        kd_tree = KDTree(ref_coords)
        distances, indices = kd_tree.query(query_coords, k=k)
        return distances, indices
    
    def _construct_graph(self, neighbor_nodes, distances):
        G = nx.Graph()
        n_nodes = neighbor_nodes.shape[0]

        for i in range(n_nodes):
            G.add_node(i, idx=i)
            for j, distance in zip(neighbor_nodes[i], distances[i]):
                no_self_loop = np.logical_or(i != j, self.is_hetero)
                if distance <= self.r and no_self_loop:
                    if self.is_weighted:
                        G.add_edge(i, j, weight=1/distance)
                    else:
                        G.add_edge(i, j)

        return G    
        

class MultiscaleDataset(XeniumDataset):
    r"""
    Load paired multi-modal ST data w/ hybrid resolutions
    """
    def __init__(
        self,
        adatas_ref : Union[sc.AnnData, List[sc.AnnData]],
        adatas_query : Union[sc.AnnData, List[sc.AnnData]],
        k : int = 30,
        n_subgraphs : int = 8,
        **kwargs
    ):
        super().__init__(
            adatas=adatas_ref, k=k, n_subgraphs=n_subgraphs, **kwargs
        )

        self.adatas_ref = [adatas_ref] if isinstance(adatas_ref, sc.AnnData) else adatas_ref
        self.adatas_query = [adatas_query] if isinstance(adatas_query, sc.AnnData) else adatas_query
        self.n_subgraphs = n_subgraphs

        # Labels for ref & query attributes
        setattr(self, 'ref', 'Xenium')                  # `reference` modality name
        setattr(self, 'query', 'DESI')                  # `query` modality name
        setattr(self, 'ref_proj_key', 'desi_map')       # `ref` -> `query` projected spatial coords
        setattr(self, 'query_proj_key', 'xenium_map')   # `query` -> `ref`` projected spatial coords
        setattr(self, 'window_size', 16)                # patch side-length for positional embedding

        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
                LOGGER.info('Update parameter {0} as {1}'.format(key, val))
        
        self.num_windows = 0  # Placeholder for # windows the each node (positional embedding)
        self.hetero_batches = self._load_hetero_graphs()
        del self.batches  # Delete dummy batch initializations
        
    def _load_hetero_graphs(self):
        # Create partitions from hetero graphs
        data_list = []

        for i, (adata_ref, adata_query) in enumerate(zip(self.adatas_ref, self.adatas_query)):
            LOGGER.info('Constructing hetero-graph partitions from paired data {}'.format(i+1))
            
            assert self.ref_proj_key in adata_ref.obsm_keys() and \
                   self.query_proj_key in adata_query.obsm.keys(), \
                "Invalid ref <==> query projection coordinates"
        
            # Retrieve cross-modality k-NN mapping
            ref_coords = adata_ref.obsm['spatial']
            query_coords = adata_query.obsm[self.query_proj_key]
            _, ref_neighbor_indices = self.query_neighbors(ref_coords, query_coords, self.k) # dim: [L, k]

            ref_windows = self.__gen_windows(adata_ref.obsm[self.ref_proj_key], self.window_size)
            query_windows = self.__gen_windows(adata_query.obsm['spatial'], self.window_size)
            self.num_windows = int(max(ref_windows.max(), query_windows.max())) + 1
    
            # Get subgraph index mappings:
            # `*idx` / `*indices`: global index in full expression matrix
            # `*neighbors`: local index (position) in each partition
            for batch in self.batches:
                query_indices = []  
                ref_neighbors = []    # Local `ref` neighbor positions to each query index
                idx_to_position = {idx.item(): pos for pos, idx in enumerate(batch.idx)}
                
                # Iterate through k top reference neighbors to each query index
                for i, indices in enumerate(ref_neighbor_indices):
                    if all(idx in batch.idx.numpy() for idx in indices):
                        query_indices.append(i)
                        ref_neighbors.append([idx_to_position[idx] for idx in indices])
            
                query_expr = adata_query[query_indices].X \
                                if isinstance(adata_query.X, np.ndarray) else \
                                adata_query[query_indices].X.A
                
                # Cross-modality subgraph
                data = HeteroData()

                # (1). query node attributes
                data[self.query].x = torch.tensor(query_expr, dtype=torch.float)
                data[self.query].idx = torch.tensor(query_indices, dtype=torch.long) 
                data[self.query].window = torch.tensor(query_windows[query_indices], dtype=torch.long)
                data[self.query].neighbor = torch.tensor(ref_neighbors, dtype=torch.long) # x -> y

                # (2). ref node attributes
                data[self.ref].x = batch.x
                data[self.ref].idx = batch.idx
                data[self.ref].window = torch.tensor(ref_windows[batch.idx], dtype=torch.long)
                data[self.ref].edge_index = batch.edge_index  # ref-to-ref graph 
                
                # (3). cross-modality edges
                r2q_edge_index, q2r_edge_index = self.__get_hetero_edges(ref_neighbors)
                data[(self.ref, 'to', self.query)].edge_index = r2q_edge_index 
                data[(self.query, 'to', self.ref)].edge_index = q2r_edge_index

                data_list.append(data)

        return data_list
        
    def len(self):
        return len(self.hetero_batches)
    
    def get(self, idx):
        return self.hetero_batches[idx]
    
    @staticmethod
    def __get_hetero_edges(ref_neighbors):
        r"""
        Compute ref -> query & query -> ref edges, 
        `ref_neighbors` dim: [L', k]
        """
        ref_to_query = []
        query_to_ref = []
        n_queries = len(ref_neighbors)
        for i in range(n_queries):
            for j in ref_neighbors[i]:
                ref_to_query.append([j, i])
                query_to_ref.append([i, j])
        
        r2q_ei = torch.tensor(ref_to_query, dtype=torch.long).t().contiguous()
        q2r_ei = torch.tensor(query_to_ref, dtype=torch.long).t().contiguous()
        return r2q_ei, q2r_ei

    @ staticmethod
    def __gen_windows(coords, window_size):
        r"""Compute unique positional embeddings per patch"""
        # Calculate the number of windows in each direction.
        width, height = coords.max(axis=0)
        n_windows_x = int(np.ceil(width / window_size))
        n_windows_y = int(np.ceil(height / window_size))

        # Initialize an array to store window indices for each coordinate.
        window_indices = np.zeros(coords.shape[0], dtype=np.int32)

        # Assign each point to a window index.
        for i, coord in enumerate(coords):
            x, y = coord
            # Calculate window indices for the current point.
            window_x = int(x // window_size)
            window_y = int(y // window_size)
            # Compute a unique index for the window.
            window_index = window_y * n_windows_x + window_x
            window_indices[i] = window_index
        
        return window_indices