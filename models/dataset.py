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
from torch_geometric.data import Batch, Data, Dataset
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
        setattr(self, 'r', np.inf)            # neighbor range (unit: pixel)
        setattr(self, 'is_weighted', False)   # weighted / unweighted k-NN graph
        setattr(self, 'get_batches', True)    # Build k-NN graph upon intialization
        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
                LOGGER.info('Update parameter {0} as {1}'.format(key, val))

        # Construct graphs
        if self.get_batches:
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
            distances, neighbors = distances[:, 1:], neighbors[:, 1:]
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
                if self.r == np.inf or distance <= self.r:
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
        setattr(self, 'ref', 'Xenium')
        setattr(self, 'query', 'DESI')
        setattr(self, 'ref_pos_key', 'spatial')
        setattr(self, 'query_pos_key', 'xenium_map')

        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
                LOGGER.info('Update parameter {0} as {1}'.format(key, val))

        self.hetero_batches = self._load_hetero_graphs()
        del self.batches  # Delete dummy batch initializations
        
    def _load_hetero_graphs(self):
        # Create partitions from hetero graphs
        data_list = []

        for i, (adata_ref, adata_query) in enumerate(zip(self.adatas_ref, self.adatas_query)):
            LOGGER.info('Constructing hetero-graph partitions from paired data {}'.format(i+1))
            
            assert self.ref_pos_key in adata_ref.obsm_keys(), \
                "Invalid `adata.obsm[{}]` access for coords".format(self.ref_pos_key)
            assert self.query_pos_key in adata_query.obsm_keys(), \
                "Invalid `adata.obsm[{}]` access for coords".format(self.query_pos_key)
        
            # Retrieve cross-modality k-NN mapping
            ref_coords = adata_ref.obsm[self.ref_pos_key]
            query_coords = adata_query.obsm[self.query_pos_key]
            _, query_neighbors = self.query_neighbors(ref_coords, query_coords, self.k) # dim: [L, k]

            # Update hetero subgraphs from each `ref` partitions 
            for batch in self.batches:
                
                # Get reference neighbor indices (convert to consective for each subgraph)
                query_indices = []
                for i, nbrs in enumerate(query_neighbors):
                    if all(idx in batch.idx.numpy() for idx in nbrs):
                        query_indices.append(i)
                
                batch_neighbors = query_neighbors[query_indices]
                _, neighbors = np.unique(batch_neighbors, return_inverse=True)
                neighbors = neighbors.reshape(batch_neighbors.shape)
                query_expr = adata_query[query_indices].X \
                                if isinstance(adata_query.X, np.ndarray) else \
                                adata_query[query_indices].X.A
                
                # Build cross-modal subgraph
                data = HeteroData()

                data[self.query].x = torch.tensor(query_expr, dtype=torch.float)
                data[self.query].idx = torch.tensor(query_indices, dtype=torch.long)
                data[self.query].neighbors = torch.tensor(neighbors, dtype=torch.long)  

                # data[('Xenium', 'to', 'DESI')].edge_index

                data[self.ref].x = batch.x
                data[self.ref].idx = batch.idx
                data[self.ref].edge_index = batch.edge_index  # ref-to-ref graph 

                data_list.append(data)

        return data_list
        
    def len(self):
        return len(self.hetero_batches)
    
    def get(self, idx):
        return self.hetero_batches[idx]
    