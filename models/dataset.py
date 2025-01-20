import os
import sys
import logging
import torch
import numpy as np
import networkx as nx
import scanpy as sc

from scipy.spatial import KDTree
from torch.utils.data import ConcatDataset
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.data import ClusterData, HeteroData
from typing import List, Union

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
        super(XeniumDataset, self).__init__()

        self.adatas = [adatas] if isinstance(adatas, sc.AnnData) else adatas
        self.k = k
        self.n_subgraphs = n_subgraphs

        # Default graph parameters
        setattr(self, 'r', np.inf)            # neighbor range (unit: pixel)
        setattr(self, 'is_weighted', False)   # weighted / unweighted k-NN graph
        setattr(self, 'is_hetero', False)     # single-modal vs. multi-modal
        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
                LOGGER.info('Update parameter {0} as {1}'.format(key, val))

        # Construct graphs
        self.batches = ConcatDataset(self._load_graphs())

    def _load_graphs(self):
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

            s = torch.tensor(
                adata.obsm['X_s'] if 'X_s' in adata.obsm.keys() else \
                    np.empty(shape=(adata.shape[0], 0)),
                dtype=torch.float
            )

            coords = adata.obsm['spatial']
            distances, neighbors = self.query_neighbors(coords, coords)
            distances, neighbors = distances[:, 1:], neighbors[:, 1:]
            G = self._construct_graph(neighbors, distances)

            data = pyg_utils.from_networkx(G)
            data.x = x
            data.u = u 
            data.s = s

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
    ):
        r"""
        Map k-nearest neighbors of `query_coords` to `ref_coords` using a KDTree
        """
        ref_coords = np.asarray(ref_coords)
        query_coords = np.asarray(query_coords)
        k = self.k if self.is_hetero else self.k+1  # Avoid self for homogeneous graph

        # Check if coordinate dimensions match
        if ref_coords.shape[1] != query_coords.shape[1]:
            raise ValueError("tree_coords must match dim of query_coords.")

        kd_tree = KDTree(ref_coords)
        distances, indices = kd_tree.query(query_coords, k=k)
        return distances, indices
    
    def _construct_graph(self, neighbor_nodes, distances):
        G = nx.Graph()
        n_nodes = len(neighbor_nodes)

        for i in range(n_nodes):
            G.add_node(i, idx=i)
            for j, distance in zip(neighbor_nodes[i], distances[i]):
                if self.r == np.inf or distance <= self.r:
                    if self.is_weighted:
                        G.add_edge(i, j, weight=1/distance)
                    else:
                        G.add_edge(i, j)

        return G     
    

class HeteroDataset(XeniumDataset):
    r"""
    Load paired multi-modal ST data into a joint heterogeneous graph
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
            adatas=adatas_query, k=k, n_subgraphs=n_subgraphs, 
            is_hetero=True, **kwargs
        )

        self.n_subgraphs = n_subgraphs
        
        self.adatas_ref = [adatas_ref] \
            if isinstance(adatas_ref, sc.AnnData) \
            else adatas_ref
        
        self.adatas_query = [adatas_query] \
            if isinstance(adatas_query, sc.AnnData) \
            else adatas_query

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
        del self.batches  # Delete the dummy batch for query partitions
        
    def _load_hetero_graphs(self):
        # Create partitions from hetero graphs
        data_list = []

        for i, (adata_ref, adata_query) in enumerate(zip(self.adatas_ref, self.adatas_query)):
            LOGGER.info('Constructing hetero-graph partitions from paired data {}'.format(i+1))
            
            assert self.ref_pos_key in adata_ref.obsm_keys(), \
                "Invalid `adata.obsm[{}]` access for coords".format(self.ref_pos_key)
            assert self.query_pos_key in adata_query.obsm_keys(), \
                "Invalid `adata.obsm[{}]` access for coords".format(self.query_pos_key)
        
            ref_coords = adata_ref.obsm[self.ref_pos_key]

            # Build hetero subgraphs from each query partition
            for batch in self.batches:
                # Retrieve `ref` neighbors to each query node
                # *_indices: absolute index from full expression 
                # *_nodes: relative  index in each subgraph partition
                query_coords = adata_query[batch.idx.numpy()].obsm[self.query_pos_key]
                distances, neighbors = self.query_neighbors(ref_coords, query_coords)
                
                query_nodes = self._convert_to_rank(batch.idx.numpy())
                ref_indices, neighbor_nodes = np.unique(neighbors, return_inverse=True)
                neighbor_nodes = neighbor_nodes.reshape(len(batch.idx), -1)  # dim: [L, # neighbors]

                # Build subgraph
                data = HeteroData()

                # Update node features
                x_ref = torch.tensor(
                    adata_ref[ref_indices].X \
                        if isinstance(adata_ref.X, np.ndarray) \
                        else adata_ref[ref_indices].X.A,
                    dtype=torch.float
                )
                data[self.ref].x = x_ref
                data[self.ref].idx = torch.tensor(ref_indices, dtype=torch.long)
                
                x_query = batch.x
                data[self.query].x = x_query
                data[self.query].idx = batch.idx

                # Update reference -> query edges
                edge_index = self._construct_hetero_graph(query_nodes, neighbor_nodes, distances)
                self.edge = (self.ref, 'to', self.query)
                data[self.edge].edge_index = edge_index
                
                data_list.append(data)

        return data_list
        
    def len(self):
        return len(self.hetero_batches)
    
    def get(self, idx):
        return self.hetero_batches[idx]
        
    def _construct_hetero_graph(self, query_nodes, neighbor_nodes, distances):
        r"""Compute directed ref -> query edges"""
        ei = []
        for i, query_node in enumerate(query_nodes):    
            for ref_node, distance in zip(neighbor_nodes[i], distances[i]):
                if self.r == np.inf or distance <= self.r:
                    ei.append([ref_node, query_node])
        return torch.tensor(ei, dtype=torch.long).t().contiguous()
    
    @staticmethod
    def _convert_to_rank(arr):
        r"""
        Reassign the array values by its rank to get 
        'contiguous' value ranges
            e.g. [5, 8, 8, 2, 3] -> [2, 3, 3, 0, 1]
        """
        _, inverse = np.unique(arr, return_inverse=True)
        return inverse.reshape(arr.shape)


    
