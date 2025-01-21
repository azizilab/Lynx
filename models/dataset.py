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
from torch_geometric.data import ClusterData
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
        super(XeniumDataset, self).__init__()

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

            s = torch.tensor(
                adata.obsm['X_s'] if 'X_s' in adata.obsm.keys() else \
                    np.empty(shape=(adata.shape[0], 0)),
                dtype=torch.float
            )

            coords = adata.obsm['spatial']
            distances, neighbors = self.query_neighbors(coords, coords, k=self.k+1)
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
    

class MultimodalDataset(XeniumDataset):
    r"""
    Load paired multi-modal ST data w/ hybrid resolutions
    """
    def __init__(
        self,
        adatas_ref: Union[sc.AnnData, List[sc.AnnData]],
        adatas_query: Union[sc.AnnData, List[sc.AnnData]],
        n_subgraphs : int = 8,
        k : int = 10,
        **kwargs
    ):
        super().__init__(
            adatas=adatas_ref, k=k, n_subgraphs=n_subgraphs, get_batches=False, **kwargs
        )

        self.adatas_ref = [adatas_ref] \
            if isinstance(adatas_ref, sc.AnnData) \
            else adatas_ref
        
        self.adatas_query = [adatas_query] \
            if isinstance(adatas_query, sc.AnnData) \
            else adatas_query

        # Labels for ref & query attributes
        setattr(self, 'ref', 'Xenium')
        setattr(self, 'query', 'DESI')
        setattr(self, 'ref_coord_key', 'spatial')
        setattr(self, 'query_coord_key', 'xenium_map')

        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
                LOGGER.info('Update parameter {0} as {1}'.format(key, val))

        self.batches = ConcatDataset(self.load_graphs())

    def load_graphs(self):
        """
        Compute 2D subgraphs from a list of paired hires (`ref`) + lowres (`query`) spatial data
        """
        data_list = []
        for adata_ref, adata_query in zip(self.adatas_ref, self.adatas_query):
            
            assert self.query_coord_key in adata_query.obsm_keys(), \
                "Please compute lowres (query) -> hires (ref) spatial mapping first!"
            
            LOGGER.info('Constructing multi-scale graph...')

            ref_coords = adata_ref.obsm[self.ref_coord_key]
            query_coords = adata_query.obsm[self.query_coord_key]

            # Build graph for `ref` modality
            distances, neighbors = self.query_neighbors(ref_coords, ref_coords, k=self.k+1)
            distances, neighbors = distances[:, 1:], neighbors[:, 1:]
            graph = self._construct_graph(neighbors, distances)

            # Get cross-modal k-NNs from ref -> query    
            _, adata_query.obsm['neighbors'] = self.query_neighbors(ref_coords, query_coords, self.k)

            # Hi-res expression observation
            expr = adata_ref.X \
                if isinstance(adata_ref.X, np.ndarray) else \
                adata_ref.X.A
            
            # Hi-res indices
            ref_indices = np.arange(expr.shape[0])
            graph_data = pyg_utils.from_networkx(graph)
            graph_data.x = torch.tensor(expr).float()
            graph_data.xenium_idx = torch.tensor(ref_indices).long()

            # Create partitioned subgraphs
            LOGGER.info('Partitioning into {} subgraphs...'.format(self.n_subgraphs))
            cluster_data = ClusterData(graph_data, num_parts=self.n_subgraphs) \
                if self.n_subgraphs > 1 else [graph_data]
            
            # Append mapped low-res expressions to subgraphs
            subgraphs = [
                data.update({
                    'y': y, 
                    'neighbors': neighbors,
                    'desi_idx': idx
                })
                for data in cluster_data
                for y, neighbors, idx in [self.__get_lowres_expr(data, adata_query)]
            ]
            data_list.append(Batch.from_data_list(subgraphs))

        return data_list
    
    def __get_lowres_expr(self, data: Data, adata_query: sc.AnnData):
        r"""
        Compute paired lowres expressions to each subgraph partition.

        Returns:
            Tuple: (y, neighbors), where:
                - y: A subset of adata_query where all neighbors are within data.idx.
                - neighbors: The corresponding neighbors idx in index space of data.
                - idx: Original indices corresponding to adata_query
        """
        #index to position in data
        idx_to_position = {idx.item(): pos for pos, idx in enumerate(data.xenium_idx)}

        neighbors = adata_query.obsm['neighbors']  # Assumes this is a 2D array or list of lists

        # Identify rows where all neighbors are within hires_idx
        valid_neighbors = []
        valid_indices = []

        for i, neighbor_indices in enumerate(neighbors):
            if all(idx in data.xenium_idx for idx in neighbor_indices):
                valid_indices.append(i)
                # Remap neighbor indices to data positions
                valid_neighbors.append([idx_to_position[idx] for idx in neighbor_indices])

        # Subset adata_query
        subset_adata_query = adata_query[valid_indices].X \
                                if isinstance(adata_query.X, np.ndarray) else \
                                adata_query[valid_indices].X.A
        
        return (
            torch.tensor(subset_adata_query), 
            torch.tensor(valid_neighbors, dtype=torch.long), 
            torch.tensor(valid_indices, dtype=torch.long)
        )
