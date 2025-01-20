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


    

class MultiscaleDatasetTest(XeniumDataset):
    r"""
    Load paired & aligned ST graphs & feature matrices w/ hybrid resolutions
    """
    def __init__(
        self,
        n_subgraphs : int = 8,
        **kwargs
    ):
        super(MultiscaleDataset, self).__init__(n_subgraphs, **kwargs)
        self.coord_to_cluster = None
        self.cluster_to_expr = None

    def load_graphs(
        self, 
        hires_adatas: List[sc.AnnData], 
        lowres_adatas: List[sc.AnnData]
    ):
        """
        Compute 2D subgraphs from a list of paired hires + lowres spatial data
        """
        data_list = []
        for adata_hires, adata_lowres in zip(hires_adatas, lowres_adatas):
            assert 'desi_map' in adata_hires.obsm_keys(), \
                "Please compute hires -> lowres spatial mapping first"
            
            LOGGER.info('Constructing multi-scale graph...')
            maps = self.__get_pooling_maps(adata_hires, adata_lowres)
            self.coord_to_cluster, self.cluster_to_expr = maps
            
            graph = construct_graph(
                self.get_coords(adata_hires),
                k=self.k, r=self.r, weighted=self.weighted
            )

            # Hi-res expression observation
            expr = adata_hires.X \
                if isinstance(adata_hires.X, np.ndarray) else \
                adata_hires.X.A
            
            # Hi-res coords => lowres cluster assignments
            cluster = [
                self.coord_to_cluster[tuple(coord)]
                for coord in adata_hires.obsm['desi_map']
            ]

            # Covariate
            s = adata_hires.obsm['X_s'] \
                if 'X_s' in adata_hires.obsm.keys() else \
                np.empty(shape=(adata_hires.shape[0], 0))
            
            graph_data = pyg_utils.from_networkx(graph)
            graph_data.x = torch.tensor(expr).float()
            graph_data.cluster = torch.tensor(cluster)
            graph_data.s = torch.tensor(s).float()

            # Create partitioned subgraphs
            LOGGER.info('Partitioning into {} subgraphs...'.format(self.n_subgraphs))
            cluster_data = ClusterData(graph_data, num_parts=self.n_subgraphs) \
                if self.n_subgraphs > 1 else [graph_data]
            
            # Append mapped low-res expressions to subgraphs
            subgraphs = [
                data.update({'y': self.__get_lowres_expr(data)})
                for data in cluster_data
            ]
            data_list.append(Batch.from_data_list(subgraphs))

        return ConcatDataset(data_list)
    
    def __get_pooling_maps(
        self,
        adata_hires: sc.AnnData, 
        adata_lowres: sc.AnnData
    ):
        r"""Compute dictionaries for multiscale feature maps
        - (1). Low-res coord  => low-res cluster ID
        - (2). Low-res cluster ID => low-res expressions
        """
        cluster_id = 0
        coord_to_cluster = {}
        cluster_to_expr = {}

        for coord in adata_hires.obsm['desi_map']:
            coord = tuple(coord)
            if coord not in coord_to_cluster:
                coord_to_cluster[coord] = cluster_id
                cluster_id += 1

        for adata_ in adata_lowres:
            coord = tuple(adata_.obsm['spatial'].squeeze())
            cluster_id = coord_to_cluster[coord]
            cluster_to_expr[cluster_id] = np.asarray(adata_.X.squeeze()) \
                if isinstance(adata_.X, np.ndarray) else \
                np.asarray(adata_.X.A.squeeze())
            
        return coord_to_cluster, cluster_to_expr
    
    def __get_lowres_expr(self, data: Data):
        r"""
        Compute paired lowres expressions to each subgraph partition
        ordered by cluster IDs 1,...,K
        """
        cluster_ids = np.unique([cluster_id.item() for cluster_id in data.cluster])
        expr = torch.tensor([self.cluster_to_expr[c] for c in cluster_ids]).float()
        return expr
    

class MultiscaleDatasetJosh(XeniumDataset):
    r"""
    Load paired & aligned ST graphs & feature matrices w/ hybrid resolutions
    """
    def __init__(
        self,
        n_subgraphs : int = 8,
        k : int = 10,
        **kwargs
    ):
        super(MultiscaleDatasetJosh, self).__init__(n_subgraphs, **kwargs)

        self.k = k

    def load_graphs(
        self, 
        hires_adatas: List[sc.AnnData], 
        lowres_adatas: List[sc.AnnData],
    ):
        """
        Compute 2D subgraphs from a list of paired hires + lowres spatial data
        """
        data_list = []
        for adata_hires, adata_lowres in zip(hires_adatas, lowres_adatas):
            # assert 'desi_map' in adata_hires.obsm_keys(), \
            #     "Please compute hires -> lowres spatial mapping first"
            
            assert 'xenium_map' in adata_lowres.obsm_keys(), \
                "Please compute lowres -> hires spatial mapping first!"
            
            LOGGER.info('Constructing multi-scale graph...')
            # maps = self.__get_pooling_maps(adata_hires, adata_lowres)
            # self.coord_to_cluster, self.cluster_to_expr = maps

            adata_lowres.obsm['neighbors'] = self.query_neighbors(adata_hires.obsm['spatial'], adata_lowres.obsm['xenium_map'], self.k)
            
            graph = construct_graph(
                self.get_coords(adata_hires),
                k=self.k, r=self.r, weighted=self.weighted
            )

            # Hi-res expression observation
            expr = adata_hires.X \
                if isinstance(adata_hires.X, np.ndarray) else \
                adata_hires.X.A
            
            # Hi-res indices
            hires_idx = np.arange(expr.shape[0])
            
            # Hi-res coords => lowres cluster assignments
            # cluster = [
            #     self.coord_to_cluster[tuple(coord)]
            #     for coord in adata_hires.obsm['desi_map']
            # ]

            # Covariate
            # s = adata_hires.obsm['X_s'] \
            #     if 'X_s' in adata_hires.obsm.keys() else \
            #     np.empty(shape=(adata_hires.shape[0], 0))
            
            graph_data = pyg_utils.from_networkx(graph)
            graph_data.x = torch.tensor(expr).float()
            graph_data.xenium_idx = torch.tensor(hires_idx).long()
            # graph_data.xenium_coords = torch.tensor(adata_hires.obsm['spatial'])
            # graph_data.desi_map = torch.tensor(adata_hires.obsm['desi_map'])
            # graph_data.cluster = torch.tensor(cluster)
            # graph_data.s = torch.tensor(s).float()

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
                # 'desi_coords': coords,
                # 'desi_pixels' : pixels
            })
            for data in cluster_data
            for y, neighbors, idx in [self.__get_lowres_expr(data, adata_lowres)]
            ]

            # subgraphs = [data for data in subgraphs if data.remove_tensor('idx')] #get rid of this to avoid confusion

            data_list.append(Batch.from_data_list(subgraphs))

        return ConcatDataset(data_list)
    
    def query_neighbors(
        self,
        tree_coords: Union[np.ndarray, torch.tensor, list], 
        query_coords: Union[np.ndarray, torch.tensor, list], 
        k: int
    ) -> np.ndarray:
        """
        Map k-nearest neighbors of query_coords to tree_coords using a KDTree.

        Parameters:
            tree_coords: Array of shape (n_samples, dim) with tree coordinates.
            query_coords: Array of shape (m_samples, dim) with query coordinates.
            k (int): Number of nearest neighbors to find for each point in query_coords.

        Returns:
            np.ndarray: Indices of k-nearest neighbors for each point in query_coords.
        """
        tree_coords = np.asarray(tree_coords)
        query_coords = np.asarray(query_coords)
        
        # Check if dimensions are suitable for KDTree (nxdim and mxdim)
        if tree_coords.ndim != 2:
            raise ValueError("tree_coords must be of shape (n_samples, dim).")
        if query_coords.ndim != 2:
            raise ValueError("query_coords must be of shape (m_samples, dim).")
        if tree_coords.shape[1] != query_coords.shape[1]:
            raise ValueError("tree_coords must match dim of query_coords.")

        # Construct KDTree
        kd_tree = KDTree(tree_coords)

        # Query KDTree
        _, indices = kd_tree.query(query_coords, k=k)
        
        return indices
    
    def __get_lowres_expr(self, data: Data, lowres_adata: sc.AnnData):
        r"""
        Compute paired lowres expressions to each subgraph partition.

        
        Returns:
            Tuple: (y, neighbors), where:
                - y: A subset of lowres_adata where all neighbors are within data.idx.
                - neighbors: The corresponding neighbors idx in index space of data.
                - idx: Original indices corresponding to lowres_adata
        """
        #index to position in data
        idx_to_position = {idx.item(): pos for pos, idx in enumerate(data.xenium_idx)}

        neighbors = lowres_adata.obsm['neighbors']  # Assumes this is a 2D array or list of lists

        # Identify rows where all neighbors are within hires_idx
        valid_neighbors = []
        valid_indices = []

        for i, neighbor_indices in enumerate(neighbors):
            if all(idx in data.xenium_idx for idx in neighbor_indices):
                valid_indices.append(i)
                # Remap neighbor indices to data positions
                valid_neighbors.append([idx_to_position[idx] for idx in neighbor_indices])

        # Subset lowres_adata
        subset_lowres_adata = lowres_adata[valid_indices].X \
                                if isinstance(lowres_adata.X, np.ndarray) else \
                                lowres_adata[valid_indices].X.A
        
        return torch.tensor(subset_lowres_adata), torch.tensor(valid_neighbors, dtype=torch.long), torch.tensor(valid_indices, dtype=torch.long)
