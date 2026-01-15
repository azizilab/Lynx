import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.neighbors import KDTree
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.loader import ClusterData 
from typing import List, Tuple, Union

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.utils import to_dense_array
import logging
LOGGER = logging.getLogger()


EPS = 1e-8


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
        adatas: Union[sc.AnnData, List[sc.AnnData]],
        k: int = 8,
        r: float = 50,
        n_subgraphs: int = 8,
        is_grid: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__()

        self.adatas = [adatas] if isinstance(adatas, sc.AnnData) else adatas
        self.k = k
        self.r = r
        self.n_subgraphs = n_subgraphs
        self.is_grid = is_grid
        self.verbose = verbose

        # Default graph parameters
        setattr(self, 'is_weighted', False)         # weighted / unweighted k-NN graph
        setattr(self, 'num_clusters', 0)            # Placeholder to max # clusters  
        setattr(self, 'cluster_key', None)          # Placeholder for cluster label key (`adata.obs`)

        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)

        # Construct graphs
        self.batches = ConcatDataset(self.load_graphs())

    def load_graphs(self):
        data_list = []

        for i, adata in enumerate(self.adatas):
            if self.verbose:
                LOGGER.info('Constructing graph partitions from data {}'.format(i+1))
            x = torch.tensor(to_dense_array(adata.X), dtype=torch.float)
            coords = adata.obsm['spatial']
        
            # Append clustering profile (do Leiden clustering if empty)
            adata_normed = sc.pp.normalize_total(adata, target_sum=None, copy=True)
            if self.cluster_key in adata.obs.keys():
                adata.obs.loc[:, 'leiden'] = pd.Categorical(adata.obs[self.cluster_key]).codes
            else:
                sc.pp.log1p(adata_normed)
                sc.pp.pca(adata_normed)
                sc.pp.neighbors(adata_normed)
                sc.tl.leiden(adata_normed, flavor='igraph', resolution=0.5, random_state=42)
                adata.obs.loc[:, 'leiden'] = adata_normed.obs['leiden'].values.astype(np.int32)

            # Construct neighbor graph
            clusters = adata.obs.leiden.to_numpy().astype(np.int32)
            self.num_clusters = clusters.max()+1
            distances, neighbors = self.get_neighbors(coords, coords, is_grid=self.is_grid, k=self.k, r=self.r)
            edge_index, edge_weight = self.construct_graph(neighbors, distances, clusters)

            data = Data(x=x, edge_index=edge_index, idx=torch.arange(len(x)))
            if self.is_weighted:
                data.edge_attr = edge_weight

            # Add bulk cluster-specific expression profile
            data.cluster = torch.tensor(clusters, dtype=torch.long)
            # data.bulk_clu = torch.stack([
            #     torch.tensor(adata_normed[adata.obs.leiden==k].X.mean(0)).reshape(-1) \
            #     for k in range(self.num_clusters)
            # ]).float()
            
            subgraph_data = ClusterData(data, num_parts=self.n_subgraphs, log=False) \
                            if self.n_subgraphs > 1 else [data]

            data_list.append(subgraph_data)
        
        return data_list

    def len(self):
        return len(self.batches)
    
    def get(self, idx):
        return self.batches[idx]

    @staticmethod
    def get_neighbors(
        ref_coords: Union[np.ndarray, torch.tensor, list],
        query_coords: Union[np.ndarray, torch.tensor, list],
        is_grid: bool = False,
        k: float = 8,
        r: float = 50,
    ):
        r"""
        Retrieve k-nearest-neighbor (or radius-bounded neighbors) of 
        `query_coords` to `ref_coords` using a KDTree
        """
        assert (k is not None) or (r is not None), \
            "Either k or r should be provided for spatial NN-graph."
        
        ref_coords = np.asarray(ref_coords)
        query_coords = np.asarray(query_coords)

        # Check if coordinate dimensions match
        if ref_coords.shape[1] != query_coords.shape[1]:
            raise ValueError("tree_coords must match dim of query_coords.")
        
        kd_tree = KDTree(ref_coords)
        if is_grid:
            distances, indices = kd_tree.query(query_coords, k=k+1)
        else:
            indices, distances = kd_tree.query_radius(query_coords, r, return_distance=True)
        return distances, indices
    
    def construct_graph(
        self,
        neighbor_nodes: Union[np.ndarray, torch.tensor, list], 
        distances: Union[np.ndarray, torch.tensor, list], 
        cluster_labels: np.ndarray = None
    ):
        r"""Compute undirected graph edges & attributes"""
        n_nodes = neighbor_nodes.shape[0]
        edge_index = []
        edge_weight = []
        
        for i in range(n_nodes):
            for j, distance in zip(neighbor_nodes[i], distances[i]):
                # Avoid self-loops
                if (i != j):
                    edge_index.append([j, i])
                    edge_weight.append(distance)
                        
        ei = torch.tensor(edge_index,  dtype=torch.long).t().contiguous()
        ew = torch.tensor(edge_weight, dtype=torch.float)
        ew = ew/ew.median()

        return ei, ew


class HeteroDataset(XeniumDataset):
    r"""
    Load paired multi-modal ST data w/ hybrid resolutions into a hetero-graph
    """
    def __init__(
        self,
        adatas_ref: Union[sc.AnnData, List[sc.AnnData]],
        adatas_query: Union[sc.AnnData, List[sc.AnnData]],
        k: int = 8, 
        r: float = 50.,
        r_bigraph: float = 30.,
        n_subgraphs: int = 8,
        is_query_grid: bool = True,
        is_ref_grid: bool = False,
        **kwargs
    ):
        super().__init__(
            adatas=adatas_ref, k=k, r=r, 
            n_subgraphs=n_subgraphs, 
            is_grid=is_ref_grid,
            **kwargs
        )
        self.k = k
        self.r = r
        self.r_bigraph = r_bigraph  # Cross-modality neighbor radius
        self.is_query_grid = is_query_grid
        self.is_ref_grid = is_ref_grid
        self.adatas_ref = [adatas_ref] if isinstance(adatas_ref, sc.AnnData) else adatas_ref
        self.adatas_query = [adatas_query] if isinstance(adatas_query, sc.AnnData) else adatas_query
        self.n_subgraphs = n_subgraphs

        # Labels for ref & query attributes
        setattr(self, 'ref', 'Xenium')                  # `reference` modality name
        setattr(self, 'query', 'DESI')                  # `query` modality name
        setattr(self, 'ref_proj_key', 'desi_map')       # `ref` -> `query` projected spatial coords
        setattr(self, 'query_proj_key', 'xenium_map')   # `query` -> `ref` projected spatial coords

        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)

        self.hetero_batches = self._load_hetero_graphs()
        
    def _load_hetero_graphs(self):
        r"""Create partitions from multi-omics (ref <-> query) heterographs"""
        data_list = []
   
        for i, (adata_ref, adata_query) in enumerate(zip(self.adatas_ref, self.adatas_query)):
            if self.verbose:
                LOGGER.info('Constructing hetero-graph partitions from paired data {}'.format(i+1))
            
            assert self.ref_proj_key in adata_ref.obsm_keys() and \
                   self.query_proj_key in adata_query.obsm.keys(), \
                "Invalid ref <==> query projection coordinates"
        
            # Retrieve cross-modality neighbor mapping: # dim: ([L, k], [L, K])
            ref_coords = adata_ref.obsm['spatial']
            query_coords = adata_query.obsm[self.query_proj_key]

            # Fixed radius for cross-modality neighbors 
            distances, ref_neighbor_indices = self.get_neighbors(
                ref_coords, query_coords, r=self.r_bigraph
            )  

            # Get subgraph index mappings:
            # `*idx` / `*indices`: global index in full expression matrix
            # `*neighbors`: local index (position) in each graph partition
            batch_idxl = i * self.n_subgraphs
            batch_idxr = batch_idxl + self.n_subgraphs
            for batch_idx in range(batch_idxl, batch_idxr):
                batch = self.batches[batch_idx]
                query_indices = []  
                ref_neighbors = []    # Local `ref` neighbor positions to each query index
                idx_to_position = {idx.item(): pos for pos, idx in enumerate(batch.idx)}

                # Iterate through k top reference neighbors to each query index
                for i, indices in enumerate(ref_neighbor_indices):
                    if all(idx in batch.idx.numpy() for idx in indices):
                        query_indices.append(i)
                        ref_neighbors.append([idx_to_position[idx] for idx in indices])
                query_expr = to_dense_array(adata_query[query_indices].X)

                # Cross-modality subgraph
                data = HeteroData()

                # (1). query node attributes
                data[self.query].x = torch.tensor(query_expr, dtype=torch.float)
                data[self.query].idx = torch.tensor(query_indices, dtype=torch.long) 

                # (2). ref node attributes
                data[self.ref].x = batch.x
                data[self.ref].idx = batch.idx
                data[self.ref].cluster = batch.cluster
                # data[self.ref].bulk_clu = batch.bulk_clu

                # (3). edges (within-modal & cross-modal)
                #  - (i). ref-to-ref graph
                data[self.ref, 'to', self.ref].edge_index = batch.edge_index

                #  - (ii). query-to-query graph
                query_coords = adata_query[query_indices].obsm['spatial']
                q2q_distances, q2q_neighbors = self.get_neighbors(
                    query_coords, query_coords, 
                    is_grid=self.is_query_grid,
                    k=self.k, r= self.r
                )

                q2q_ei, _ = self.construct_graph(q2q_neighbors, q2q_distances)
                data[(self.query, 'to', self.query)].edge_index = q2q_ei

                #  - (iii). ref-to-query & query-to-ref graph 
                r2q_ei, q2r_ei = self.construct_hetero_graph(
                    ref_neighbors, 
                    distances[query_indices]
                )
                data[(self.ref, 'to', self.query)].edge_index = r2q_ei
                data[(self.query, 'to', self.ref)].edge_index = q2r_ei

                # - (iv). edge weights, append hetero graph batch
                data[(self.ref, 'to', self.ref)].edge_attr = batch.edge_attr
                data_list.append(data)

        
         # Delete dummy batch object from initialization
        del self.batches 
        return data_list
        
    def len(self):
        return len(self.hetero_batches)
    
    def get(self, idx):
        return self.hetero_batches[idx]
    
    def construct_hetero_graph(self, ref_neighbors, distances):
        r"""
        Compute ref -> query & query -> ref edges & attributes 
        to construct hetero-graph, `ref_neighbors` - (dim: [L', k])
        """
        ref_to_query = []
        query_to_ref = []
        n_queries = len(ref_neighbors)
        
        for i in range(n_queries):
            for j, distance in zip(ref_neighbors[i], distances[i]):
                if distance < self.r:
                    ref_to_query.append([j, i])
                    query_to_ref.append([i, j])
        
        r2q_ei = torch.tensor(ref_to_query, dtype=torch.long).t().contiguous()
        q2r_ei = torch.tensor(query_to_ref, dtype=torch.long).t().contiguous()
        return r2q_ei, q2r_ei