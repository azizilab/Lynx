import os
import sys
import logging
import torch
import numpy as np
import scanpy as sc

from sklearn.neighbors import KDTree
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.data import ClusterData, HeteroData
from typing import List, Tuple, Union

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import logging
LOGGER = logging.getLogger()

def to_dense_array(x):
    return x if isinstance(x, np.ndarray) else x.A


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
        k : int = 15,
        n_subgraphs : int = 8,
        receptor_ids : List[str] = [],
        ligand_ids : List[str] = [],
        **kwargs
    ):
        super().__init__()

        self.adatas = [adatas] if isinstance(adatas, sc.AnnData) else adatas
        self.k = k
        self.n_subgraphs = n_subgraphs
        self.receptor_ids = receptor_ids
        self.ligand_ids = ligand_ids

        # Default graph parameters
        setattr(self, 'r', np.inf)                  # neighbor range (unit: pixel)
        setattr(self, 'sigma', 25.)                 # standard deviation term for RBF kernel
        setattr(self, 'is_weighted', False)         # weighted / unweighted k-NN graph
        setattr(self, 'num_clusters', 0)            # Placeholder to max # clusters 
        setattr(self, 'cluster', None)              # Placeholder for cluster IDs

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
            x = torch.tensor(to_dense_array(adata.X), dtype=torch.float)
            coords = adata.obsm['spatial']
            distances, neighbors = self.query_neighbors(coords, coords, k=self.k+1, use_radius=False)
            edge_index, edge_weight = self.construct_graph(neighbors, distances, allow_self_loop=False)

            data = Data(x=x, edge_index=edge_index, idx=torch.arange(len(x)))
            
            if 'X_aux' in adata.obsm.keys():
                data.u = torch.tensor(adata.obsm['X_aux'], dtype=torch.float)

            if self.is_weighted:
                data.edge_attr = edge_weight
            
            receptors = adata.var_names.intersection(self.receptor_ids)
            ligands = adata.var_names.intersection(self.ligand_ids)

            data.receptors = torch.tensor(adata.var_names.get_indexer(receptors), dtype=torch.long)
            data.ligands = torch.tensor(adata.var_names.get_indexer(ligands), dtype=torch.long)

            # Append clustering profile
            if self.cluster is None and 'leiden' not in adata.obs.keys():
                adata_norm = adata.copy()
                sc.pp.normalize_total(adata_norm)
                sc.pp.log1p(adata_norm)

                sc.pp.pca(adata_norm)
                sc.pp.neighbors(adata_norm)
                sc.tl.leiden(adata_norm, random_state=42) 
                adata.obs['leiden'] = adata_norm.obs['leiden'].copy()
                del adata_norm  

            clusters = adata.obs.leiden.to_numpy().astype(np.int32)
            self.num_clusters = clusters.max()+1
            self.cluster = torch.tensor(clusters, dtype=torch.long)
            data.cluster = torch.tensor(clusters, dtype=torch.long)
            normed = sc.pp.normalize_total(adata, target_sum=None, copy=True)
            data.bulk_clu = torch.stack([torch.tensor(normed[normed.obs.leiden==str(k)].X.mean(0)).reshape(-1) \
                                         if str(k) in normed.obs.leiden.unique() else \
                                          torch.zeros(normed.shape[-1]) for k in range(normed.obs['leiden'].astype(int).max()+1)])

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
        k,
        use_radius: bool = False
        
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
        if use_radius:
            indices, distances = kd_tree.query_radius(query_coords, k, return_distance=True)
        else:
            distances, indices = kd_tree.query(query_coords, k=k)
        return distances, indices
    
    def construct_graph(self, neighbor_nodes, distances, allow_self_loop):
        r"""Compute undirected graph edges & attributes"""
        n_nodes = neighbor_nodes.shape[0]
        edge_index = []
        edge_weight = []
        
        for i in range(n_nodes):
            for j, distance in zip(neighbor_nodes[i], distances[i]):
                # if distance <= self.r and i != j:
                if i != j or allow_self_loop:
                    edge_index.append([j, i])
                    # edge_weight.append(self.dist_to_rbf(distance, self.sigma))
                    edge_weight.append(distance)
                    # assert (distance != 0)
                    # edge_weight.append(distance)

        ei = torch.tensor(edge_index,  dtype=torch.long).t().contiguous()
        ew = torch.tensor(edge_weight, dtype=torch.float)
        ew = ew/ew.median()

        # to undirected
        return (
            ei, ew
            # torch.cat((ei, ei.flip(0)), dim=1),
            # torch.cat((ew, ew), dim=0)
        )

    def dist_to_rbf(self, distance, sigma):
        return np.exp(- (distance**2) / (2*sigma**2))


class HeteroDataset(XeniumDataset):
    r"""
    Load paired multi-modal ST data w/ hybrid resolutions into a hetero-graph
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
        setattr(self, 'window_size', 16)                # patch side-length (positional embedding)

        for key, val in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
                LOGGER.info('Update parameter {0} as {1}'.format(key, val))

        self.hetero_batches = self._load_hetero_graphs()
        
    def _load_hetero_graphs(self):
        r"""Create partitions from hetero graphs"""

        data_list = []

        for i, (adata_ref, adata_query) in enumerate(zip(self.adatas_ref, self.adatas_query)):
            LOGGER.info('Constructing hetero-graph partitions from paired data {}'.format(i+1))
            
            assert self.ref_proj_key in adata_ref.obsm_keys() and \
                   self.query_proj_key in adata_query.obsm.keys(), \
                "Invalid ref <==> query projection coordinates"
        
            # Retrieve cross-modality neighbor mapping: # dim: ([L, k], [L, K])
            ref_coords = adata_ref.obsm['spatial']
            query_coords = adata_query.obsm[self.query_proj_key]
            distances, ref_neighbor_indices = self.query_neighbors(
                ref_coords, query_coords, self.r, use_radius=True
            )  

            # ref_windows = self.__gen_windows(adata_ref.obsm[self.ref_proj_key], self.window_size)
            # query_windows = self.__gen_windows(adata_query.obsm['spatial'], self.window_size)
            # self.num_windows = int(max(ref_windows.max(), query_windows.max())) + 1
    
            # Get subgraph index mappings:
            # `*idx` / `*indices`: global index in full expression matrix
            # `*neighbors`: local index (position) in each graph partition
            for batch in self.batches:
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
                data.ligands = batch.ligands
                data.receptors = batch.receptors

                # (3). edges (within-modal & cross-modal)
                # (i). ref-to-ref graph
                data[self.ref, 'to', self.ref].edge_index = batch.edge_index

                # (ii). ref-to-query & query-to-ref graph 
                r2q_ei, r2q_ew, q2r_ei, q2r_ew = self.construct_hetero_graph(
                    ref_neighbors, 
                    distances[query_indices]
                )
                data[(self.ref, 'to', self.query)].edge_index = r2q_ei
                data[(self.query, 'to', self.ref)].edge_index = q2r_ei

                # (4). edge weights
                if self.is_weighted:
                    data[(self.ref, 'to', self.ref)].edge_attr = batch.edge_attr
                    data[(self.ref, 'to', self.query)].edge_attr = r2q_ew
                    data[(self.query, 'to', self.ref)].edge_attr = q2r_ew

                data_list.append(data)

        del self.batches  # Delete dummy batch initializations
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
        ref_to_query, ref_to_query_weight = [], []
        query_to_ref, query_to_ref_weight = [], []

        n_queries = len(ref_neighbors)
        for i in range(n_queries):
            for j, distance in zip(ref_neighbors[i], distances[i]):
                if distance < self.r:
                    ref_to_query.append([j, i])
                    ref_to_query_weight.append(self.dist_to_rbf(distance, self.sigma))
                    query_to_ref.append([i, j])
                    query_to_ref_weight.append(self.dist_to_rbf(distance, self.sigma))
        
        r2q_ei = torch.tensor(ref_to_query, dtype=torch.long).t().contiguous()
        r2q_ew = torch.tensor(ref_to_query_weight, dtype=torch.float)
        q2r_ei = torch.tensor(query_to_ref, dtype=torch.long).t().contiguous()
        q2r_ew = torch.tensor(query_to_ref_weight, dtype=torch.float)

        return r2q_ei, r2q_ew, q2r_ei, q2r_ew

    @staticmethod
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