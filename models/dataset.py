import os
import sys
import logging
import tifffile
import torch
import numpy as np
import scanpy as sc

from torch.utils.data import Dataset, ConcatDataset
from torch_geometric import utils as pyg_utils
from torch_geometric.data import ClusterData
from typing import Tuple

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.gen_graph import *
import logging

LOGGER = logging.getLogger()


class XeniumDataset(Dataset):
    """
    Load Xenium ST feature matrices
    """
    def __init__(
        self,
        adata,
        **kwargs
    ):
        self.params = {
            'min_counts':   10,
            'min_cells':    5
        }
        for k, v in kwargs.items():
            self.params[k] = v
        
        # self._preprocess(adata)
        self.feature_mat = adata.X if isinstance(adata.X, np.ndarray) else adata.X.A  
        
    def __len__(self):
        return self.feature_mat.shape[0]

    def __getitem__(self, idx):
        assert idx < self.feature_mat.shape[0]
        return torch.tensor(self.feature_mat[idx]).float()
    
    def _preprocess(self, adata):
        sc.pp.filter_cells(adata, min_counts=self.params['min_counts'])
        sc.pp.filter_genes(adata, min_cells=self.params['min_cells'])
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)


class XeniumGraphDataset:
    """
    Load Xenium ST graphs & feature matrices
    """
    def __init__(
        self,
        n_subgraphs : int = 4,
        **kwargs
    ):
        self.n_subgraphs = n_subgraphs
        self.params = {
            'k': 10,            # k-NN
            'r': np.inf,        # neighbor range (unit: pixel)
            'weighted': False   # weighted / unweighted k-NN graph
        }   
        for k, v in kwargs.items():
            if k in self.params.keys():
                LOGGER.info('Updating graph param {0} as {1}'.format(k, v))

    def load_graphs(self, adata_list):
        """
        Compute 2D subgraphs from a list of Xenium expressions
        """
        data_list = []
        for adata in adata_list:
            feature_mat = adata.X if isinstance(adata.X, np.ndarray) else adata.X.A
            u = adata.obsm['X_aux'] if 'X_aux' in adata.obsm.keys() else \
                np.zeros_like(feature_mat, dtype=np.float32)  # Dim. reduced auxiliary observation
            
            graph = construct_graph(
                self._get_coords(adata),
                k=self.params['k'],
                r=self.params['r'],
                weighted=self.params['weighted']
            )
            
            data = pyg_utils.from_networkx(graph)
            data.x = torch.tensor(feature_mat).float()
            data.u = torch.tensor(u).float()
            
            graph_data = ClusterData(data, num_parts=self.n_subgraphs) \
                         if self.n_subgraphs > 1 \
                         else data
            data_list.append(graph_data)

        return ConcatDataset(data_list)


    def _get_coords(self, adata):
        assert 'x_centroid' in adata.obs.columns and 'y_centroid' in adata.obs.columns, \
            "Lack of spatial coords for Xenium adata"
        coords = adata.obs[['y_centroid', 'x_centroid']].copy().to_numpy()
        return coords

