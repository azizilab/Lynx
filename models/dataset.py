import os
import sys
import logging
import tifffile
import torch
import numpy as np
import scanpy as sc

from torch.utils.data import ConcatDataset
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.data import ClusterData
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.gen_graph import construct_graph
import logging

LOGGER = logging.getLogger()


class XeniumDataset:
    r"""
    Load Xenium ST graphs & feature matrices w/ auxiliary modality
    """
    def __init__(
        self,
        n_subgraphs : int = 8,
        **kwargs
    ):
        self.n_subgraphs = n_subgraphs
        
        # Default dataset settings
        setattr(self, 'k', 10)                # k-NN
        setattr(self, 'r', np.inf)            # neighbor range (unit: pixel)
        setattr(self, 'weighted', False)      # weighted / unweighted k-NN graph
        setattr(self, 'covariate', False)     # loading covariate

        for k, v in kwargs.items():
            if k in self.__dict__.keys():
                setattr(self, k, v)
                LOGGER.info('Update parameter {0} as {1}'.format(k, v))

    def load_graphs(self, adatas):
        """
        Compute 2D subgraphs from a list of Xenium expressions
        """
        data_list = []
        for adata in adatas:
            LOGGER.info('Constructing graph...')
            graph = construct_graph(
                self.get_coords(adata),
                k=self.k, r=self.r, weighted=self.weighted
            )

            # Expression observation
            expr = adata.X if isinstance(adata.X, np.ndarray) else \
                   adata.X.A
            
            # Auxiliary observation
            u = adata.obsm['X_aux'] if 'X_aux' in adata.obsm.keys() else \
                np.zeros_like(expr, dtype=np.float32)  
            
            # Covariate
            s = adata.obsm['X_s'] if 'X_s' in adata.obsm.keys() else \
                np.empty(shape=(adata.shape[0], 0))
            
            graph_data = pyg_utils.from_networkx(graph)
            graph_data.x = torch.tensor(expr).float()
            graph_data.u = torch.tensor(u).float()
            graph_data.s = torch.tensor(s).float()
            
            # Create partitioned subgraphs
            LOGGER.info('Partitioning {} subgraphs...'.format(self.n_subgraphs))
            subgraph_data = ClusterData(graph_data, num_parts=self.n_subgraphs) if self.n_subgraphs > 1 \
                            else [graph_data]
            
            data_list.append(subgraph_data)

        return ConcatDataset(data_list)

    def get_coords(self, adata):
        assert 'x_centroid' in adata.obs.columns and 'y_centroid' in adata.obs.columns, \
            "Lack of spatial coords for Xenium adata"
        coords = adata.obs[['x_centroid', 'y_centroid']].copy().to_numpy()  # XY-index
        return coords
    

class MultiscaleDataset(XeniumDataset):
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
    
