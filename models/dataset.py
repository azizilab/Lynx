import os
import sys
import logging
import tifffile
import torch
import numpy as np
import scanpy as sc

from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import ClusterData
from typing import Tuple

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.utils import norm_transform
from util.gen_graph import *
import logging

LOGGER = logging.getLogger()


class DESIDataset(Dataset):
    """
    Load metabolomics (DESI-MSI) feature matrices
    """
    def __init__(
        self,
        data_path: str,
        prior_path: str = None,
    ):
        # self.filenames = [
        #     os.path.join(data_path, f)
        #     for f in sorted(os.listdir(data_path))
        #     if f[-3:] == 'tif' or f[-4:] == 'tiff'
        # ]

        # self.prior_filenames = []
        # if os.path.exists(prior_path):
        #     self.prior_filenames = [
        #         os.path.join(data_path, f)
        #         for f in sorted(os.listdir(data_path))
        #         if (f[-3:] == 'tif' or f[-4] == 'tiff') and prior_suffix in f
        #     ]

        img = tifffile.imread(data_path)
        nchans, ny, nx = img.shape
        self.feature_mat = torch.tensor(img.transpose(2, 1, 0).reshape(-1, nchans))
        if os.path.exists(prior_path):
            u_img = tifffile.imread(prior_path)
            self.u_prior = torch.tensor(u_img.reshape(ny*nx, -1))
        else:
            self.u_prior = torch.rand(ny*nx)
        
    def __len__(self):
        # return len(self.filenames)
        return self.feature_mat.shape[0]

    def __getitem__(self, idx):
        # assert 0 <= idx < len(self.filenames), "Dataloading index {} out of bound".format(idx)
        # img = tifffile.imread(self.filenames[idx])
        # nchans, ny, nx = img.shape
        # feature_mat = torch.tensor(img.transpose(2, 1, 0).reshape(-1, nchans))  # dim: [X*Y, C]

        # fname = self.filenames[idx].split('/')[-1].split('.')[0]  # trim full path & .tif suffix
        # if any(fname in f for f in self.prior_filenames):
        #     u_prior = tifffile.imread(os.path.join(self.prior_path, fname+'_'+self.prior_suffix+'.tif')).flatten()
        #     u_prior = torch.tensor(u_prior).float()
        # else:
        #     u_prior = torch.rand(ny*nx)

        # return feature_mat, u_prior

        assert idx < self.feature_mat.shape[0]
        return (torch.tensor(self.feature_mat[idx]).float(), torch.tensor(self.u_prior[idx]).float())
    

class XeniumDataset(Dataset):
    """
    Load Xenium ST feature matrices
    """
    def __init__(
        self,
        data_path: str,
        **kwargs
    ):
        filename = os.path.join(data_path, 'cell_feature_matrix.h5')
        assert os.path.isfile(filename), "Xenium expression h5 file doesn't exist"
        self.params = {
            'min_counts':   10,
            'min_cells':    5
        }
        for k, v in kwargs.items():
            self.params[k] = v
        
        adata = sc.read_10x_h5(filename)
        self._preprocess(adata)
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
        # TODO: for Xenium, load Dataset from processed `adata`
        self.n_subgraphs = n_subgraphs
        self.params = {
            'k': 10,            # k-NN
            'r': np.inf,         # neighbor range (unit: pixel)
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
            aux_mat = adata.obsm['X_aux'] if 'X_aux' in adata.obsm.keys() else \
                      np.zeros_like(feature_mat, dtype=np.float32) 
            
            graph = construct_graph(self._get_coords(adata),
                                    k=self.params['k'],
                                    r=self.params['r'],
                                    weighted=self.params['weighted'])
            
            data = pyg_utils.from_networkx(graph)
            data.x = torch.tensor(feature_mat).float()
            data.u = torch.tensor(aux_mat).float()
            
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


class DESIGraphDataset:
    """
    Load paired metabolomics (DESI-MSI) graphs & feature matrices
    """
    def __init__(
        self,
        data_path: str,
        prior_path: str = None,
        prior_suffix: str = 'prior',
        n_subgraphs: int = 4,
        **kwargs
    ):
        self.filenames = [
            os.path.join(data_path, f)
            for f in sorted(os.listdir(data_path))
            if f[-3:] == 'tif' or f[-4:] == 'tiff'
        ]

        self.priors = {}
        self.prior_path = prior_path
        self.prior_suffix = prior_suffix
        self.n_subgraphs = n_subgraphs
        self.params = {
            'k': 10,            # k-NN
            'r': 5,             # neighbor range (unit: pixel)
            'weighted': False   # weighted / unweighted k-NN graph
        }
        for k, v in kwargs.items():
            self.params[k] = v
            if k in self.params.keys():
                LOGGER.info('Updating graph param {0} as {1}'.format(k, v))

    def load_graphs(self):
        """
        Compute individual 2D graphs from DESI images with
        Node embeddings, coords & prior values
        """
        data_list = []
        prior_filenames = []
        if os.path.exists(self.prior_path):
            prior_filenames = [
                f for f in sorted(os.listdir(self.prior_path))
                if f[-3:] == 'tif'
            ]

        for filename in self.filenames:
            # Build graph from feature matrix
            img = tifffile.imread(filename)  # dim: [C, Y, X]
            nchans = img.shape[0]
            feature_mat = img.transpose(2, 1, 0).reshape(-1, nchans)  # dim: [X*Y, C]
            graph = construct_graph(self._get_coords(img),
                                    k=self.params['k'],
                                    r=self.params['r'],
                                    weighted=self.params['weighted'])
            
            data = pyg_utils.from_networkx(graph)
            data.x = torch.tensor(feature_mat).float()
            data.u_prior = None
            
            fname = filename.split('/')[-1].split('.')[0]  # trim full path & .tif suffix
            if any(fname in f for f in prior_filenames):
                # Load optional u_prior
                prior = tifffile.imread(os.path.join(self.prior_path, fname+'_'+self.prior_suffix+'.tif'))
                prior = torch.tensor(prior)
                
                self.priors[fname] = prior
                xpos, ypos = data.pos.T
                data.u_prior = prior[tuple([ypos, xpos])]  # ij-index ordering
            else:
                # Sample u_prior from standard Gaussian
                data.u_prior = torch.rand_like(data.x)

            graph_data = ClusterData(data, num_parts=self.n_subgraphs) \
                         if self.n_subgraphs > 1 \
                         else data
            
            data_list.append(graph_data)

        return ConcatDataset(data_list)
            
    def _get_coords(self, img):
        yy, xx = np.meshgrid(np.arange(img.shape[-2]),
                             np.arange(img.shape[-1]))
        return np.array([yy.flatten(), xx.flatten()]).T
