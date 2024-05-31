import os
import sys
import tifffile
import torch
import numpy as np

from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import ClusterData
from typing import Tuple

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.utils import norm_transform
from util.gen_graph import *


class IMSDataset(Dataset):
    def __init__(
        self,
        norm_stats: Tuple[float, float],
        data_path: str,
        prior_path: str = None,
    ):
        mean, var = norm_stats
        self.normalize = norm_transform(mean, var)
        self.filenames = [
            os.path.join(data_path, f) 
            for f in sorted(os.listdir(data_path))
            if f[-3:] == 'tif' or f[-4:] == 'tiff'
        ]

        self.prior_names = None
        if isinstance(prior_path, str):
            self.prior_names = [
                os.path.join(prior_path, f)
                for f in sorted(os.listdir(prior_path))
                if f[-3:] == 'tif' and 'prior' in f
            ]

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img = tifffile.imread(self.filenames[index])
        if self.prior_names is None:
            return self.normalize(img.transpose(1,2,0))
        else:
            pz_mean = tifffile.imread(self.prior_names[index])
            return self.normalize(img.transpose(1,2,0)), pz_mean
        

class DESIGraphDataset:
    """
    Load paired metabolomics graphs & feature matrices
    """
    def __init__(
        self,
        data_path: str,
        prior_path: str = None,
        prior_suffix: str = 'prior',
        n_subgraphs: int = 1,
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
                print('Updating graph param {0} as {1}'.format(k, v))

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
            
            fname = filename.split('/')[-1].split('.')[0]  # trim full path & suffix
            if any(fname in f for f in prior_filenames):
                # Load optional u_prior
                prior = tifffile.imread(os.path.join(self.prior_path, fname+'_'+self.prior_suffix+'.tif'))
                prior = torch.tensor(prior)
                
                self.priors[fname] = prior
                xpos, ypos = data.pos.T
                data.u_prior = prior[tuple([ypos, xpos])]  # ij-index ordering
            else:
                # Sample u_prior from standard Gaussian
                data.u_prior = torch.randn_like(data.x)

            graph_data = ClusterData(data, num_parts=self.n_subgraphs) \
                         if self.n_subgraphs > 1 \
                         else data
            
            data_list.append(graph_data)

        return ConcatDataset(data_list)
            
    def _get_coords(self, img):
        yy, xx = np.meshgrid(np.arange(img.shape[-2]),
                             np.arange(img.shape[-1]))
        return np.array([yy.flatten(), xx.flatten()]).T
