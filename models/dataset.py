import os
import sys
import tifffile
import torch
from torch.utils.data import Dataset
from typing import Tuple

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import norm_transform


class IMSDataset(Dataset):
    def __init__(
        self,
        norm_stats: Tuple[float, float],
        data_path: str,
        prior_path: str = None,
    ):
        mean, var = norm_stats
        self.normalize = norm_transform(mean, var)
        self.img_names = [
            os.path.join(data_path, f) 
            for f in sorted(os.listdir(data_path))
            if f[-3:] == 'tif' or f[-4:] == 'tiff'
        ]

        self.prior_names = None
        if isinstance(prior_path, str):
            self.prior_names = [
                os.path.join(prior_path, f)
                for f in sorted(os.listdir(prior_path))
                if f[-3:] == 'tif' and 'dynamics' in f
            ]

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img = tifffile.imread(self.img_names[index])
        if self.prior_names is None:
            return self.normalize(img.transpose(1,2,0))
        else:
            pz_mean = tifffile.imread(self.prior_names[index])
            return self.normalize(img.transpose(1,2,0)), pz_mean
