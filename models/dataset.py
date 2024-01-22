import os
import sys
import tifffile
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from constants import *
from utils import norm_transform


class CyIFDataset(Dataset):
    def __init__(
        self,
        data_path
    ):
        self.normalize = norm_transform(CYIF_MEAN, CYIF_STD)
        self.img_names = [
            os.path.join(data_path, f) 
            for f in sorted(os.listdir(data_path))
            if f[-3:] == 'tif' or f[-4:] == 'tiff'
        ]

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img = tifffile.imread(self.img_names[index])
        return self.normalize(img.transpose(1,2,0))
