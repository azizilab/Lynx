import os
import tifffile
import torch
from torch.utils.data import Dataset


class CyIFDataset(Dataset):
    def __init__(
        self,
        data_path,
        prior_path
    ):
        # TODO: write I/O loading from gcloud
        self.img_names = [
            os.path.join(data_path, f) 
            for f in sorted(os.listdir(data_path))
            if f[-3:] == 'tif' or f[-4:] == 'tiff'
        ]

        self.prior_names = [
            os.path.join(prior_path, f)
            for f in sorted(os.listdir(prior_path))
            if (f[-3:] == 'tif' or f[-4:] == 'tiff') and 'dynamic' in f  # TODO: refactor this!
        ]

        assert len(self.img_names) == len(self.prior_names), "Unequal data & prior sizes:{0}, {1}".format(
            len(self.img_names), len(self.prior_names)
        )

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img = tifffile.imread(self.img_names[index])
        pz_mu = tifffile.imread(self.prior_names[index])
        return torch.tensor(img), torch.tensor(pz_mu)

