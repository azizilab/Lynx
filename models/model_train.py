import os
import gc
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyro

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import comb
from sklearn.metrics import r2_score
from ml_collections import ConfigDict
from torch_geometric.loader import DataLoader
from tqdm import trange, tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import AdamW


from abc import ABC, abstractmethod

# class BaseModel(nn.Module, ABC):
#     r"""Base Class for multi-modal VGAEs"""
#     def __init__(
#         self, 
#         configs: ConfigDict, 
#         device: torch.device = torch.device('cuda')
#     ):
#         super().__init__()
#         self.configs = configs
#         self.device = device
#         self.to(device)

#     @abstractmethod
#     def model(self, data: Data):
#         r"""Generative model"""
#         pass

#     @abstractmethod
#     def guide(self, data: Data):
#         r"""Variational guide"""
#         pass

#     @abstractmethod
#     def predict(self, data: Data, device: torch.device):
#         r"""Get latent (z) & reconstructions from batched data object"""
#         pass

#     @abstractmethod
#     def fit(
#         self, train_configs: ConfigDict, 
#         train_dl: DataLoader, val_dl: DataLoader,
#         save_path: str, DEBUG: bool
#     ):
#         r"""Full model training"""
#         pass

#     @abstractmethod
#     def evaluate(
#         self, adata: sc.AnnData, k: int, 
#         n_subgraphs: int, device: torch.device
#     ):
#         r"""Full model inference"""
#         pass

#     def setup(self, train_configs: ConfigDict):
#         r"""Setup optimizer & inference objects"""
#         self.device = train_configs.device
#         self.to(train_configs.device)

#         optim_params = {
#             'lr': train_configs.lr,
#             'weight_decay': train_configs.weight_decay,
#             'betas': train_configs.betas
#         }
#         optimizer = Adam(optim_params)
#         elbo = Trace_ELBO()
#         svi = SVI(self.model, self.guide, optimizer, elbo)
#         pbar = tqdm(range(train_configs.n_epochs))

#         return svi, pbar
    
#     def train_step(
#         self, dataloader: DataLoader, svi: SVI, 
#         device: torch.device, key: str = None
#     ):
#         r"""Single-epoch training step"""
#         self.train()
#         total_loss, n_obs = 0., 0.
#         for data in dataloader:
#             data = data.to(device)
#             loss = svi.step(data)
#             n_obs += data.x.size(0) if key is None else data[key].x.size(0)
#             total_loss += loss

#         return total_loss / n_obs
    
#     def val_step(
#         self, dataloader: DataLoader, svi: SVI,
#         device: torch.device, key: str = None
#     ):
#         r"""Single-epoch validation step"""
#         self.eval()
#         total_loss, n_obs = 0., 0.
#         with torch.no_grad():
#             for data in dataloader:
#                 data = data.to(device)
#                 loss = svi.evaluate_loss(data)
#                 n_obs += data.x.size(0) if key is None else data[key].x.size(0)
#                 total_loss += loss
#         return total_loss / n_obs
    
#     def monitor_metrics(self, data: Data, device: torch.device, key: str = None):
#         r"""(Debug-only) Monitor latent factor correlations & reconstruction"""
#         res = self.predict(data, device)

#         # Latent factor correlations
#         pz = res.pz.detach().cpu().numpy()
#         qz = res.qz_params[0].detach().cpu().numpy()
#         px = res.px.detach().cpu().numpy()

#         # Compute avg. pariwise factor correlations (lower triangular matrix)
#         pz_corr = np.corrcoef(pz.T)
#         pz_corr_score = np.abs(np.tril(pz_corr, k=-1)).sum() / comb(pz_corr.shape[0], 2) 
#         qz_corr = np.corrcoef(qz.T)
#         qz_corr_score = np.abs(np.tril(qz_corr, k=-1)).sum() / comb(qz_corr.shape[0], 2)

#         # Reconstruction quality
#         r2 = r2_score(
#             data.x.detach().cpu().numpy().flatten() \
#             if key is None else \
#             data[key].x.detach().cpu().numpy().flatten(), 
#             px.flatten()
#         )
#         return pz_corr_score, qz_corr_score, r2

#     def checkpoint(self, curr_loss, min_loss, patience, max_patience, save_path):
#         if curr_loss < min_loss:
#             min_loss = curr_loss
#             patience = max_patience
#             torch.save(self.state_dict(), save_path)
#         else:
#             patience -= 1
#         return min_loss, patience
    
#     @staticmethod
#     def set_desc(
#         pbar: tqdm, epoch: int, train_loss: float, val_loss: float,
#         r2: float = 0., corr_score: float = 0., DEBUG: bool = False
#     ):
#         if DEBUG:
#             pbar.set_description(
#                 "Epoch {0} train ELBO: {1}; val ELBO: {2}; val R2: {3}; val corr: {4}".format(
#                     epoch, 
#                     np.round(train_loss, 3), 
#                     np.round(val_loss, 3), 
#                     np.round(r2, 3), 
#                     np.round(corr_score, 3)
#                 )
#             ) 
#         else:
#             pbar.set_description(
#                 "Epoch {0} train ELBO: {1}; val ELBO: {2}".format(
#                     epoch, 
#                     np.round(train_loss, 3), 
#                     np.round(val_loss, 3), 
#                 )
#             )       
        
#         return None
        
#     @staticmethod
#     def plot_loss(train_losses, val_losses):
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.plot(np.arange(len(train_losses)), train_losses, label='Train')
#         ax.plot(np.arange(len(val_losses)), val_losses, label='Val')
#         ax.set_xlabel('Epochs')
#         ax.set_ylabel('-ELBO')

#         ax.legend()
#         ax.spines[['right', 'top']].set_visible(False)
#         ax.get_xaxis().tick_bottom()
#         ax.get_yaxis().tick_left()
#         plt.show()
        
#         return None
    
#     @staticmethod
#     def plot_latent_corr(pz_corr_scores, qz_corr_scores):
#         if len(pz_corr_scores) == 0 or len(qz_corr_scores) == 0:
#             return None
        
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.plot(np.arange(len(pz_corr_scores)), pz_corr_scores, '.--', label='Prior')
#         ax.plot(np.arange(len(qz_corr_scores)), qz_corr_scores, '.--', label='Posterior')

#         ax.set_xlabel('Epoch checkpoint')
#         ax.set_ylabel('Avg. factor correlations')
#         ax.legend()

#         ax.spines[['right', 'top']].set_visible(False)
#         ax.get_xaxis().tick_bottom()
#         ax.get_yaxis().tick_left()
#         plt.show()

#         return None
