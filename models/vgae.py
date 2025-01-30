import os
import sys
import numpy as np
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from abc import ABC, abstractmethod
from typing import Callable
from ml_collections import ConfigDict
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, AdamW

# modules for debug
import gc
from scipy.special import comb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from module import Prior, AggregatePrior
from module import Encoder, AggregateEncoder
from module import Decoder, AggregateDecoder
from dataset import XeniumDataset, MultiscaleDataset
from model_train import *

EPS = 1e-8


class BaseModel(nn.Module, ABC):
    r"""Base Class for multi-modal VGAEs"""
    def __init__(
        self, 
        configs: ConfigDict, 
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        self.configs = configs
        self.device = device
        self.to(device)

    @abstractmethod
    def model(self, data: Data):
        r"""Generative model"""
        pass

    @abstractmethod
    def guide(self, data: Data):
        r"""Variational guide"""
        pass

    @abstractmethod
    def predict(self, data: Data, device: torch.device):
        r"""Get latent (z) & reconstructions from batched data object"""
        pass

    @abstractmethod
    def fit(
        self, train_configs: ConfigDict, 
        train_dataloader: DataLoader, val_dataloader: DataLoader, 
        DEBUG: str = False
    ):
        r"""Full model training"""
        pass

    @abstractmethod
    def evaluate(
        self, adata: sc.AnnData, k: int, 
        n_subgraphs: int, device: torch.device
    ):
        r"""Full model inference"""
        pass
    
    def model_train(
        self, model, train_configs: ConfigDict, 
        train_dl: DataLoader, val_dl: DataLoader,
        key: str = None, save_path: str = 'best_model.pth', 
        DEBUG: bool = False
    ):
        # Setup optimizer & inference schemes
        svi, progress_bar = self.setup(model, train_configs)
        
        # Loss configs
        train_losses, val_losses = [], []
        patience = train_configs.patience
        max_patience = train_configs.patience
        min_val_loss = np.inf

        # Debug configs
        r2, qz_corr_score, pz_corr_scores, qz_corr_scores = 0., 0., [], []
    
        for epoch in progress_bar:
            train_loss = self.train_step(model, train_dl, svi, key=key, device=train_configs.device)
            val_loss = self.val_step(model, val_dl, svi, key=key, device=train_configs.device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save the best model params
            min_val_loss, patience = self.checkpoint(
                val_loss, min_val_loss, patience, max_patience, save_path
            )
            if patience == 0:
                break

            # DEBUG: disentanglement monitor
            if DEBUG and epoch % 10 == 0:
                data = next(iter(val_dl))
                pz_corr_score, qz_corr_score, r2 = self.monitor_metrics(data, key=key, device=train_configs.device)
                pz_corr_scores.append(pz_corr_score)
                qz_corr_scores.append(qz_corr_score)

            self.set_desc(progress_bar, epoch, train_loss, val_loss, r2, qz_corr_score, DEBUG)
            gc.collect()

        self.load_state_dict(torch.load(save_path))  # Load the best model
        self.plot_latent_corr(pz_corr_scores, qz_corr_scores)
        self.plot_loss(train_losses, val_losses)
        return None

    @staticmethod
    def setup(model: nn.Module, train_configs: ConfigDict):
        r"""Setup optimizer & inference objects"""
        model.device = train_configs.device
        model.to(train_configs.device)

        optim_params = {
            'lr': train_configs.lr,
            'weight_decay': train_configs.weight_decay,
            'betas': train_configs.betas
        }
        optimizer = Adam(optim_params)
        elbo = Trace_ELBO()
        svi = SVI(model.model, model.guide, optimizer, elbo)
        pbar = tqdm(range(train_configs.n_epochs))

        return svi, pbar
    
    @staticmethod
    def train_step(
        model: nn.Module, dataloader: DataLoader, svi: SVI, 
        device: torch.device, key: str = None
    ):
        r"""Single-epoch training step"""
        model.train()
        total_loss, n_obs = 0., 0.
        for data in dataloader:
            data = data.to(device)
            loss = svi.step(data)
            n_obs += data.x.size(0) if key is None else data[key].x.size(0)
            total_loss += loss

        return total_loss / n_obs
    
    @staticmethod
    def val_step(
        model: nn.Module, dataloader: DataLoader, svi: SVI,
        device: torch.device, key: str = None
    ):
        r"""Single-epoch validation step"""
        model.eval()
        total_loss, n_obs = 0., 0.
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                loss = svi.evaluate_loss(data)
                n_obs += data.x.size(0) if key is None else data[key].x.size(0)
                total_loss += loss
        return total_loss / n_obs
    
    def monitor_metrics(self, data: Data, device: torch.device, key: str = None):
        r"""(Debug-only) Monitor latent factor correlations & reconstruction"""
        res = self.predict(data, device)

        # Latent factor correlations
        pz = res.pz.detach().cpu().numpy()
        qz = res.qz_params[0].detach().cpu().numpy()
        px = res.px.detach().cpu().numpy() \
            if 'px' in res.keys() else \
            res.py.detach().cpu().numpy()

        # Compute avg. pariwise factor correlations (lower triangular matrix)
        pz_corr = np.corrcoef(pz.T)
        pz_corr_score = np.abs(np.tril(pz_corr, k=-1)).sum() / comb(pz_corr.shape[0], 2) 
        qz_corr = np.corrcoef(qz.T)
        qz_corr_score = np.abs(np.tril(qz_corr, k=-1)).sum() / comb(qz_corr.shape[0], 2)

        # Reconstruction quality
        r2 = r2_score(
            data.x.detach().cpu().numpy().flatten() \
            if key is None else \
            data[key].x.detach().cpu().numpy().flatten(), 
            px.flatten()
        )
        return pz_corr_score, qz_corr_score, r2

    def checkpoint(self, curr_loss, min_loss, patience, max_patience, save_path):
        if curr_loss < min_loss:
            min_loss = curr_loss
            patience = max_patience
            torch.save(self.state_dict(), save_path)
        else:
            patience -= 1
        return min_loss, patience
    
    @staticmethod
    def set_desc(
        pbar: tqdm, epoch: int, train_loss: float, val_loss: float,
        r2: float = 0., corr_score: float = 0., DEBUG: bool = False
    ):
        if DEBUG:
            pbar.set_description(
                "Epoch {0} train ELBO: {1}; val ELBO: {2}; val R2: {3}; val corr: {4}".format(
                    epoch, 
                    np.round(train_loss, 3), 
                    np.round(val_loss, 3), 
                    np.round(r2, 3), 
                    np.round(corr_score, 3)
                )
            ) 
        else:
            pbar.set_description(
                "Epoch {0} train ELBO: {1}; val ELBO: {2}".format(
                    epoch, 
                    np.round(train_loss, 3), 
                    np.round(val_loss, 3), 
                )
            )       
        
        return None
        
    @staticmethod
    def plot_loss(train_losses, val_losses):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.arange(len(train_losses)), train_losses, label='Train')
        ax.plot(np.arange(len(val_losses)), val_losses, label='Val')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('-ELBO')

        ax.legend()
        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.show()
        
        return None
    
    @staticmethod
    def plot_latent_corr(pz_corr_scores, qz_corr_scores):
        if len(pz_corr_scores) == 0 or len(qz_corr_scores) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.arange(len(pz_corr_scores)), pz_corr_scores, '.--', label='Prior')
        ax.plot(np.arange(len(qz_corr_scores)), qz_corr_scores, '.--', label='Posterior')

        ax.set_xlabel('Epoch checkpoint')
        ax.set_ylabel('Avg. factor correlations')
        ax.legend()

        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.show()

        return None


class VGAE(BaseModel):
    r"""Learning latent manifold w/ Conditional VGAE
    Generative path: DESI (u) -> Latent (z) -> Xenium (x)
    """
    def __init__(
        self, 
        configs: ConfigDict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(configs, device)
        self.prior = Prior(configs, device=device)
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)

    def model(self, data):
        pyro.module("VGAE", self)
        x = data.x
        u = data.u

        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z_mu, z_logvar = self.prior(u)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))

            mu = self.decode(z)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - (self.theta).log()

            nb_dist = dist.NegativeBinomial(total_count=self.theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, data):
        pyro.module("VGAE", self)
        x = data.x
        u = data.u
        edge_index = data.edge_index

        x = torch.log1p(x)
        z_mu, z_logvar = self.encode(x, u, edge_index)  # Global sample per subgraph

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta): 
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            pyro.sample("z", z_dist.to_event(1)) 

    def get_z(self, x, u, edge_index):
        x = torch.log1p(x)
        return self.encode(x, u, edge_index)
    
    def get_x(self, x, z_param):
        self.eval()
        l = x.sum(axis=-1, keepdim=True)
        z_mu = z_param[0]            
        x_mu = l * self.decode(z_mu)
        return x_mu
    
    def predict(self, data: Data, device: torch.device):
        r"""Get latent representation & predictions from batched data object"""
        self.eval()
        data = data.to(device)
        x = data.x.float()
        u = data.u.float()

        pz_u, _ = self.prior(u)
        qz_xu_params = self.get_z(x, u, data.edge_index)
        px_z = self.get_x(x, qz_xu_params)

        return ConfigDict({
            'qz_params':    qz_xu_params,
            'pz':           pz_u,
            'px':           px_z
        })
    
    def fit(self, train_configs, train_dl, val_dl: DataLoader, DEBUG=False):  
        super().model_train(self, train_configs, train_dl, val_dl, DEBUG=DEBUG)
        return None
        
    def evaluate(
        self, 
        adata: sc.AnnData,
        k: int = 30, 
        n_subgraphs: int = 8, 
        device: torch.device = torch.device('cuda')
    ):
        r"""Full inference"""
        self.eval()
        self.device = device
        self.to(device)

        graph_data = XeniumDataset(adata, k=k, n_subgraphs=n_subgraphs)

        dataloader = DataLoader(graph_data, shuffle=False)
        qz = np.zeros((adata.shape[0], self.configs.c_latent), dtype=np.float32)
        pz = np.zeros_like(qz)
        px = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)

        # Recover batched predictions in correct spatial orders
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qz = res.qz_params[0].detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()

            for idx, qz_i, pz_i, px_i in zip(data.idx, batch_qz, batch_pz, batch_px):
                qz[idx], pz[idx], px[idx] = qz_i, pz_i, px_i
        
        return ConfigDict({
            'qz':           qz,
            'pz':           pz,
            'px':           px
        })
        
                
class MultiscaleVGAE(BaseModel):
    r"""Learning latent manifold w/ Conditional VGAE
    Generative path: DESI (u) -> Latent (z) -> Xenium (x)
    """
    def __init__(self, configs, device=torch.device('cuda')):
        super().__init__(configs, device)
        self.ref = configs.ref
        self.query = configs.query

        self.prior = AggregatePrior(configs)
        self.encode = AggregateEncoder(configs)
        self.decode = AggregateDecoder(configs)

    def model(self, data):
        pyro.module("VAE", self)

        x = data[self.ref].x
        x = self.lognorm(x)   # Normalize by library size & scale
        r2r_edge_index = data[self.ref].edge_index  # reference-reference edges
        
        y = data[self.query].x
        neighbors = data[self.query].neighbor
        
        # q2r_edge_index = data[(self.query, 'to', self.ref)].edge_index

        with pyro.plate("batch", y.size(0)):
            with poutine.scale(scale=self.configs.beta):
                z_mu, z_logvar = self.prior(x, r2r_edge_index, neighbors)
                z_dist = dist.Normal(z_mu, torch.exp(z_logvar))
                z = pyro.sample("z", z_dist.to_event(1))
            
            y_mu, y_logvar = self.decode(z)
            normal_dist = dist.Normal(y_mu, torch.exp(y_logvar))
            pyro.sample("y", normal_dist.to_event(1), obs=y)

    def guide(self, data):
        pyro.module("VAE", self)
        
        x = data[self.ref].x
        x = self.lognorm(x)    # Normalize by library size & scale
        x_windows = data[self.ref].window

        y = data[self.query].x
        y_windows = data[self.query].window
        neighbors = data[self.query].neighbor

        z_mu, z_logvar, _ = self.encode(x, y, neighbors, x_windows, y_windows)  

        with pyro.plate("batch", y.size(0)): 
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar))
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("z", z_dist.to_event(1)) 

    def get_z(self, x, y, neighbors, x_windows, y_windows):
        z_mu, z_logvar, attn_scores = self.encode(
            x, y, neighbors, x_windows, y_windows
        )  
        return z_mu, z_logvar, attn_scores
    
    def get_y(self, z):
        # Note: linear-layer decoder:
        y, _ = self.decode(z)
        return y
    
    def predict(self, data, device):
        r"""Get latent representation & predictions from batched data"""
        self.eval()
        data = data.to(device)
        x = data[self.ref].x
        x = self.lognorm(x) 
        x_windows = data[self.ref].window
        r2r_edge_index = data[self.ref].edge_index

        y = data[self.query].x
        y_windows = data[self.query].window
        neighbors = data[self.query].neighbor

        # q2r_edge_index = data[(self.query, 'to', self.ref)].edge_index

        pz_x, _ = self.prior(x, r2r_edge_index, neighbors)
        qz_xy_params = self.get_z(x, y, neighbors, x_windows, y_windows)
        py_z = self.get_y(qz_xy_params[0])

        return ConfigDict({
            'qz_params':    qz_xy_params,
            'pz':           pz_x,
            'py':           py_z
        })
        
    def fit(self, train_configs, train_dl, val_dl, DEBUG=False):
        r"""Full training"""
        super().model_train(self, train_configs, train_dl, val_dl, key=self.query, DEBUG=DEBUG)
        return None

    def evaluate(
        self, 
        adata_hires: sc.AnnData,
        adata_lowres: sc.AnnData,
        k: int = 10, 
        n_subgraphs: int = 8, 
        device: torch.device = torch.device('cuda')
    ):
        r"""Full inference"""
        self.eval()
        self.device = device
        self.to(device)

        n_cells = adata_hires.shape[0]
        n_pixels, n_features = adata_lowres.shape

        graph_data = MultiscaleDataset(
            adatas_ref=adata_hires, adatas_query=adata_lowres, k=k, n_subgraphs=n_subgraphs
        )

        dataloader = DataLoader(graph_data, shuffle=False)
        qzy = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)  # lowres latent
        qzx = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)   # hires latent
        pz = np.zeros_like(qzy)
        py = np.zeros((n_pixels, n_features), dtype=np.float32)
        attn = np.zeros((n_pixels, k), dtype=np.float32)

        # Cell-level (`ref`) latent assignment based on weighted sum attention values
        qzx_weighted_sum = np.zeros_like(qzx)
        qzx_attention_sum = np.zeros((n_cells), dtype=np.float32)

        # Recover batched predictions in correct spatial orders
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qzy = res.qz_params[0].detach().cpu().numpy()  # dim: [L, K]
            batch_attn = res.qz_params[2].detach().cpu().numpy() # dim: [L, K]
            batch_pz = res.pz.detach().cpu().numpy()  
            batch_py = res.py.detach().cpu().numpy()

            query_indices = data[self.query].idx.numpy()

            qzy[query_indices] = batch_qzy
            attn[query_indices] = batch_attn
            pz[query_indices] = batch_pz
            py[query_indices] = batch_py

            # Compute `ref` representations as weighted sum across connected `queries`
            ref_indices = data[self.ref].idx
            for i, neighbors in enumerate(data[self.query].neighbor): 
                xenium_idx = ref_indices[neighbors]
                qzx_weighted_sum[xenium_idx] += batch_attn[i, :, None] * batch_qzy[i]  # dim: [k, latent_dim]
                qzx_attention_sum[xenium_idx] += batch_attn[i]  

        # Average highres latent representations
        valid = qzx_attention_sum > 0
        if not np.all(valid):
            raise AssertionError("Not all cells have mapped pixels!")
        qzx[valid.squeeze()] = qzx_weighted_sum[valid.squeeze()] / qzx_attention_sum[valid.squeeze(), None]

        return ConfigDict({
            'qzx':          qzx,
            'qzy':          qzy,
            'pz':           pz,
            'py':           py,
        })

    @staticmethod
    def lognorm(x):
        l = x.sum(axis=-1, keepdim=True) + EPS
        x = x / l * l.median() 
        return torch.log1p(x)  
    