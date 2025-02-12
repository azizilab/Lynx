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

from module import Prior
from module import Encoder, GATEncoder
from module import Decoder, GATDecoder
from dataset import XeniumDataset, HeteroDataset

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
    
    def monitor_metrics(self, data: Data, device: torch.device, key: str = None):
        r"""(Debug-only) Monitor latent factor correlations & reconstruction"""
        res = self.predict(data, device)

        # Latent factor correlations
        pz = res.pz.detach().cpu().numpy()
        qz = res.qz.detach().cpu().numpy()
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
    def setup(model: nn.Module, train_configs: ConfigDict):
        r"""Setup optimizer & inference objects"""
        model.device = train_configs.device
        model.to(train_configs.device)

        optim_params = {
            'lr': train_configs.lr,
            'weight_decay': train_configs.weight_decay,
            'betas': train_configs.betas
        }
        optimizer = AdamW(optim_params)
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

        batch = next(iter(dataloader))
        batch = batch.to(device)

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

        batch = next(iter(dataloader))
        batch = batch.to(device)
        # if hasattr(model, 'init_lazy_modules'):
        #     model.init_lazy_modules(batch)

        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                loss = svi.evaluate_loss(data)
                n_obs += data.x.size(0) if key is None else data[key].x.size(0)
                total_loss += loss
        return total_loss / n_obs
    
    @staticmethod
    def set_desc(
        pbar: tqdm, epoch: int, train_loss: float, val_loss: float,
        r2: float = 0., corr_score: float = 0., DEBUG: bool = False
    ):
        if DEBUG:
            pbar.set_description(
                "Epoch {0} train -ELBO: {1}; val -ELBO: {2}; val R2: {3}; val corr: {4}".format(
                    epoch, 
                    np.round(train_loss, 3), 
                    np.round(val_loss, 3), 
                    np.round(r2, 3), 
                    np.round(corr_score, 3)
                )
            ) 
        else:
            pbar.set_description(
                "Epoch {0} train -ELBO: {1}; val -ELBO: {2}".format(
                    epoch, 
                    np.round(train_loss, 3), 
                    np.round(val_loss, 3), 
                )
            )       
        
        return None
    
    @staticmethod
    def lognorm(x):
        l = x.sum(axis=-1, keepdim=True) + EPS
        x = x / l * l.median() 
        return torch.log1p(x)  
        
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
    
    def get_x(self, x, z):
        self.eval()
        l = x.sum(axis=-1, keepdim=True)         
        x_mu = l * self.decode(z)
        return x_mu
    
    def predict(self, data: Data, device: torch.device):
        r"""Get latent representation & predictions from batched data object"""
        self.eval()
        data = data.to(device)
        x = data.x.float()
        u = data.u.float()

        pz_u, _ = self.prior(u)
        qz_xu, _ = self.get_z(x, u, data.edge_index)
        px_z = self.get_x(x, qz_xu)

        return ConfigDict({
            'qz':           qz_xu,
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
        
# TODO: [DEBUG] extend s_i as probabilistic cell-level prior, run VI w/ GAT
class HeteroVGAE(BaseModel):
    r"""Learning latent manifold w/ Conditional VGAE on hetero-graph
    Generative path: DESI (u) -> Latent (z) -> Xenium (x)
    """
    def __init__(
        self,
        configs: ConfigDict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(configs, device)
        self.ref = configs.ref
        self.query = configs.query
        self.ref_to_query = (self.ref, 'to', self.query)
        self.query_to_ref = (self.query, 'to', self.ref)

        self.prior = Prior(configs, device=device)
        self.cluster_embedding = nn.Embedding(configs.num_clusters, configs.c_latent)
        self.encode = GATEncoder(configs)
        self.decode = GATDecoder(configs)

    def model(self, data):
        pyro.module("VAE", self)

        u = data[self.query].x
        x = data[self.ref].x
        clusters = data[self.ref].cluster
        l = x.sum(axis=-1, keepdim=True)

        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        # Conditional prior: query-dim
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar = self.prior(u)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            with poutine.scale(scale=self.configs.beta):
                z = pyro.sample("z", z_dist.to_event(1))

        # Observation: reference-dim
        with pyro.plate("hires", x.size(0)):
            c = self.cluster_embedding(clusters)
            mu = self.decode(z, c, data.edge_index_dict)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - theta.log()

            nb_dist = dist.NegativeBinomial(total_count=theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, data):
        pyro.module("VAE", self)
        x = data[self.ref].x
        x = self.lognorm(x)
        u = data[self.query].x
        z_mu, z_logvar, _ = self.encode(x, u, data.edge_index_dict)

        with pyro.plate("lowres", u.size(0)):
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("z", z_dist.to_event(1))

    def get_z(self, x, u, edge_index_dict):
        x = self.lognorm(x)
        return self.encode(x, u, edge_index_dict)
    
    def get_x(self, x, z, c, edge_index_dict):
        l = x.sum(axis=-1, keepdim=True)          
        x = l * self.decode(z, c, edge_index_dict)
        return x

    def predict(self, data, device):
        data = data.to(device)
        x = data[self.ref].x
        c = self.cluster_embedding(data[self.ref].cluster).to(device)
        u = data[self.query].x

        pz, _ = self.prior(u)
        qz, _, attn_score = self.get_z(x, u, data.edge_index_dict)
        px = self.get_x(x, qz, c, data.edge_index_dict)

        return ConfigDict({
            'qz':           qz,
            'pz':           pz,
            'px':           px,  
            'attn_score':   attn_score          
        })

    def fit(self, train_configs, train_dl, val_dl, DEBUG=False):
        super().model_train(self, train_configs, train_dl, val_dl, key=self.ref, DEBUG=DEBUG)
        return None
    
    def evaluate(
        self, 
        adata_ref: sc.AnnData,
        adata_query: sc.AnnData,
        graph_data: HeteroDataset,
        device: torch.device = torch.device('cuda')
    ):
        self.eval()
        self.device = device
        self.to(device)

        n_cells, n_features = adata_ref.shape
        n_pixels, _ = adata_query.shape

        full_graph_data = HeteroDataset(
            adatas_ref=adata_ref, 
            adatas_query=adata_query, 
            n_subgraphs=1,
            k=graph_data.k, r=graph_data.r, is_weighted=graph_data.is_weighted,
            cluster=graph_data.cluster, cluster_res=graph_data.cluster_res,
            ref=graph_data.ref, ref_proj_key=graph_data.ref_proj_key,
            query=graph_data.query, query_proj_key=graph_data.query_proj_key
        )

        dataloader = DataLoader(full_graph_data, shuffle=False)
        qzu = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)    # lowres latent
        qzx = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)   # hires latent
        pz = np.zeros_like(qzu)
        px = np.zeros((n_cells, n_features), dtype=np.float32)
        attn = np.zeros(n_cells, dtype=np.float32)

        # Temporary accumulators for weighted averages
        qzx_weighted_sum = np.zeros_like(qzx)
        qzx_attention_sum = np.zeros((n_cells), dtype=np.float32)
        qzx_attention_counter = np.zeros((n_cells), dtype=np.float32)

        # Recover batched predictions in correct spatial orders
        for data in dataloader:
            res = self.predict(data, device)

            batch_qzu = res.qz.detach().cpu().numpy()  # dim: [L, K]
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()
            batch_edges = res.attn_score[0].detach().cpu().numpy().T  # dim: [edges, 2]
            batch_attn = res.attn_score[1].detach().cpu().numpy()    # dim: [edges, 1]

            query_indices = data[self.query].idx.numpy()
            qzu[query_indices] = batch_qzu
            pz[query_indices] = batch_pz

            ref_indices = data[self.ref].idx.numpy()
            px[ref_indices] = batch_px

            # Compute highres latent representations via attention assignments
            # TODO: Double-check implementations on in-degree normed attention
            for edge, a in zip(batch_edges, batch_attn):
                ref_idx = data[self.ref].idx[edge[0]]
                
                # Update accumulators for highres
                attn[ref_idx] += a
                qzx_weighted_sum[ref_idx] += a * batch_qzu[edge[1]]  # [N, latent_dim]
                qzx_attention_sum[ref_idx] += a  # [N]
                qzx_attention_counter[ref_idx] += 1

        # Average highres latent representations
        valid = qzx_attention_counter > 0
        qzx[valid.squeeze()] = qzx_weighted_sum[valid.squeeze()] / qzx_attention_sum[valid.squeeze(), None]
        attn[valid.squeeze()] = attn[valid.squeeze()] / qzx_attention_counter[valid.squeeze()]

        return ConfigDict({
            'qzu':          qzu,
            'qzx':          qzx, 
            'pz':           pz,
            'px':           px,
            'attn':         attn,
        })
    