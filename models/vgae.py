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

from ml_collections import ConfigDict
from typing import Dict, List

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean


sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from module import ConditionalPrior
from module import Encoder, AggregateEncoder
from module import Decoder, AggregateDecoder
from dataset import XeniumDataset, HeteroDataset

EPS = 1e-8


class VGAE(nn.Module):
    r"""Learning latent manifold w/ Conditional VGAE
    U (DESI) -> Z (latent) -> X (Xenium)
    """
    def __init__(
        self, 
        configs: ConfigDict,
        device: torch.device = torch.device('cuda')
    ):
        super(VGAE, self).__init__()
        self.configs = configs
        self.device = device

        self.prior = ConditionalPrior(configs, device=device)
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)

        self.to(device)

    def model(self, x, u, s, edge_index):
        pyro.module("prior", self.prior)
        pyro.module("decoder", self.decode)

        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z_mu, z_logvar = self.prior(u, device=self.device)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))

            mu = self.decode(z, s, edge_index)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - (self.theta).log()

            nb_dist = dist.NegativeBinomial(total_count=self.theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, x, u, s, edge_index):
        pyro.module("encoder", self.encode)

        if self.configs.embed_option == 'attn':
            l = x.sum(axis=-1, keepdim=True)
            x = x / l * l.median()

        x = torch.log1p(x)
        z_mu, z_logvar, _ = self.encode(x, u, s, edge_index) # Global sample per subgraph

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta): 
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            pyro.sample("z", z_dist.to_event(1)) 

    def get_z(self, x, u, s, edge_index):
        if self.configs.embed_option == 'attn':
            l = x.sum(axis=-1, keepdim=True)
            x = x / l * l.median()
        x = torch.log1p(x)
        return self.encode(x, u, s, edge_index)
    
    def sample_z(self, x, u, s, edge_index, n_samples=100):
        z_mu, z_logvar, _ = self.get_z(x, u, s, edge_index)
        z_samples = dist.Normal(z_mu, torch.exp(z_logvar//2)).sample((n_samples,))
        return z_samples
    
    def get_x(self, x, s, edge_index, z_param):
        self.eval()
        l = x.sum(axis=-1, keepdim=True)
        z_mu = z_param[0]            
        x_mu = l * self.decode(z_mu, s, edge_index)
        return x_mu
    
    def sample_x(self, x, u, edge_index, n_samples=100):
        self.eval()
        x = torch.tensor(x).float()
        x = torch.log(x + EPS)
        u = torch.tensor(u).float()
        ei = torch.tensor(edge_index)

        predictive = pyro.infer.Predictive(self, self.guide, n_samples)
        pxs = predictive(x, u, ei)
        return pxs["x"]
    
    def predict(self, data: Data, device: torch.device):
        r"""Get latent representation & predictions from `pyg` Data object"""
        self.eval()
        data = data.to(device)
        x = data.x.float()
        u = data.u.float()
        s = data.s.float()

        pz_u, _ = self.prior(u)
        qz_xu_params = self.get_z(x, u, s, data.edge_index)
        px_z = self.get_x(x, s, data.edge_index, qz_xu_params)

        return ConfigDict({
            'qz_params':    qz_xu_params,
            'pz':           pz_u,
            'px':           px_z
        })
    
    def evaluate(
        self, 
        adata: sc.AnnData,
        k: int = 30, 
        n_subgraphs: int = 8, 
        device: torch.device = torch.device('cuda')
    ):
        r"""Get latent representation & predictions on subgraph batches"""
        self.eval()
        self.device = device
        self.to(device)
        self._move_attr_to(device)

        pos_to_index = {
            tuple(pos): i
            for i, pos in enumerate(
                adata.obsm['spatial'].astype(np.float32)
            )
        }

        graph_data = XeniumDataset(
            k=k, n_subgraphs=n_subgraphs
        ).load_graphs([adata])

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

            for pos, qz_i, pz_i, px_i in zip(data.pos, batch_qz, batch_pz, batch_px):
                pos = tuple(pos.detach().cpu().numpy().astype(np.float32))
                idx = pos_to_index[pos]
                qz[idx], pz[idx], px[idx] = qz_i, pz_i, px_i
        
        return ConfigDict({
            'qz':           qz,
            'pz':           pz,
            'px':           px
        })
        
    def _move_attr_to(self, device):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))


class HeteroVGAE(VGAE):
    r"""Learning latent manifold w/ Conditional VGAE
    via Xenium (x) -> Latent (z) -> DESI (y)
    """
    def __init__(self, configs, device=torch.device('cuda')):
        super(HeteroVGAE, self).__init__(configs)
        self.configs = configs
        self.device = device
        
        self.prior = ConditionalPrior(configs, device=device)
        self.encode = AggregateEncoder(configs)
        self.decode = AggregateDecoder(configs)
        
        self.edge_label = configs.edge_label
        self.ref = self.edge_label[0]  # High-res modality (Xenium)
        self.query = self.edge_label[-1]     # Low-res modality (DESI)
        self.to(device)

    def model(self, x_dict, edge_index_dict):
        pyro.module("prior", self.prior)
        pyro.module("decoder", self.decode)

        x = x_dict[self.ref]
        y = x_dict[self.query]
        edge_index = edge_index_dict[self.edge_label]

        # Whiten Xenium counts (for cond. prior w/ ICA weight init.)
        # TODO: compare w/ just lognorm?
        # x = torch.log1p(x)
        # x = self.whiten(x)
        x = self.lognorm(x)

        with pyro.plate("batch", y.size(0)), poutine.scale(scale=self.configs.beta):
            # Cell-level stats
            z_mu, z_logvar = self.prior(x)

            # Pooling latent (z_i0,...,z_im) -> z_j based on ref->query k-NN graphs
            # edge_index: (row0: ref_indices, row1: mapped query indices)
            z_mu_pooled = scatter_mean(z_mu[edge_index[0]], edge_index[1], dim=0)
            z_logvar_pooled = scatter_mean(z_logvar[edge_index[0]], edge_index[1], dim=0)

            z_dist = dist.Normal(z_mu_pooled, torch.exp(z_logvar_pooled/2))
            z = pyro.sample("z", z_dist.to_event(1))
            
            y_mu, y_logvar = self.decode(z)
            normal_dist = dist.Normal(y_mu, torch.exp(y_logvar/2))
            pyro.sample("y", normal_dist.to_event(1), obs=y)

    def guide(self, x_dict, edge_index_dict):
        pyro.module("encoder", self.encode)
        
        x_dict[self.ref] = self.lognorm(x_dict[self.ref])  # Normalize Xenium counts
        z_mu, z_logvar = self.encode(x_dict, edge_index_dict) # dim: [L, K]
        n_queries = x_dict[self.query].size(0)

        with pyro.plate("batch", n_queries), poutine.scale(scale=self.configs.beta): 
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            pyro.sample("z", z_dist.to_event(1)) 

    def get_z(self, x_dict, edge_index_dict):
        # TODO: add attention weight retrieval
        x_dict[self.ref] = self.lognorm(x_dict[self.ref])
        z_mu, z_logvar = self.encode(x_dict, edge_index_dict)
        return z_mu, z_logvar
    
    def get_y(self, z):
        y, _ = self.decode(z)
        return y
        
    def predict(self, data: Data, device: torch.device):
        r"""Get latent representation & predictions from `pyg` Data object
        """
        self.eval()
        data = data.to(device)
        x = data.x_dict[self.ref]
        edge_index = data.edge_index_dict[self.edge_label]

        pz_x, _ = self.prior(x)
        pz = scatter_mean(pz_x[edge_index[0]], edge_index[1], dim=0) 
        qz_params = self.get_z(data.x_dict, data.edge_index_dict)
        py = self.get_y(qz_params[0])

        return ConfigDict({
            'qz_params':    qz_params,
            'pz':           pz,
            'py':           py
        })
    
    @torch.no_grad()
    def evaluate(
        self, 
        adata_ref: sc.AnnData,
        adata_query: sc.AnnData,
        k: int = 30, 
        n_subgraphs: int = 1, 
        device: torch.device = torch.device('cuda')
    ):
        r"""Get latent representation & predictions on subgraph batches"""

        self.eval()
        self.device = device
        self.to(device)
        self._move_attr_to(device)

        n_cells = adata_ref.shape[0]
        n_pixels, n_features = adata_query.shape

        graph_data = HeteroDataset(
            adata_ref, adata_query, k=k, n_subgraphs=n_subgraphs,
            ref_label=self.ref, query_label=self.query
        )
        dataloader = DataLoader(graph_data, shuffle=False)
        
        qz = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)  # lowres latent
        pz = np.zeros_like(qz)
        py = np.zeros((n_pixels, n_features), dtype=np.float32)

        # Recover batched predictions in the correct spatial orders
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qz = res.qz_params[0].detach().cpu().numpy()  # dim: [L, K]
            batch_pz = res.pz.detach().cpu().numpy()  
            batch_py = res.py.detach().cpu().numpy()

            query_indices = data[self.query].idx.numpy()
            qz[query_indices] = batch_qz
            pz[query_indices] = batch_pz
            py[query_indices] = batch_py

            # TODO: 'reverse' attention maps to get Xenium-level gradient predictions
            # ref_indices = data.x_dict[self.ref].idx.numpy()
        
        return ConfigDict({
            'qz':           qz,
            'pz':           pz,
            'py':           py
        })
    
    def init_lazy_modules(self, data):
        with torch.no_grad():
            _ = self.encode.attention(data.x_dict, data.edge_index_dict)

    @staticmethod
    def lognorm(x):
        l = x.sum(axis=-1, keepdim=True) + EPS
        x = x / l * l.median() 
        return torch.log1p(x)  
    
    @staticmethod
    def whiten(x):
        l = x.sum(axis=-1, keepdims=True) + EPS
        x = x / l * l.median()
        x = torch.log1p(x)
        x -= x.mean(0)

        cov = torch.cov(x.T)
        u, s, _ = torch.svd(cov)
        transform_matrix = u @ torch.diag(1 / torch.sqrt(s+EPS))
        return x @ transform_matrix


