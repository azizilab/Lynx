import os
import sys
import numpy as np
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from ml_collections import ConfigDict
from typing import Dict, List

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean


sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from module import ConditionalPrior, GPCALayer
from module import Encoder, FlowEncoder, AggregateEncoder
from module import Decoder, AggregateDecoder, SpikeSlabLassoDecoder
from dataset import XeniumDataset, MultiscaleDataset, MultiscaleDatasetJosh

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
        pyro.module("VGAE", self)

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
        pyro.module("VGAE", self)

        if self.configs.embed_option == 'attn':
            l = x.sum(axis=-1, keepdim=True)
            x = x / l * l.median()

        x = torch.log1p(x)
        z_mu, z_logvar, _ = self.encode(x, u, s, edge_index)  # Global sample per subgraph

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
                

class MultiscaleVGAE(nn.Module):
    r"""Learning latent manifold w/ Conditional VGAE (normal likelihood) 
    X (Xenium) -> Z (latent) -> Y (DESI)
    """
    def __init__(self, configs, device=torch.device('cuda')):
        super(MultiscaleVGAEJosh, self).__init__()
        self.configs = configs
        self.device = device

        self.prior = ConditionalPrior(configs)
        self.encode = AggregateEncoder(configs)
        self.decode = AggregateDecoder(configs)


    def model(self, x, y, edge_index, neighbors):

        pyro.module("VAE", self)

        # Normalize Xenium counts
        x = self.__lognorm(x) 
        edge_index = edge_index

        with pyro.plate("batch", y.size(0)):
            z_mu, z_logvar = self.prior(x, edge_index, neighbors)

            z_dist = dist.Normal(z_mu, torch.exp(z_logvar))

            z = pyro.sample("z", z_dist.to_event(1))
            
            y_mu, y_logvar = self.decode(z)

            normal_dist = dist.Normal(y_mu, torch.exp(y_logvar))
        # normal_dist = dist.Normal(y_mu, torch.exp(y_logvar)).to_event(1)
            pyro.sample("y", normal_dist.to_event(1), obs=y)

    def guide(self, x, y, edge_index, neighbors):

        pyro.module("VAE", self)
        
        # Normalize Xenium counts
        x = self.__lognorm(x) 

        z_mu, z_logvar, _ = self.encode(
            x, y, neighbors
        )  

        with pyro.plate("batch", y.size(0)): 
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar))
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("z", z_dist.to_event(1)) 

        # if isinstance(self.decode, SpikeSlabLassoDecoder):
        #     #Slab Slab Weight
        #     w_shape = self.decode.z_to_hid.weight.shape
        #     w_mu = pyro.param(
        #         "z_to_hid_weight_mu",
        #         torch.zeros(w_shape, device=self.device)
        #     )
        #     w_sigma = pyro.param(
        #         "z_to_hid_weight_sigma",
        #         0.1 * torch.ones(w_shape, device=self.device),
        #         constraint=constraints.positive
        #     )
        #     pyro.sample(
        #         "z_to_hid.weight",
        #         dist.Normal(w_mu, w_sigma).to_event(len(w_shape))
        #     )

        #     #Spike Slab Lasoo
        #     b_shape = self.decode.z_to_hid.bias.shape
        #     b_mu = pyro.param(
        #         "z_to_hid_bias_mu",
        #         torch.zeros(b_shape, device=self.device)
        #     )
        #     b_sigma = pyro.param(
        #         "z_to_hid_bias_sigma",
        #         0.1 * torch.ones(b_shape, device=self.device),
        #         constraint=constraints.positive
        #     )
        #     pyro.sample(
        #         "z_to_hid.bias",
        #         dist.Normal(b_mu, b_sigma).to_event(len(b_shape))
        #     )


    def get_z(self, x, y, neighbors):
        # Normalize Xenium counts
        x = self.__lognorm(x) 

        z_mu, z_logvar, attn_scores = self.encode(
            x, y, neighbors
        )  
        return z_mu, z_logvar, attn_scores
    
    def get_y(self, z):
        # Note: linear-layer decoder:
        y, _ = self.decode(z)
        return y
        
    def predict(self, data: Data, device: torch.device):
        r"""Get latent representation & predictions from `pyg` Data object
        Note: 
            data.x & data.y aren't ordered the same spatially
            data.y is ordered based on the sorted `cluster_id`
            See `dataset.MultiscaleDataset` for further details
        """
        self.eval()
        x = data.x.to(device).float()
        y = data.y.to(device).float()
        neighbors = data.neighbors.to(device).long()
        edge_index = data.edge_index.contiguous().to(device)

        pz_x, _ = self.prior(x, edge_index, neighbors)
        qz_xy_params = self.get_z(x, y, neighbors)
        py_z = self.get_y(qz_xy_params[0])

        return ConfigDict({
            'qz_params':    qz_xy_params,
            'pz':           pz_x,
            'py':           py_z
        })
    
    def evaluate(
        self, 
        adata_hires: sc.AnnData,
        adata_lowres: sc.AnnData,
        k: int = 10, 
        n_subgraphs: int = 8, 
        device: torch.device = torch.device('cuda')
    ):
        r"""Get latent representation & predictions on subgraph batches"""
        self.eval()
        self.device = device
        self.to(device)

        

        n_cells = adata_hires.shape[0]
        n_pixels, n_features = adata_lowres.shape

        graph_data = MultiscaleDatasetJosh(
            k=k, n_subgraphs=n_subgraphs
        ).load_graphs([adata_hires], [adata_lowres])

        dataloader = DataLoader(graph_data, shuffle=False)
        qzy = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)  # lowres latent
        qzx = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)   # hires latent
        pz = np.zeros_like(qzy)
        py = np.zeros((n_pixels, n_features), dtype=np.float32)
        attn = np.zeros((n_pixels, k), dtype=np.float32)

        # Temporary accumulators for weighted averages
        qzx_weighted_sum = np.zeros_like(qzx)
        qzx_attention_sum = np.zeros((n_cells), dtype=np.float32)

        # Recover batched predictions in correct spatial orders
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qzy = res.qz_params[0].detach().cpu().numpy()  # dim: [L, K]
            batch_attn = res.qz_params[2].detach().cpu().numpy() # dim: [L, K]
            batch_pz = res.pz.detach().cpu().numpy()  
            batch_py = res.py.detach().cpu().numpy()

            qzy[data.desi_idx] = batch_qzy
            attn[data.desi_idx] = batch_attn
            pz[data.desi_idx] = batch_pz
            py[data.desi_idx] = batch_py

            # Compute highres representations
            # Weighted sum for each neighbor
            for i, neighbors in enumerate(data.neighbors):  # Iterate over L
                # neighbors dim : k
                
                xenium_idx = data.xenium_idx[neighbors]

                # Update accumulators for highres
                qzx_weighted_sum[xenium_idx] += batch_attn[i, :, None] * batch_qzy[i]  # [k, latent_dim]
                qzx_attention_sum[xenium_idx] += batch_attn[i]  # [k]


        if not np.all(qzx_attention_sum > 0):
            raise AssertionError("Not all cells have mapped pixels!")

        valid = qzx_attention_sum > 0

        # Average highres latent representations
        qzx[valid.squeeze()] = qzx_weighted_sum[valid.squeeze()] / qzx_attention_sum[valid.squeeze(), None]

        
        return ConfigDict({
            'qzx':          qzx,
            'qzy':          qzy,
            'pz':           pz,
            'py':           py,
        })

    def __lognorm(self, x):
        l = x.sum(axis=-1, keepdim=True) + EPS
        x = x / l * l.median() 
        return torch.log1p(x)  


class AutoencoderJosh(nn.Module):
    r"""Deterministic autoencoder for debugging purposes.
    X (input) -> Z (latent) -> Y (reconstruction)
    """
    def __init__(self, configs, device=torch.device('cuda')):
        super(AutoencoderJosh, self).__init__()
        self.configs = configs
        self.device = device

        # Encoder and Decoder
        self.encode = AggregateEncoder(configs)
        self.decode = AggregateDecoder(configs)
        self.test_encoder = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            nn.ReLU(),
            nn.Linear(configs.c_hidden, configs.c_latent),
        )
        self.test_decoder = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            nn.ReLU(),
            nn.Linear(configs.c_hidden, configs.c_in),
        )
        self.to(device)

    def __lognorm(self, x):
        # Dummy normalization for example; replace with the actual implementation
        return x / x.sum(dim=1, keepdim=True)

    def forward(self, x, y, neighbors):
        """
        Perform a deterministic autoencoder forward pass.
        Args:
            x (torch.Tensor): Input features.
            y (torch.Tensor): Target features.
            neighbors (torch.Tensor): Neighbor information.
        
        Returns:
            torch.Tensor: Reconstruction of y.
        """
        # Normalize input
        x = self.__lognorm(x)
        # Encoder step
        z, _, attn_scores = self.encode(x, y, neighbors)

        # Decoder step
        y_recon, _ = self.decode(z)

        return y_recon, attn_scores