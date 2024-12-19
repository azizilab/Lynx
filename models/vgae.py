import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from ml_collections import ConfigDict
from pyro.contrib.zuko import ZukoToPyro
from torch_geometric.loader import DataLoader
from zuko import flows


sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from module import ConditionalPrior, Encoder, FlowEncoder, AggregateEncoder, SingleViewEncoder
from module import Decoder, AggregateDecoder
from dataset import XeniumDataset

EPS = 1e-8


class VGAE(nn.Module):
    r"""Learning latent manifold w/ Conditional VGAE
    U (DESI) -> Z (latent) -> X (Xenium)
    """
    def __init__(self, configs, device='cuda'):
        super(VGAE, self).__init__()
        self.configs = configs
        self.device = device

        self.pz_u = ConditionalPrior(configs)
        self.encode = Encoder(configs)
        # self.encode = SingleViewEncoder(configs)
        self.decode = Decoder(configs)

        self.to(device)

    def model(self, x, u, s, edge_index):
        pyro.module("prior", self.pz_u)
        pyro.module("decoder", self.decode)

        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z_mu = self.pz_u(u, edge_index)
            # z_mu = torch.zeros(self.configs.c_latent, dtype=torch.float, device=self.device)
            z_std = torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device)
            z_dist = dist.Normal(z_mu, z_std)
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
        z_mu, z_logvar, _ = self.encode(x, u, s, edge_index) # Global parameters per subgraph

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
        z_logvar = z_param[1]
        z = dist.Normal(z_mu, torch.exp(z_logvar/2)).sample()
            
        mu  = self.decode(z, s, edge_index)
        px_mu = l * mu
        return px_mu
    
    def sample_x(self, x, u, edge_index, n_samples=100):
        self.eval()
        x = torch.tensor(x).float()
        x = torch.log(x + EPS)
        u = torch.tensor(u).float()
        ei = torch.tensor(edge_index)

        predictive = pyro.infer.Predictive(self, self.guide, n_samples)
        pxs = predictive(x, u, ei)
        return pxs["x"]
    
    def predict(self, data, device):
        r"""Get latent representation & predictions on full data"""
        self.eval()
        x = data.x.to(device).float()
        u = data.u.to(device).float()
        s = data.s.to(device).float()
        edge_index = data.edge_index.to(device)

        pz = self.pz_u(u, edge_index)
        qz_params = self.get_z(x, u, s, edge_index)
        x_mu = self.get_x(x, s, edge_index, qz_params)

        return ConfigDict({
            'qz_params':    qz_params,
            'pz':           pz,
            'px':           x_mu
        })
    
    def evaluate(self, adata, k=30, n_subgraphs=8, device=torch.device('cuda')):
        r"""Get latent representation & predictions on subgraph batches"""
        self.eval()
        self.device = device
        self.to(device)
        self._move_attr_to(device)

        position_map = {
            tuple(pos): i
            for i, pos in enumerate(
                adata.obs[['x_centroid', 'y_centroid']].values.astype(np.float32)
            )
        }
        graph_data = XeniumDataset(
            k=k, n_subgraphs=n_subgraphs
        ).load_graphs([adata])

        dataloader = DataLoader(graph_data, shuffle=False)
        qz = np.zeros((adata.shape[0], self.configs.c_latent), dtype=np.float32)
        pz = np.zeros_like(qz)
        px = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qz = res.qz_params[0].detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()

            for pos, qz_i, pz_i, px_i in zip(data.pos, batch_qz, batch_pz, batch_px):
                idx = position_map[tuple(pos.detach().cpu().numpy().astype(np.float32))]
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


class FlowVGAE(VGAE):
    r"""Learning latent manifold w/ Conditional VGAE 
    (flow-based prior & posterior) 
    """
    def __init__(self, configs=torch.device('cuda')):
        super(FlowVGAE, self).__init__(configs)

        self.pz_u = flows.MAF(
            features=self.configs.c_latent,
            context=self.configs.c_aux,
            hidden_features=(32, 32),
            activation=nn.SiLU
        )
        
        self.encode = FlowEncoder(configs)
        
        self.qz_h = flows.MAF(
            features=self.configs.c_latent,
            context=self.configs.c_hidden,
            hidden_features=(32, 32), 
            activation=nn.SiLU
        )

    def model(self, x, u, s, edge_index):
        pyro.module("prior", self.pz_u)
        pyro.module("decoder", self.decode)

        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z = pyro.sample("z", ZukoToPyro(self.pz_u(u)))
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
        h = self.encode(x, u, s, edge_index)  # Global per subgraph

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta): 
            pyro.sample("z", ZukoToPyro(self.qz_h(h)))

    def get_z(self, x, u, s, edge_index):
        if self.configs.embed_option == 'attn':
            l = x.sum(axis=-1, keepdim=True)
            x = x / l * l.median()
        x = torch.log1p(x)

        h = self.encode(x, u, s, edge_index)
        z_mu = self.qz_h(h).sample((100,)).mean(0)
        return (z_mu,)
    
    def get_x(self, x, s, edge_index, z_param):
        self.eval()
        l = x.sum(axis=-1, keepdim=True)
        z = z_param[0]            
        mu  = self.decode(z, s, edge_index)
        px_mu = l * mu
        return px_mu
    
    def predict(self, data, device):
        r"""Get latent representation & predictions on full data"""
        self.eval()
        x = data.x.to(device).float()
        u = data.u.to(device).float()
        s = data.s.to(device).float()
        edge_index = data.edge_index.to(device)

        pz = self.pz_u(u).sample((100,)).mean(0)
        qz_params = self.get_z(x, u, s, edge_index)
        x_mu = self.get_x(x, s, edge_index, qz_params)

        return ConfigDict({
            'qz_params':    qz_params,
            'pz':           pz,
            'px':           x_mu
        })
    

class MultiscaleVGAE(VGAE):
    r"""Learning latent manifold w/ Conditional VGAE (normal likelihood) 
    X (Xenium) -> Z (latent) -> Y (DESI)
    """
    def __init__(self, configs, device=torch.device('cuda')):
        super(MultiscaleVGAE, self).__init__(configs)
        self.configs = configs
        self.device = device
        
        self.pz_x = ConditionalPrior(configs)
        self.encode = AggregateEncoder(configs)
        self.decode = AggregateDecoder(configs)
        self.to(device)

    def model(self, x, y, s, edge_index, cell_pixel_map):
        pyro.module("prior", self.pz_x)
        pyro.module("decoder", self.decode)

        x = self.__lognorm(x) # Normalize Xenium counts

        with pyro.plate("batch", y.size(0)), poutine.scale(scale=self.configs.beta):
            z_mu = self.pz_x(x, edge_index)
            z_std = torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device)
            z = pyro.sample("z", dist.Normal(z_mu, z_std).to_event(1))

            y_mu, y_logvar = self.decode(z, s, edge_index, cell_pixel_map)
            normal_dist = dist.Normal(y_mu, torch.exp(y_logvar//2))
            pyro.sample("y", normal_dist.to_event(1), obs=y)

    def guide(self, x, y, s, edge_index, cell_pixel_map):
        # TODO: attention btw `x` & `y`
        # dim(x) - [N, G], dim(y) - [L, M]

        pyro.module("encoder", self.encode)

        x = self.__lognorm(x) # Normalize Xenium counts
        z_mu, z_logvar = self.encode(
            x, y, s, edge_index, cell_pixel_map
        )  # Global parameters per subgraph 

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta): 
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            pyro.sample("z", z_dist.to_event(1)) 


    def __lognorm(self, x):
        l = x.sum(axis=-1, keepdim=True)
        x = x / l * l.median()
        return torch.log1p(x)  
    








