import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from torch_geometric.nn import GCNConv, ChebConv, Sequential

EPS = 1e-15


class LogitVGAE(nn.Module):
    """
    Mixture latent VGAE with Logit Normal prior
    """
    def __init__(self, configs):
        super(LogitVGAE, self).__init__()
        self.configs = configs
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)

    def model(self, x, edge_index, cov):
        pyro.module("Logit_VGAE", self)
        l = x.sum(axis=-1, keepdim=True)
        
        self.z_mu = pyro.param(
            "mu_0",
            torch.ones(self.configs.c_latent),
        )
        
        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        )
        
        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z = pyro.sample(
                "z", 
                dist.MultivariateNormal(self.z_mu, self.z_Sigma)
            )
            
            x_mu = self.decode(torch.softmax(z, axis=-1), edge_index)
            logits = torch.log1p(l*x_mu+EPS) - torch.log1p(self.theta+EPS)
            
            pyro.sample(
                "x",
                dist.NegativeBinomial(total_count=self.theta+EPS, logits=logits).to_event(1),
                obs=x
            )

    def guide(self, x, edge_index, cov):
        pyro.module("Logit_VGAE", self)

        self.z_Sigma = pyro.param(
            "Sigma",
            torch.tensor(cov, dtype=torch.float),
            constraint=dist.constraints.positive_definite
        )
        
        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            # x = torch.log1p(x)
            qz_mu = self.encode(x, edge_index)
            z_mu = qz_mu.mean(0)
            pyro.sample("z", dist.MultivariateNormal(z_mu, self.z_Sigma))

    def sample_z(self, x, edge_index, cov, n_samples=100):
        self.eval()

        x = torch.tensor(x).float()
        Sigma = torch.tensor(cov).float()
        ei = torch.tensor(edge_index)    
        
        z_mu = self.encode(x, ei)
        z_samples = dist.MultivariateNormal(z_mu, torch.tensor(Sigma)).sample((n_samples,))
        z = torch.softmax(z_samples.mean(0), dim=-1).detach().cpu().numpy()  # Additive logit transformation

        return z
    
    def sample_px(self, x, edge_index, z, n_samples=100):
        self.eval()

        x = torch.tensor(x).float()
        ei = torch.tensor(edge_index)
        l = x.sum(axis=-1, keepdim=True)
        qz = torch.tensor(z).float()

        px_mu = self.decode(qz, ei)
        logits = torch.log1p(l*px_mu+EPS) - torch.log1p(self.theta+EPS)
        px_samples = dist.NegativeBinomial(total_count=self.theta+EPS, logits=logits).sample((n_samples,))
        
        return px_samples.mean(0).detach().cpu().numpy()
    
    def get_z_assignment(self, z):
        """Hard clustering assignment based on argmax value"""
        argmax_indices = z.argmax(1)
        z_hard = np.zeros_like(z)
        for i, idx in enumerate(argmax_indices):
            z_hard[i][idx] = 1
        return z_hard


class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        
        # # Single-layer
        # self.x_to_zmu = ChebConv(configs.c_in, configs.c_latent, K=configs.k_hop)

        self.x_to_hid = ChebConv(configs.c_in, configs.c_hidden, K=configs.k_hop)
        self.hid_to_zmu = ChebConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)

    def forward(self, x, edge_index):
        # # Single-layer
        # z_mu = self.x_to_zmu(x, edge_index)

        x = torch.log1p(x)  # Mute if processed upon preprocessing
        h = self.x_to_hid(x, edge_index)
        z_mu = self.hid_to_zmu(h, edge_index)
        return z_mu


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()
        
        # # Single-layer
        # self.z_to_xmu = Sequential('z, edge_index', [
        #     (ChebConv(configs.c_latent, configs.c_in, K=configs.k_hop), 'z, edge_index -> x_mu'),
        #     nn.Softplus(),
        #     nn.Dropout(p=configs.dropout)
        # ])
        
        self.z_to_hid = Sequential('z, edge_index', [
            (ChebConv(configs.c_latent, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
            nn.Dropout(p=configs.dropout)
        ])
        self.hid_to_xmu = Sequential('h, edge_index', [
            (ChebConv(configs.c_hidden, configs.c_in, K=configs.k_hop), 'h, edge_index -> x_loc'),
            nn.Softplus(),
            nn.Dropout(p=configs.dropout)
        ])
        
    def forward(self, z, edge_index):
        # # Single-layer
        # x_mu = self.z_to_xmu(z, edge_index)
        
        h = self.z_to_hid(z, edge_index)
        x_mu = self.hid_to_xmu(h, edge_index)
        return x_mu
    
