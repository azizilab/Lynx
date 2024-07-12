import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from torch_geometric.nn import SGConv, Sequential

EPS = 1e-6


class LogitVGAE(nn.Module):
    """
    Mixture latent VGAE with conditional Logistic Normal prior
    """
    def __init__(self, configs, device='cpu'):
        super(LogitVGAE, self).__init__()
        self.configs = configs
        self.device = device
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)
        self.to(device)

        self.pz_u = None 
        self.z_std = None

    def model(self, x, u, edge_index):
        pyro.module("Logit_VGAE", self)
        l = x.sum(axis=-1, keepdim=True)

        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        # Debug: try GNN?
        # self.pz_u = nn.Sequential(
        #     nn.Linear(self.configs.c_aux, self.configs.c_hidden),
        #     nn.ReLU(),
        #     nn.Linear(self.configs.c_hidden, self.configs.c_latent)
        # ).to(self.device)

        self.pz_u = Sequential('u, edge_index', [
            (SGConv(self.configs.c_aux, self.configs.c_hidden, K=self.configs.k_hop), 'u, edge_index -> h'),
            nn.ReLU(),
            (SGConv(self.configs.c_hidden, self.configs.c_latent, K=self.configs.k_hop), 'h, edge_index -> z')
        ]).to(self.device)

        z_mu = self.pz_u(u, edge_index) 
        z_std = torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z = pyro.sample(
                "z",
                dist.LogisticNormal(z_mu, z_std)
            )

            x_mu = l*self.decode(z, edge_index).exp()
            logits = torch.log(x_mu+EPS) - torch.log(theta + EPS)
            pyro.sample(
                "x",
                dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1),
                obs=x
            )

    def guide(self, x, u, edge_index):
        pyro.module("Logit_VGAE", self)
        x = torch.log(x+EPS)
        z_mu = self.encode(x, u, edge_index)
        self.z_std = pyro.param(
            "z_std",
            torch.ones(self.configs.c_latent, device=self.device)
        )
        
        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            pyro.sample("z", dist.LogisticNormal(z_mu, self.z_std))

    def get_z(self, x, u, edge_index, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
        u = torch.tensor(u).float().to(device)
        ei = torch.tensor(edge_index).to(device)
        
        z_mu = self.encode(x, u, ei)
        z = self._add_logistic_transform(z_mu)  # project onto K-dim simplex
        return z, z_mu
    
    def sample_z(self, x, u, edge_index, n_samples=100, device='cpu'):
        self.z_std = self.z_std.to(device)
        _, z_mu = self.get_z(x, u, edge_index)
        z_samples = dist.LogisticNormal(z_mu, self.z_std).sample((n_samples,))
        return z_samples
    
    def get_x(self, x, edge_index, qz_mu, device='cpu'):
        self.eval()
        self.z_std = self.z_std.to(device)

        l = torch.tensor(x).float().sum(axis=-1, keepdim=True).to(device)
        z_mu = torch.tensor(qz_mu).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        z = dist.LogisticNormal(z_mu, self.z_std).sample()
        px_mu = l*self.decode(z, ei).exp()
        return px_mu
    
    def sample_x(self, x, u, edge_index, n_samples=100, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
        u = torch.tensor(u).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        predictive = pyro.infer.Predictive(self, self.guide, n_samples)
        pxs = predictive(x, u, ei)
        return pxs["x"]
    
    def get_z_assignment(self, z):
        """Hard clustering assignment based on argmax value"""
        argmax_indices = z.argmax(1)
        z_hard = np.zeros_like(z)
        for i, idx in enumerate(argmax_indices):
            z_hard[i][idx] = 1
        return z_hard
    
    @staticmethod
    def _add_logistic_transform(y):
        """
        Additive Logistic Transform:
        Map (K-1)-dim means to K-dim simplex
        """
        denom = 1+torch.exp(y).sum(1, keepdim=True)
        x0 = torch.exp(y) / denom   # first K-1 dims
        x = torch.cat([x0, 1/denom], axis=1)
        return x
    
    def _PD_approx(self, cov, UPLO='L'):
        eigvals, Q = torch.linalg.eigh(cov @ cov.T) if UPLO == 'L' else torch.linalg.eigh(cov)
        Qt = Q.T
        Lambda = torch.diag(torch.tensor([torch.max(v, EPS) for v in eigvals]))
        return Q @ Lambda @ Qt


class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.c_latent = configs.c_latent

        self.xu_to_hid = Sequential('xu, edge_index', [
            (SGConv(configs.c_in+configs.c_aux, configs.c_hidden, K=configs.k_hop), 'xu, edge_index -> h'),
            nn.ReLU()
        ])
        self.hid_to_zmu = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        # self.hid_to_zlogstd = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)

    def forward(self, x, u, edge_index):
        xu = torch.cat([x, u], dim=-1)
        h = self.xu_to_hid(xu, edge_index)
        z_mu = self.hid_to_zmu(h, edge_index)
        # z_logstd = self.hid_to_zlogstd(h, edge_index)

        return z_mu


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()        
        self.z_to_hid = Sequential('z, edge_index', [
            (SGConv(configs.c_latent+1, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
            nn.ReLU(),
            nn.Dropout(p=configs.dropout)
        ])
        self.hid_to_xmu = Sequential('h, edge_index', [
            (SGConv(configs.c_hidden, configs.c_in, K=configs.k_hop), 'h, edge_index -> x_loc'),
            nn.Dropout(p=configs.dropout)
        ])
        
    def forward(self, z, edge_index):
        h = self.z_to_hid(z, edge_index)
        x_mu = self.hid_to_xmu(h, edge_index)
        return x_mu
