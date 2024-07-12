import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from torch_geometric.nn import GCNConv,  SGConv, Sequential

EPS = 1e-6


class LogitVGAE(nn.Module):
    """
    Mixture latent VGAE with Logit Normal prior
    """
    def __init__(self, configs):
        super(LogitVGAE, self).__init__()
        self.configs = configs
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)
        self.pd_eps = torch.tensor(1e-6)

    def model(self, x, edge_index, cov):
        pyro.module("Logit_VGAE", self)
        l = x.sum(axis=-1, keepdim=True)
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        )

        z_mu = pyro.param("z_mu", torch.zeros(self.configs.c_latent))
        z_Sigma = torch.tensor(cov, dtype=torch.float)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z = pyro.sample(
                "z",
                dist.MultivariateNormal(z_mu, z_Sigma)
            )

            x_mu = l*self.decode(torch.softmax(z, axis=-1), edge_index)
            logits = torch.log(x_mu+EPS) - torch.log(theta + EPS)
            
            pyro.sample(
                "x",
                dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1),
                obs=x
            )

    def guide(self, x, edge_index, cov):
        pyro.module("Logit_VGAE", self)
        z_mu, z_Sigma = self.encode(x, edge_index)
        # z_Sigma = pyro.param(
        #     "z_Sigma",
        #     torch.tensor(cov, dtype=torch.float),
        #     constraint=dist.constraints.positive_definite
        # )

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            try:
                pyro.sample("z", dist.MultivariateNormal(z_mu, z_Sigma))
            except:
                pyro.sample("z", dist.MultivariateNormal(z_mu, self._PD_approx(z_Sigma)))

    def get_z(self, x, edge_index):
        self.eval()
        x = torch.tensor(x).float()
        ei = torch.tensor(edge_index)
        z_mu, z_Sigma = self.encode(x, ei)
        z_mu = torch.softmax(z_mu, dim=-1)
        return z_mu.detach().cpu().numpy(), z_Sigma.detach().cpu().numpy()
    
    def sample_z(self, x, edge_index, cov, n_samples=100):
        z_mu = self.get_z(x, edge_index)
        z_Sigma = torch.tensor(cov).float()
        z_samples = dist.MultivariateNormal(z_mu, z_Sigma).sample((n_samples,))
        return torch.softmax(z_samples, dim=-1)

    def get_x(self, x, edge_index, z):
        self.eval()
        l = torch.tensor(x).float().sum(axis=-1, keepdim=True)
        qz = torch.tensor(z).float()
        ei = torch.tensor(edge_index)

        px_mu = l*self.decode(qz, ei)
        return px_mu.detach().cpu().numpy()
    
    def sample_x(self, x, edge_index, cov, n_samples=100):
        self.eval()
        x = torch.tensor(x).float()
        ei = torch.tensor(edge_index)
        cov = torch.tensor(cov).float()

        predictive = pyro.infer.Predictive(self, self.guide, n_samples)
        pxs = predictive(x, ei, cov)
        return pxs["x"]
    
    def get_z_assignment(self, z):
        """Hard clustering assignment based on argmax value"""
        argmax_indices = z.argmax(1)
        z_hard = np.zeros_like(z)
        for i, idx in enumerate(argmax_indices):
            z_hard[i][idx] = 1
        return z_hard
    
    def _PD_approx(self, cov):
        eigvals, Q = torch.linalg.eigh(cov)
        Qt = Q.T
        L_prime = torch.diag(torch.tensor([torch.max(v, self.pd_eps) for v in eigvals]))
        return Q @ L_prime @ Qt

class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.c_latent = configs.c_latent
        self.row_idx, self.col_idx = torch.tril_indices(configs.c_latent, 
                                                configs.c_latent, 
                                                offset=-1)
        
        self.x_to_hid = SGConv(configs.c_in, configs.c_hidden, K=configs.k_hop)
        self.hid_to_zmu = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zlogvar = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_tril = SGConv(configs.c_hidden, len(self.row_idx), K=configs.k_hop)

        # TODO: X need two separate encoders for diag & off-diag. terms of L

    def forward(self, x, edge_index):
        h = self.x_to_hid(x, edge_index)
        z_mu = self.hid_to_zmu(h, edge_index)
        
        # TODO: try Global avg. pooling on \Sigma: [N, K, K] --> [K, K]? 
        z_logvar = self.hid_to_zlogvar(h, edge_index).mean(0)  # Cholesky diag. values
        z_tril = self.hid_to_tril(h, edge_index).mean(0)  # Cholesky off-diagonal values
        
        L = torch.zeros(self.c_latent, self.c_latent, dtype=torch.float)
        L[self.row_idx, self.col_idx] = z_tril
        L += torch.diag_embed(torch.exp(z_logvar) + EPS)

        z_Sigma = L @ L.T

        return z_mu, z_Sigma


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()        
        self.z_to_hid = Sequential('z, edge_index', [
            (SGConv(configs.c_latent, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
            nn.Dropout(p=configs.dropout)
        ])
        self.hid_to_xmu = Sequential('h, edge_index', [
            (SGConv(configs.c_hidden, configs.c_in, K=configs.k_hop), 'h, edge_index -> x_loc'),
            nn.Softplus(),
            nn.Dropout(p=configs.dropout)
        ])
        
    def forward(self, z, edge_index):
        h = self.z_to_hid(z, edge_index)
        x_mu = self.hid_to_xmu(h, edge_index)
        return x_mu
