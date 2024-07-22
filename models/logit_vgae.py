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
        # TODO [DEBUG]: p(z|u) representation unstable in multiple runs,
        # needs more than simple NN?
        super(LogitVGAE, self).__init__()
        self.configs = configs
        self.device = device
        self.pz_u = ConditionalPrior(configs)
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)
        self.to(device)

    def model(self, x, u_raw, u, edge_index):
        pyro.module("Logit_VGAE", self)
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)
        z_concentration = self.pz_u(u, edge_index)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            # z_mu = self.pz_u(u)
            # z_std = torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device)

            z = pyro.sample(
                "z",
                # dist.Normal(z_mu, z_std).to_event(1)
                dist.Dirichlet(z_concentration)
            )

            x_mu = l*self.decode(z, edge_index).softmax(-1)
            logits = torch.log(x_mu+EPS) - torch.log(theta + EPS)
            pyro.sample(
                "x",
                dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1),
                obs=x
            )

    def guide(self, x, u_raw, u, edge_index):
        pyro.module("Logit_VGAE", self)
        x = torch.log(x+EPS)
        # z_mu, z_logstd = self.encode(x, u_raw, edge_index)
        z_concentration = self.encode(x, u_raw, edge_index)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            pyro.sample(
                "z", 
                # dist.Normal(z_mu, z_logstd.exp()).to_event(1)
                dist.Dirichlet(z_concentration)
            )

    def get_cond_prior(self, u, edge_index, device='cpu'):
        u = torch.tensor(u).to(device)
        ei = torch.tensor(edge_index).to(device)
        return self.pz_u(u, ei)

    def get_z(self, x, u_raw, edge_index, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
        u_raw = torch.tensor(u_raw).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        # z_mu, z_logstd = self.encode(x, u_raw, ei)
        # return z_mu, z_logstd
        z_concentration = self.encode(x, u_raw, ei)
        return z_concentration
    
    def sample_z(self, x, u_raw, edge_index, n_samples=100):
        # z_mu, z_logstd = self.get_z(x, u_raw, edge_index)
        # z_samples = dist.Normal(z_mu, z_logstd.exp()).sample((n_samples,))
        z_concentration = self.get_z(x, u_raw, edge_index)
        z_samples = dist.Dirichlet(z_concentration).sample((n_samples,))
        return z_samples
    
    def get_x(self, x, edge_index, qz_conc, device='cpu'):
        self.eval()

        l = torch.tensor(x).float().sum(axis=-1, keepdim=True).to(device)
        # z_mu = torch.tensor(qz_mu).float().to(device)
        # z_logstd = torch.tensor(qz_logstd).float().to(device)
        z_concentration = torch.tensor(qz_conc).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        # z = dist.Normal(z_mu, z_logstd.exp()).sample()
        z = dist.Dirichlet(z_concentration).sample()
        px_mu = l*self.decode(z, ei).softmax(-1)
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
        
    def _PD_approx(self, cov, UPLO='L'):
        eigvals, Q = torch.linalg.eigh(cov @ cov.T) if UPLO == 'L' else torch.linalg.eigh(cov)
        Qt = Q.T
        Lambda = torch.diag(torch.tensor([torch.max(v, EPS) for v in eigvals]))
        return Q @ Lambda @ Qt


class ConditionalPrior(nn.Module):
    def __init__(self, configs):
        super(ConditionalPrior, self).__init__()
        self.layer = Sequential('u, edge_index', [
            (SGConv(configs.c_aux, configs.c_latent, K=configs.k_hop), 'u, edge_index -> z_conc'),
            nn.Softplus()
        ])

    def forward(self, x, edge_index):
        return self.layer(x, edge_index) + EPS


class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.uraw_to_u = nn.Sequential(
            nn.Linear(configs.c_u, configs.c_aux),
            nn.ReLU()
        )

        self.xu_to_hid = Sequential('x, edge_index', [
            (SGConv(configs.c_in+configs.c_aux, configs.c_hidden, K=configs.k_hop), 'x, edge_index -> h'),
            nn.ReLU()
        ])

        # self.hid_to_zmu = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        # self.hid_to_zlogstd = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zconc = Sequential('h, edge_index', [
            (SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop), 'h, edge_index -> z_conc'),
            nn.Softplus()
        ]) 
        

    def forward(self, x, u_raw, edge_index):
        u = self.uraw_to_u(u_raw)
        xu = torch.cat([x, u], dim=-1)
        h = self.xu_to_hid(xu, edge_index)

        # z_mu = self.hid_to_zmu(h, edge_index)
        # z_logstd = self.hid_to_zlogstd(h, edge_index)
        # return z_mu, z_logstd
        z_conc = self.hid_to_zconc(h, edge_index) + EPS
        return z_conc


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()        
        self.z_to_hid = Sequential('z, edge_index', [
            (SGConv(configs.c_latent, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
            nn.ReLU(),
            nn.Dropout(p=configs.dropout)
        ])
        self.hid_to_xmu = Sequential('h, edge_index', [
            (SGConv(configs.c_hidden, configs.c_in, K=configs.k_hop), 'h, edge_index -> x_loc'),
            nn.ReLU(),
            nn.Dropout(p=configs.dropout)
        ])
        
    def forward(self, z, edge_index):
        h = self.z_to_hid(z, edge_index)
        x_mu = self.hid_to_xmu(h, edge_index) + EPS
        return x_mu
