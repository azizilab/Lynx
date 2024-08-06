import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from ml_collections import ConfigDict
from pyro.infer.reparam import ProjectedNormalReparam
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
        self.prior_dist = configs.prior
        self.pz_u = ConditionalPrior(configs)
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)
        self.to(device)

        assert self.prior_dist == 'normal' or self.prior_dist == 'vMF', \
            """Prior distribution type {} not implemented yet\n
               Please choose from `normal` & `vMF`""".format(self.prior_dsit)

    def model(self, x, u, edge_index):
        pyro.module("Logit_VGAE", self)
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)
        if self.prior_dist == 'vMF':
            z_concentration = self.pz_u(u)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            if self.prior_dist == 'normal':
                z_mu = self.pz_u(u)
                z_std = torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device)
                z = pyro.sample(
                    "z",
                    dist.Normal(z_mu, z_std).to_event(1)
                )
            else:
                z = self._sample_von_mise_fisher(z_concentration)

            x_mu = l * (self.decode(z, edge_index).softmax(dim=-1))
            logits = torch.log(x_mu+EPS) - torch.log(theta + EPS)
            pyro.sample(
                "x",
                dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1),
                obs=x
            )

    def guide(self, x, u, edge_index):
        pyro.module("Logit_VGAE", self)
        x = torch.log(x+EPS)
        z_param = self.encode(x, u, edge_index)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            if self.prior_dist == 'normal':
                z_mu, z_logstd = z_param
                pyro.sample(
                    "z", 
                    dist.Normal(z_mu, z_logstd.exp()).to_event(1)
                )
            else:
                self._sample_von_mise_fisher(z_param)

    @poutine.reparam(config={"z": ProjectedNormalReparam()})
    def _sample_von_mise_fisher(concentration):
        return pyro.sample(
            "z",
            dist.ProjectedNormal(concentration)
        )

    def get_cond_prior(self, u, edge_index, device='cpu'):
        u = torch.tensor(u).to(device)
        return self.pz_u(u)

    def get_z(self, x, u, edge_index, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
        u = torch.tensor(u).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        if self.prior_dist == 'normal':
            z_mu, z_logstd = self.encode(x, u, ei)
            return z_mu, z_logstd
        else:
            z_concentration = self.encode(x, u, ei)
            return z_concentration
    
    def sample_z(self, x, u, edge_index, n_samples=100):
        if self.prior_dist == 'normal':
            z_mu, z_logstd = self.get_z(x, u, edge_index)
            z_samples = dist.Normal(z_mu, z_logstd.exp()).sample((n_samples,))
        else:
            z_concentration = self.get_z(x, u, edge_index)
            z_samples = dist.ProjectedNormal(z_concentration).sample((n_samples,))
        return z_samples
    
    def get_x(self, x, edge_index, z_param, device='cpu'):
        self.eval()

        l = torch.tensor(x).float().sum(axis=-1, keepdim=True).to(device)
        ei = torch.tensor(edge_index).to(device)
        if self.prior_dist == 'normal':
            z_mu = torch.tensor(z_param[0]).float().to(device)
            z_logstd = torch.tensor(z_param[1]).float().to(device)
            z = dist.Normal(z_mu, z_logstd.exp()).sample()
        else:
            z_concentration = torch.tensor(z_param).float().to(device)
            z = dist.ProjectedNormal(z_concentration).sample()
            
        px_mu = l * (self.decode(z, ei).softmax(dim=-1))
        return px_mu
    
    def sample_x(self, x, u, edge_index, n_samples=100, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
        u = torch.tensor(u).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        predictive = pyro.infer.Predictive(self, self.guide, n_samples)
        pxs = predictive(x, u, ei)
        return pxs["x"]
    
    def predict(self, x, u, edge_index, device='cpu'):
        x = torch.tensor(x).float().to(device)
        u = torch.tensor(u).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        pz = self.get_cond_prior(u, ei)
        qz_params = self.get_z(x, u, ei)
        px = self.get_x(x, ei, qz_params)

        return ConfigDict({
            'qz_params':    qz_params,
            'pz':           pz,
            'px':           px
        })
        
    def _PD_approx(self, cov, UPLO='L'):
        eigvals, Q = torch.linalg.eigh(cov @ cov.T) if UPLO == 'L' else torch.linalg.eigh(cov)
        Qt = Q.T
        Lambda = torch.diag(torch.tensor([torch.max(v, EPS) for v in eigvals]))
        return Q @ Lambda @ Qt


class ConditionalPrior(nn.Module):
    def __init__(self, configs):
        super(ConditionalPrior, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            nn.SiLU(),
            nn.Linear(configs.c_hidden, configs.c_latent),
        )

    def forward(self, x):
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.prior_dist = configs.prior
        self.xu_to_hid = Sequential('x, edge_index', [
            (SGConv(configs.c_in+configs.c_aux, configs.c_hidden, K=configs.k_hop), 'x, edge_index -> h'),
            nn.SiLU()
        ])
        # self.x_to_hid = Sequential('x, edge_index', [
        #     (SGConv(configs.c_in, configs.c_hidden, K=configs.k_hop), 'x, edge_index -> h'),
        #     nn.SiLU()
        # ])
        # self.u_to_hid = Sequential('u, edge_index', [
        #     (SGConv(configs.c_aux, configs.c_hidden, K=configs.k_hop), 'u, edge_index -> h'),
        #     nn.SiLU()
        # ])

        self.hid_to_zmu = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zlogstd = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zconc = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        

    def forward(self, x, u, edge_index):
        xu = torch.cat([x, u], dim=-1)
        h = self.xu_to_hid(xu, edge_index)
        # hx = self.x_to_hid(x, edge_index)
        # hu = self.u_to_hid(u, edge_index)
        # h = F.scaled_dot_product_attention(hu, hx, hx)

        if self.prior_dist == 'normal':
            z_mu = self.hid_to_zmu(h, edge_index)
            z_logstd = self.hid_to_zlogstd(h, edge_index)
            return z_mu, z_logstd
        else:
            z_conc = self.hid_to_zconc(h, edge_index)
            return z_conc


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()        
        self.z_to_hid = Sequential('z, edge_index', [
            (SGConv(configs.c_latent, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
            nn.SiLU(),
            nn.Dropout(p=configs.dropout)
        ])
        self.hid_to_xmu = Sequential('h, edge_index', [
            (SGConv(configs.c_hidden, configs.c_in, K=configs.k_hop), 'h, edge_index -> x_loc'),
            nn.SiLU(),
            nn.Dropout(p=configs.dropout)
        ])
        
    def forward(self, z, edge_index):
        h = self.z_to_hid(z, edge_index)
        x_mu = self.hid_to_xmu(h, edge_index) + EPS
        return x_mu
