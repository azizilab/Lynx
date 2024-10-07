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
from pyro.infer.reparam import ProjectedNormalReparam
from torch_geometric.nn import SGConv, Sequential
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dataset import XeniumGraphDataset

EPS = 1e-8


class VGAE(nn.Module):
    """
    Conditional VGAE to learn Latent Manifold 
    """
    def __init__(self, configs, device='cpu'):
        super(VGAE, self).__init__()
        self.configs = configs
        self.device = device

        self.prior_dist = configs.prior 
        self.pz_u = ConditionalPrior(configs)
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)

        self.to(device)

        assert self.prior_dist == 'normal' or self.prior_dist == 'vMF', \
            """Prior distribution type {} not implemented yet\n
               Please choose from `normal` & `vMF`""".format(self.prior_dist)

    def model(self, x, u, s, edge_index):
        pyro.module("VGAE", self)
        self.theta = pyro.param(
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

            mu = self.decode(z, s, edge_index)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - (self.theta).log()

            nb_dist = dist.NegativeBinomial(
                total_count=self.theta,
                logits=logits
            )

            pyro.sample(
                "x",
                nb_dist.to_event(1),
                obs=x
            )

    def guide(self, x, u, s, edge_index):
        pyro.module("Logit_VGAE", self)
        x = torch.log(x+EPS)
        z_param = self.encode(x, u, s, edge_index)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            if self.prior_dist == 'normal':
                z_mu, z_logvar = z_param
                z_std = torch.exp(z_logvar/2)
                pyro.sample(
                    "z", 
                    dist.Normal(z_mu, z_std).to_event(1)
                )
            else:
                self._sample_von_mise_fisher(z_param)

    @poutine.reparam(config={"z": ProjectedNormalReparam()})
    def _sample_von_mise_fisher(concentration):
        return pyro.sample(
            "z",
            dist.ProjectedNormal(concentration)
        )

    def get_cond_prior(self, u, device='cpu'):
        u = torch.tensor(u).to(device)
        return self.pz_u(u)

    def get_z(self, x, u, s, edge_index, device='cpu'):
        x = torch.log(x+EPS).to(device) 
        u = u.to(device)
        s = s.to(device)
        edge_index = edge_index.to(device)

        if self.prior_dist == 'normal':
            z_mu, z_logvar = self.encode(x, u, s, edge_index)
            return z_mu, z_logvar
        else:
            z_concentration = self.encode(x, u, s, edge_index)
            return z_concentration
    
    def sample_z(self, x, u, s, edge_index, n_samples=100):
        if self.prior_dist == 'normal':
            z_mu, z_logvar = self.get_z(x, u, s, edge_index)
            z_samples = dist.Normal(z_mu, torch.exp(z_logvar//2)).sample((n_samples,))
        else:
            z_concentration = self.get_z(x, u, s, edge_index)
            z_samples = dist.ProjectedNormal(z_concentration).sample((n_samples,))
        return z_samples
    
    def get_x(self, x, s, edge_index, z_param, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
        l = x.sum(axis=-1, keepdim=True)
        edge_index = edge_index.to(device)

        if self.prior_dist == 'normal':
            z_mu = z_param[0].to(device)
            z_logvar = z_param[1].to(device)
            z = dist.Normal(z_mu, torch.exp(z_logvar/2)).sample()
        else:
            z_conc = z_param.to(device)
            z = dist.ProjectedNormal(z_conc).sample()
            
        mu  = self.decode(z, s, edge_index)
        px_mu = l * mu
        return px_mu
    
    def sample_x(self, adata, edge_index, n_samples=100, device='cpu'):
        self.eval()
        x = torch.tensor(adata.X.A).float().to(device)
        u = torch.tensor(adata.obsm['X_aux']).float().to(device)
        s = torch.tensor(adata.obsm['X_s']).float().to(device) if 'X_s' in adata.obsm_keys() else \
            torch.empty(size=(0,)).to(device)
        edge_index = edge_index.to(device)

        predictive = pyro.infer.Predictive(self, self.guide, n_samples)
        pxs = predictive(x, u, s, edge_index)
        return pxs["x"]
    
    def predict(self, data, device=torch.device('cpu')):
        """
        Predict latent representation & reconstructions 
        on full data
        """
        self.eval()
        x = data.x.to(device).float()
        u = data.u.to(device).float()
        s = data.s.to(device).float()
        edge_index = data.edge_index.to(device)

        pz = self.get_cond_prior(u)
        qz_params = self.get_z(x, u, s, edge_index)
        x_mu = self.get_x(x, s, edge_index, qz_params)

        return ConfigDict({
            'qz_params':    qz_params,
            'pz':           pz,
            'px':           x_mu
        })
    
    def evaluate(self, adata, k=30, n_subgraphs=8, device=torch.device('cpu')):
        """
        Predict latent representation & reconstructions 
        on mini-batched subgraphs
        """
        self.to(device)
        self.device = device
        self.eval()

        position_map = {
            tuple(pos): i
            for i, pos in enumerate(
                adata.obs[['x_centroid', 'y_centroid']].values.astype(np.float32)
            )
        }
        graph_data = XeniumGraphDataset(
            k=k, n_subgraphs=n_subgraphs
        ).load_graphs([adata])

        dataloader = DataLoader(graph_data, shuffle=False)
        qz = np.zeros((adata.shape[0], self.configs.c_latent), dtype=np.float32)
        pz = np.zeros_like(qz)
        px = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qz = res.qz_params[0].detach().cpu().numpy() \
                       if isinstance(res.qz_params, tuple) \
                       else res.qz_params[0].detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()

            for pos, qz_i, pz_i, px_i in zip(data.pos, batch_qz, batch_pz, batch_px):
                idx = position_map[tuple(pos.detach().cpu().numpy().astype(np.float32))]
                qz[idx], pz[idx], px[idx] = qz_i, pz_i, px_i
        
        return ConfigDict({
            'qz':   qz,
            'pz':   pz,
            'px':   px
        })
        
    def _PD_approx(self, cov, UPLO='L'):
        eigvals, Q = torch.linalg.eigh(cov @ cov.T) if UPLO == 'L' else \
                     torch.linalg.eigh(cov)
        Qt = Q.T
        Lambda = torch.diag(torch.tensor([torch.max(v, EPS) for v in eigvals]))
        return Q @ Lambda @ Qt
    

class ConditionalPrior(nn.Module):
    def __init__(self, configs):
        super(ConditionalPrior, self).__init__()
        activation = configs.act
        c_hidden = min(configs.c_aux, configs.c_hidden)
        self.layer = nn.Sequential(
            nn.Linear(configs.c_aux, c_hidden),
            activation,
            nn.Linear(c_hidden, configs.c_latent),
        )

    def forward(self, x):
        return self.layer(x)


class Encoder(nn.Module):
    # DEBUG the wrong factorization but force learning q(z | u):
    # q(z | x, u) ~approx q(z | x) * q(z | u)
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.prior_dist = configs.prior
        self.embed_option = configs.embed_option
        activation = configs.act

        c_obs = configs.c_in + configs.c_aux
        self.obs_to_hid = nn.Sequential(
            nn.Linear(c_obs, configs.c_hidden),
            activation,
            nn.Linear(configs.c_hidden, configs.c_hidden)
        )

        self.hid_to_zmu = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zlogvar = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zconc = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        
    def forward(self, x, u, s, edge_index):
        if self.embed_option == 'cat':
            obs = torch.cat([x, u], dim=-1)
            h = self.obs_to_hid(obs)

        elif self.embed_option == 'attn':
            hx = self.x_to_hid(x, edge_index)
            hu = self.u_to_hid(u, edge_index)
            h = F.scaled_dot_product_attention(hu, hx, hx)
        else:
            raise NotImplementedError(
                'Integration option {} not implemented in Encoder'.format(
                    self.integrate_option
                )
            )

        if self.prior_dist == 'normal':
            z_mu = self.hid_to_zmu(h, edge_index)
            z_logvar = self.hid_to_zlogvar(h, edge_index)
            return z_mu, z_logvar
        else:
            z_conc = self.hid_to_zconc(h, edge_index)
            return z_conc


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()        
        activation = configs.act
        c_hid_covariate = configs.c_hidden + configs.c_covariate  # dim. for f(z, s) 

        self.z_to_hid = Sequential('z, edge_index', [
            (SGConv(configs.c_latent, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
            activation,
            nn.Dropout(p=configs.dropout)
        ])

        self.hid_to_xmu = nn.Sequential(
            nn.Linear(c_hid_covariate, configs.c_in),
            activation,
            nn.Dropout(p=configs.dropout),
            nn.Linear(configs.c_in, configs.c_in),
            nn.Softmax(-1)
        )

    def forward(self, z, s, edge_index):
        h = self.z_to_hid(z, edge_index)
        hs = torch.cat([h, s], dim=-1)
        mu = self.hid_to_xmu(hs) + EPS
        return mu
