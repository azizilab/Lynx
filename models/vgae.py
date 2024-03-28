import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch.distributions import Bernoulli, Normal
from torch.distributions import kl_divergence as kl
from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder
from torch_geometric.utils import to_dense_adj


MAX_LOGSTD = 10


class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        self.qz_mu = GCNConv(configs.c_in, configs.c_hidden)
        self.qz_logstd = GCNConv(configs.c_in, configs.c_hidden)

        self.qu_mu = GCNConv(configs.c_hidden, configs.c_latent)
        self.qu_logstd = GCNConv(configs.c_hidden, configs.c_latent)
        self.eps = 1e-10

    def forward(self, x, edge_index, edge_weight):
        z_mu = F.gelu(self.qz_mu(
            x, 
            edge_index=edge_index, edge_weight=edge_weight
        ))
        z_logstd = F.softplus(self.qz_logstd(
            x,
            edge_index=edge_index, edge_weight=edge_weight
        )) + self.eps
        z = self.reparametrize(z_mu, z_logstd)

        u_mu = self.qu_mu(
            z, 
            edge_index=edge_index, edge_weight=edge_weight
        )
        u_mu = torch.tanh(u_mu)
        u_logstd = self.qu_logstd(
            z,
            edge_index=edge_index, edge_weight=edge_weight
        )
        u = self.reparametrize(u_mu, u_logstd)

        latent = ConfigDict()

        latent.z = z
        latent.z_mu = z_mu
        latent.z_logstd = z_logstd
        latent.u_mu = u_mu
        latent.u_logstd = u_logstd
        latent.u = u

        return latent
    
    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    

class MultiLevelDecoder(nn.Module):
    def __init__(self, configs):
        super(MultiLevelDecoder, self).__init__()
        self.pu_std = torch.tensor(configs.pu_std)
        self._pz_std = nn.Parameter(configs.pz_std * torch.ones(configs.c_hidden))
        self.pz_from_u = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            nn.GELU()
        )
        self.pa_from_z = InnerProductDecoder()
        self.eps = 1e-10
    
    def forward(self, latent):
        pz_mu = self.pz_from_u(latent.u)

        A_hat = F.relu(latent.z @ latent.z.t())
        A_hat = A_hat / A_hat.max()

        reconst = ConfigDict()
        reconst.pz_mu = pz_mu
        reconst.A_hat = A_hat

        return reconst

    @property
    def pz_std(self):
        return F.softplus(self._pz_std) + self.eps


class SparseVGAE(VGAE):
    def __init__(self, configs):
        super(SparseVGAE, self).__init__(
            encoder=GCNEncoder(configs),
            decoder=MultiLevelDecoder(configs)
        )
        self.beta = configs.beta
        self.__mu__ = None
        self.__logstd__ = None

    def loss(self, latent, pu_mu,
             edge_index, edge_weight):
        n_nodes = edge_index.shape[1]

        reconst = self.decoder(latent)
        recon_loss = self.get_recon_loss(reconst.A_hat, edge_index, edge_weight)
        reg_loss = self.get_smoothness_loss(latent.z, edge_index, edge_weight)
        sign_loss = self.get_sign_loss(latent.u_mu, pu_mu)

        # self.__mu__ = latent.u_mu
        # self.__logstd__ = latent.u_logstd
        # kl_u = self.kl_loss()  # KL-divergence for `u`

        kl_u = kl(
            Normal(latent.u_mu, torch.exp(latent.u_logstd)),
            Normal(pu_mu, self.decoder.pu_std)
        ).sum(dim=1).mean()
        
        kl_z = kl(
            Normal(latent.z_mu, torch.exp(latent.z_logstd)),
            Normal(reconst.pz_mu, self.decoder.pz_std)
        ).sum(dim=1).mean()
        kl_loss = kl_u + kl_z
        
        # loss = recon_loss + self.beta*reg_loss + self.beta*sign_loss + (1/n_nodes)*kl_loss
        loss = recon_loss + self.beta*(sign_loss + kl_loss + reg_loss) 
        return loss, recon_loss, reg_loss, kl_loss, sign_loss

    def get_recon_loss(self, A_hat, edge_index, edge_weight):
        # Compute BCE as the surrogate loss function for NLL
        A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze(0)
        recon_loss = torch.norm(A-A_hat, p=2)
        return recon_loss
    
    def get_smoothness_loss(self, z, edge_index, edge_weight):
        A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze(0)
        A_prime = A + torch.diag(torch.ones(A.shape[0]))
        D = torch.diag(torch.sum(A_prime, dim=-1))
        D_prime = torch.sqrt(torch.inverse(D))

        L = D - A
        L_prime = D_prime.t() @ L @ D_prime
        lap_loss = torch.trace(z.t() @ L_prime @ z)

        return lap_loss

    def get_sign_loss(self, qu_mu, pu_mu):
        u, v = qu_mu.squeeze(), pu_mu.squeeze()
        prod = u * v
        sign_loss = torch.sum(F.relu(-prod))
        return sign_loss