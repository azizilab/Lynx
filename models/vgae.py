import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch.distributions import Normal, Dirichlet, Distribution
from torch.distributions import kl_divergence as kl
from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder
from torch_geometric.utils import to_dense_adj


class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        self.x_to_zloc = GCNConv(configs.c_in, configs.c_hidden)
        self.x_to_zlogscale = GCNConv(configs.c_in, configs.c_hidden)

        self.z_to_uloc = GCNConv(configs.c_hidden, configs.c_latent)
        self.z_to_ulogscale = GCNConv(configs.c_hidden, configs.c_latent)
        self.eps = 1e-10

    def forward(self, x, edge_index, edge_weight):
        qz_loc = F.softplus(self.x_to_zloc(
            x, 
            edge_index=edge_index, 
            edge_weight=edge_weight
        )) + self.eps

        # qz_logscale = F.softplus(self.x_to_zlogscale(
        #     x,
        #     edge_index=edge_index, 
        #     edge_weight=edge_weight
        # )) + self.eps
        
        qz = Dirichlet(qz_loc).rsample()

        qu_loc = self.z_to_uloc(
            qz, 
            edge_index=edge_index, 
            edge_weight=edge_weight
        )
        qu_loc = torch.tanh(qu_loc)

        qu_logscale = F.softplus(self.z_to_ulogscale(
            qz,
            edge_index=edge_index, 
            edge_weight=edge_weight
        )) + self.eps
        
        qu = self.reparametrize(qu_loc, qu_logscale)

        latent = ConfigDict()
        latent.qu_loc = qu_loc
        latent.qu_logscale = qu_logscale
        latent.qu = qu

        latent.qz_loc = qz_loc
        # latent.qz_logscale = qz_logscale
        latent.qz = qz

        return latent
    
    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()
        self.pu_scale = torch.tensor(configs.pu_scale)
        self._px_scale = nn.Parameter(torch.ones(configs.c_in) * configs.px_scale)

        self.u_to_zloc = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            nn.Softplus()
        )

        self.z_to_xloc = nn.Sequential(
            nn.Linear(configs.c_hidden, configs.c_in),
            nn.Softplus()
        )
        
        self.eps = 1e-10
    
    def forward(self, latent):
        pz_loc = self.u_to_zloc(latent.qu) + self.eps
        px_loc = self.z_to_xloc(latent.qz)

        A_hat = F.relu(latent.qz @ latent.qz.t())

        recon = ConfigDict()
        recon.pu_scale = self.pu_scale

        recon.pz_loc = pz_loc
        # recon.pz_scale = self.pz_scale
        
        recon.A_hat = A_hat        
        recon.px_loc = px_loc
        recon.px_scale = self.px_scale
        return recon
    
    @property
    def pz_scale(self):
        return F.softplus(self._pz_scale) + self.eps
    
    @property
    def px_scale(self):
        return F.softplus(self._px_scale) + self.eps
    
    @property
    def w(self):
        return F.relu(self._w)


class SparseVGAE(VGAE):
    """
    Hierarchical VGAE with stochastic variables
    """
    def __init__(self, configs):
        super(SparseVGAE, self).__init__(
            encoder=GCNEncoder(configs),
            decoder=Decoder(configs)
        )
        self.beta = configs.beta

    def loss(self, latent, pu_loc,
             x, edge_index, edge_weight):
        
        A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze(0)

        recon = self.decoder(latent)
        recon_loss = self.get_recon_loss(recon, x, A)
        reg_loss = self.get_smoothness_loss(latent.qu, A)
        orient_loss = self.get_orient_loss(latent.qu_loc, pu_loc)

        kl_u = kl(
            Normal(latent.qu_loc, torch.exp(latent.qu_logscale)),
            Normal(pu_loc, recon.pu_scale)
        ).sum(dim=1).mean()
        
        kl_z = kl(
            Dirichlet(latent.qz_loc),
            Dirichlet(recon.pz_loc)
        ).mean()

        kl_loss = kl_u + kl_z
        
        loss = recon_loss + self.beta*(kl_loss + reg_loss + orient_loss) 
        return loss, recon_loss, reg_loss, kl_loss, orient_loss

    def get_recon_loss(self, recon, x, A):
        graph_loss = torch.norm(A-recon.A_hat, p=2)
        expr_loss = -Normal(recon.px_loc, recon.px_scale).log_prob(x).sum(-1).mean()
        return graph_loss + expr_loss
    
    def get_smoothness_loss(self, u, A):
        A_prime = A + torch.diag(torch.ones(A.shape[0]))
        D = torch.diag(torch.sum(A_prime, dim=-1))
        L = D - A
        lap_loss = torch.trace(u.t() @ L @ u)
        return lap_loss

    def get_orient_loss(self, qu_mu, pu_mu):
        u, v = qu_mu.squeeze(), pu_mu.squeeze()
        prod = u * v
        sign_loss = torch.sum(F.relu(-prod))
        return sign_loss
    