import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch.distributions import Normal, Beta, Dirichlet
from torch.distributions import kl_divergence as kl
from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder
from torch_geometric.utils import to_dense_adj

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.utils import binary_concrete


class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        # x -> v Beta parameters
        self.x_to_c1 = GCNConv(configs.c_in, configs.c_hidden)
        self.x_to_c0 = GCNConv(configs.c_in, configs.c_hidden)

        # x -> z Normal parameters
        self.x_to_zloc = GCNConv(configs.c_in, configs.c_hidden)
        # self.x_to_zlogscale = GCNConv(configs.c_in, configs.c_hidden)

        # z -> u Normal parameters
        self.z_to_uloc = GCNConv(configs.c_hidden, configs.c_latent)
        self.z_to_ulogscale = GCNConv(configs.c_hidden, configs.c_latent)
        self.eps = 1e-10

    def forward(self, x, edge_index, edge_weight):
        # Sample v, pi & b
        qc1 = F.softplus(self.x_to_c1(
                x, 
                edge_index=edge_index, 
                edge_weight=edge_weight
        )) + self.eps

        qc0 = F.softplus(self.x_to_c0(
            x,
            edge_index=edge_index, 
            edge_weight=edge_weight
        )) + self.eps

        qv = Beta(qc1, qc0).rsample()
        pi = torch.cumprod(qv, dim=1)
        qb = binary_concrete(pi)

        # Sample z:
        qz_loc = F.softplus(self.x_to_zloc(
            x,
            edge_index=edge_index,
            edge_weight=edge_weight
        ))

        # qz_logscale = F.softplus(self.x_to_zlogscale(
        #     x,
        #     edge_index=edge_index,
        #     edge_weight=edge_weight
        # ))

        qr = Dirichlet(qz_loc).rsample()
        qz = qb * qr

        qu_loc = self.z_to_uloc(
            qz, 
            edge_index=edge_index, 
            edge_weight=edge_weight
        )
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

        latent.qc1 = qc1
        latent.qc0 = qc0
        latent.qv = qv
        latent.log_pi = torch.cumsum(torch.log(qv+self.eps), dim=1) 
        latent.qb = qb

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
        # self._pz_scale = nn.Parameter(torch.ones(configs.c_hidden) * configs.pz_scale)
        self._px_scale = nn.Parameter(torch.ones(configs.c_in) * configs.px_scale)
        self.alpha = configs.alpha
        self.c_hidden = configs.c_hidden

        # weighted InnerProduct decoder
        # w = torch.ones(configs.c_hidden, configs.c_hidden) * 0.1
        # w = w + torch.diag(0.9 - torch.diag(w))
        # self._w = nn.Parameter(w)

        self.u_to_zloc = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            nn.Softplus()
        )

        self.z_to_xloc = nn.Sequential(
            nn.Linear(configs.c_hidden, configs.c_in),
            nn.ReLU()
        )

        self.eps = 1e-10
    
    def forward(self, latent):
        pv = Beta(self.alpha, 1.).sample((self.c_hidden,))
        # pi = torch.cumprod(pv, dim=0)
        log_pi = torch.cumsum(torch.log(pv + self.eps), dim=0)

        pz_loc = self.u_to_zloc(F.sigmoid(latent.qu)) + self.eps
        px_loc = self.z_to_xloc(latent.qz)
        A_hat = F.relu(latent.qz @ latent.qz.t())

        recon = ConfigDict()
        recon.pu_scale = self.pu_scale

        recon.pc1, recon.pc0 = 1., self.alpha
        recon.pv = pv
        recon.log_pi = log_pi

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


# TODO: try SBM w/ IBP prior on Zs
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
        reg_loss = self.get_smoothness_loss(latent.qz, A)
        # reg_loss = 0.
        orient_loss = self.get_orient_loss(latent.qu_loc, pu_loc)

        kl_u = kl(
            Normal(latent.qu_loc, torch.exp(latent.qu_logscale)),
            Normal(pu_loc, recon.pu_scale)
        ).sum(dim=1).mean()
        
        kl_v = kl(
            Beta(latent.qc1, latent.qc0),
            Beta(recon.pc1, recon.pc0)
        ).sum(dim=1).mean()

        kl_b = self._get_bern_kl(latent.log_pi, recon.log_pi, latent.qb).sum(dim=1).mean()

        kl_z = kl(
            Dirichlet(latent.qz_loc),
            Dirichlet(recon.pz_loc)
        ).mean()

        kl_loss = kl_u + kl_v + kl_b + kl_z  

        loss = recon_loss + self.beta*(kl_loss + reg_loss + orient_loss) 
        
        return loss, recon_loss, reg_loss, kl_loss, orient_loss
    
    def _get_bern_kl(self, log_qpi, log_ppi, b, temp=1):
        qpi = log_qpi - temp*b
        lprob_qpi = qpi + torch.log(torch.tensor(temp)) - 2.*F.softplus(qpi)
        ppi = log_ppi - temp*b
        lprob_ppi = ppi + torch.log(torch.tensor(temp)) - 2.*F.softplus(ppi)

        return lprob_ppi - lprob_qpi

    def get_recon_loss(self, recon, x, A):
        graph_loss = torch.norm(A-recon.A_hat, p=2)
        expr_loss = -Normal(recon.px_loc, recon.px_scale).log_prob(x).sum(-1).mean()
        return graph_loss + expr_loss
    
    def get_smoothness_loss(self, z, A):
        A_prime = A + torch.diag(torch.ones(A.shape[0])) 
        D = torch.diag(torch.sum(A_prime, dim=-1))
        L = D - A_prime
        lap_loss = torch.trace(z.t() @ L @ z)
        return lap_loss

    def get_orient_loss(self, qu_mu, pu_mu):
        u, v = qu_mu.squeeze(), pu_mu.squeeze()
        prod = u * v
        sign_loss = torch.sum(F.relu(-prod))
        return sign_loss
    