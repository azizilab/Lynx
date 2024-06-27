import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch.distributions import Normal, Beta
from torchrl.modules import TruncatedNormal
from torch.distributions import kl_divergence as kl
from torch_geometric.nn import VGAE, GCNConv, Sequential
from torch_geometric.utils import negative_sampling 

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.utils import binary_concrete

EPS = 1e-15  # epsilon for positive constraint


class SparseVGAE(VGAE):
    """
    Hierarchical VGAE with stochastic variables
    """
    def __init__(
        self, 
        encoder,
        decoder, 
        beta=1.0
    ):
        super(SparseVGAE, self).__init__(
            encoder=encoder,
            decoder=decoder
        )
        self.beta = beta
        self.l1_weight = 1e-3

    def loss(self, latent, recon, pu,
             x, edge_index):

        kl_v = kl(
            Beta(latent.qc1, latent.qc0),
            Beta(recon.pc1, recon.pc0)
        ).sum(dim=1).mean()

        kl_b = self._get_bern_kl(latent.log_pi, recon.log_pi, latent.qb).sum(dim=1).mean()

        kl_z = kl(
            Normal(latent.qz_loc, torch.exp(latent.qz_logscale)),
            Normal(recon.pz_loc, torch.exp(recon.pz_logscale))
        ).sum(dim=1).mean()
        
        recon_loss = self._get_recon_loss(latent.qz, recon, x, edge_index)
        kl_loss = kl_v + kl_b + kl_z
        reg_loss = self._get_l1_regularization()
        ortho_loss = self._get_ortho_loss(latent.qz)
        orient_loss = self._get_orient_loss(latent.qu, pu)

        loss = recon_loss + reg_loss + self.beta*(kl_loss + ortho_loss + orient_loss)

        return (loss, recon_loss, reg_loss, ortho_loss, kl_loss, orient_loss)
    
    def _get_bern_kl(self, log_qpi, log_ppi, b, temp=1.):
        # prior p(\pi)
        logit_ppi = torch.logit(torch.exp(log_ppi) + EPS)
        ppi = logit_ppi - temp*b
        log_prob_ppi = ppi + torch.log(torch.tensor(temp)) - 2.*F.softplus(ppi)
        
        # posterior q(\pi)
        logit_qpi = torch.logit(torch.exp(log_qpi) + EPS)
        qpi = logit_qpi - temp*b
        log_prob_qpi = qpi + torch.log(torch.tensor(temp)) - 2.*F.softplus(qpi)

        return log_prob_qpi - log_prob_ppi

    def _get_recon_loss(self, qz, recon, x, edge_index):
        """Feature matrix reconstruction loss"""
        # neg_edge_index = negative_sampling(edge_index, force_undirected=True)
        # graph_loss = (-torch.log(self.ipd(qz, edge_index, sigmoid=True)+EPS).mean()) + \
        #              (-torch.log(1 - self.ipd(qz, neg_edge_index, sigmoid=True)+EPS).mean())
        expr_loss = -Normal(recon.px_loc, recon.px_scale).log_prob(x).sum(-1).mean()
        return expr_loss

    def _get_ortho_loss(self, z):
        z_norm = F.normalize(z, dim=0)
        ztz = z_norm.t() @ z_norm
        I = torch.eye(ztz.size(0), device=z.device)
        return F.mse_loss(ztz, I)

    def _get_orient_loss(self, q, p):
        return F.binary_cross_entropy(q, p, reduction='sum')

    def _get_l1_regularization(self):
        return self.l1_weight * torch.tensor([param.view(-1).abs().sum() for param in self.parameters()]).sum()


class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        self.x_to_c1 = Sequential('x, edge_index, edge_weight', [
            (GCNConv(configs.c_in, configs.c_hidden), 'x, edge_index, edge_weight -> qc1'),
            nn.Softplus(),
        ])
        self.x_to_c0 = Sequential('x, edge_index, edge_weight', [
            (GCNConv(configs.c_in, configs.c_hidden), 'x, edge_index, edge_weight -> qc0'),
            nn.Softplus(),
        ])

        self.x_to_zloc = Sequential('x, edge_index, edge_weight', [
            (GCNConv(configs.c_in, configs.c_hidden), 'x, edge_index, edge_weight -> qz_loc'),
            nn.Softplus()
        ])
        self.x_to_zlogscale = GCNConv(configs.c_in, configs.c_hidden)

        self.z_to_uloc = Sequential('z, edge_index, edge_weight', [
            (GCNConv(configs.c_hidden, configs.c_latent), 'z, edge_index, edge_weight -> qu_loc'),
            nn.Sigmoid()
        ])
        
    def forward(self, x, edge_index, edge_weight):
        # q(\pi | x, A); q(b | \pi)
        qc1 = self.x_to_c1(x, edge_index, edge_weight) + EPS
        qc0 = self.x_to_c0(x, edge_index, edge_weight) + EPS
        qv = Beta(qc1, qc0).rsample()
        log_pi = self._stick_break_logprob(qv)
        qb = binary_concrete(torch.exp(log_pi))

        # q(z | x, A)
        qz_loc = self.x_to_zloc(x, edge_index, edge_weight)
        qz_logscale = self.x_to_zlogscale(x, edge_index, edge_weight)
        qz = self.reparametrize(qz_loc, qz_logscale) * qb

        # q(u | z, A)
        qu = self.z_to_uloc(qz, edge_index, edge_weight)

        return ConfigDict({
            'qc1': qc1, 'qc0': qc0,  'log_pi': log_pi,  'qb': qb,
            'qz_loc': qz_loc,  'qz_logscale':  qz_logscale,  'qz': qz,
            # 'qu_loc': qu_loc, 'qu_logscale': qu_logscale,  'qu': qu
            'qu': qu
        })
    
    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
            
    def _stick_break_logprob(self, v):
        log_1mv = torch.log(1 - v[:, :-1] + EPS)
        logv = torch.log(v + EPS)
        log_pi0 = F.pad(torch.cumsum(log_1mv, dim=1), (1, 0), value=0)
        log_pi = logv + log_pi0
        return log_pi
        

class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()
        self.configs = configs

        # Linear decoder
        self.u_to_zloc = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            nn.Dropout(p=configs.dropout)
        )
        self.u_to_zlogscale = nn.Linear(configs.c_latent, configs.c_hidden)

        self.z_to_xloc = nn.Sequential(
            nn.Linear(configs.c_hidden, configs.c_in),
            nn.ReLU(),
            nn.Dropout(p=configs.dropout)
        )

        # Graph decoder
        # self.u_to_zloc = Sequential('qu, edge_index', [
        #     (GCNConv(configs.c_latent, configs.c_hidden), 'qu, edge_index -> pz_loc'),
        #     nn.Dropout(p=configs.dropout),
        #     nn.ReLU()
        # ])
        # self.u_to_zlogscale = GCNConv(configs.c_latent, configs.c_hidden)

        # self.z_to_xloc = Sequential('qz, edge_index', [
        #     (GCNConv(configs.c_hidden, configs.c_in), 'qz, edge_index -> px_loc'),
        #     nn.Dropout(p=configs.dropout),
        #     nn.ReLU()
        # ])
        
        self._px_scale = nn.Parameter(torch.ones(configs.c_in) * configs.px_scale)

    def forward(self, latent, edge_index):
        n_nodes = latent.qz.shape[0]
        pv = Beta(1., self.configs.c0).sample((self.configs.c_hidden,))
        log_pi = self._stick_break_logprob(pv).expand(n_nodes, -1)

        pz_loc = self.u_to_zloc(latent.qu)
        pz_logscale = self.u_to_zlogscale(latent.qu)

        px_loc = self.z_to_xloc(latent.qz)

        return ConfigDict({
            'pc1': 1., 'pc0': self.configs.c0,  'pv': pv,  'log_pi': log_pi,
            'pz_loc': pz_loc,  'pz_logscale': pz_logscale,
            'px_loc': px_loc,  'px_scale': self.px_scale
        })
    
    def _stick_break_logprob(self, v):
        logv = torch.log(v + EPS)
        log_1mv = torch.log(1 - v[:-1] + EPS)
        log_pi0 = logv[1:] + torch.cumsum(log_1mv, dim=0)
        log_pi = torch.cat([logv[:1], log_pi0])
        return log_pi
    
    @property
    def px_scale(self):
        return F.softplus(self._px_scale) + EPS
    