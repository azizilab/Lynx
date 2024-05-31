import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch.distributions import Normal, Beta
from torch.distributions import kl_divergence as kl
from torch_geometric.nn import VGAE, GCNConv, GATv2Conv, InnerProductDecoder
from torch_geometric.utils import negative_sampling

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.utils import binary_concrete

EPS = 1e-15  # epsilon for distribution's positive constraint


class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        self.configs = configs 

        self.x_to_c1 = GCNConv(configs.c_in, configs.c_hidden)
        self.x_to_c0 = GCNConv(configs.c_in, configs.c_hidden)

        self.x_to_zloc = GCNConv(configs.c_in, configs.c_hidden)
        self.x_to_zlogscale = GCNConv(configs.c_in, configs.c_hidden)
        
        self.z_to_t = GCNConv(configs.c_hidden, configs.c_latent)
        self.z_to_ulogscale = GCNConv(configs.c_hidden, configs.c_latent)

    def forward(self, x, edge_index, edge_weight):
        # q(pi | x, A) & q(b | pi)
        qc1 = F.softplus(self.x_to_c1(
            x, edge_index=edge_index, edge_weight=edge_weight
        )) + EPS

        qc0 = F.softplus(self.x_to_c0(
            x, edge_index=edge_index, edge_weight=edge_weight
        )) + EPS

        qv = Beta(qc1, qc0).rsample()
        log_pi = self._stick_break_logprob(qv)
        qb = binary_concrete(torch.exp(log_pi))

        # q(z | b, x, A)
        qz_loc = F.sigmoid(
            self.x_to_zloc(x, edge_index=edge_index, edge_weight=edge_weight
        )) + EPS
        qz_logscale = self.x_to_zlogscale(x, edge_index=edge_index, edge_weight=edge_weight)
        qz = self.reparametrize(qz_loc, qz_logscale)
        qz_trunc = qz * qb

        # q(t | z, A), q(u_σ | z, A) & q(u | t, u_σ)
        qt = self.z_to_t(qz, edge_index=edge_index, edge_weight=edge_weight)
        qt = (qt - qt.min()) / (qt.max() - qt.min())
        qu_logscale = self.z_to_ulogscale(qz, edge_index=edge_index, edge_weight=edge_weight)
        qu = qt * self.reparametrize(torch.tensor([1.]), qu_logscale) + \
             (1-qt) * self.reparametrize(torch.tensor([0.]), qu_logscale)
        
        return ConfigDict({
            'qc1':          qc1,
            'qc0':          qc0,
            'log_pi':       log_pi,
            'qb':           qb,

            'qz_loc':       qz_loc,
            'qz_logscale':  qz_logscale,
            'qz':           qz,
            'qz_trunc':     qz_trunc,

            'qt':           qt,
            'qu_logscale':  qu_logscale,
            'qu':           qu
        })
    
    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(logstd) * torch.exp(logstd)
            
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
        self.pu_scale = torch.tensor(configs.pu_scale)
        self._px_scale = nn.Parameter(torch.rand(configs.c_in) * configs.px_scale)

        self.u_to_zloc = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            nn.Softplus()
        )
        self.u_to_zlogscale = nn.Linear(configs.c_latent, configs.c_hidden)
        
        self.z_to_xloc = GATv2Conv(
            configs.c_hidden, configs.c_in, 
            heads=1, concat=False, share_weights=False,
            dropout=configs.drop_rate
        )
    
    def forward(self, latent, edge_index, edge_weight):
        n_nodes = latent.qz.shape[0]
        pv = Beta(1., self.configs.alpha).sample((self.configs.c_hidden,))
        log_pi = self._stick_break_logprob(pv.expand(n_nodes, -1))

        pz_loc = self.u_to_zloc(latent.qu) + EPS
        pz_logscale = self.u_to_zlogscale(latent.qu)

        px_loc, attn_zx = self.z_to_xloc(latent.qz, edge_index, edge_weight, return_attention_weights=True)
        px_loc = F.relu(px_loc)
        A_hat = F.sigmoid(latent.qz_trunc) @ F.sigmoid(latent.qz_trunc.t())

        return ConfigDict({
            'pu_scale':     self.pu_scale,
            'pc1':          1.,
            'pc0':          self.configs.alpha,
            'log_pi':       log_pi,
            
            'pz_loc':       pz_loc,
            'pz_logscale':  pz_logscale,

            'A_hat':        A_hat,
            'px_loc':       px_loc,
            'px_scale':     self.px_scale,
            'attn_zx':      attn_zx            
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
        self.ipd = InnerProductDecoder()

    def loss(self, latent, pt,
             x, edge_index, edge_weight):
        recon = self.decoder(latent, edge_index, edge_weight)
        recon_loss = self.get_recon_loss(latent.qz, recon, x, edge_index)
        reg_loss = self.get_ortho_loss(latent.qz)
        orient_loss = self.get_orient_loss(latent.qt, pt)

        # Optimize for q(u_σ | z, A) only
        kl_u = kl(
            Normal(0, torch.exp(latent.qu_logscale)+EPS),
            Normal(0, recon.pu_scale)
        ).sum(dim=1).mean()
         
        kl_v = kl(
            Beta(latent.qc1, latent.qc0),
            Beta(recon.pc1, recon.pc0)
        ).sum(dim=1).mean()

        kl_b = self._get_bern_kl(latent.log_pi, recon.log_pi, latent.qb).sum(dim=1).mean()

        kl_z = kl(
            Normal(latent.qz_loc, torch.exp(latent.qz_logscale)+EPS),
            Normal(recon.pz_loc, torch.exp(recon.pz_logscale)+EPS)
        ).sum(dim=1).mean()

        kl_loss = kl_u + kl_v + kl_b + kl_z
        loss = recon_loss + self.beta*(kl_loss + reg_loss + orient_loss)
        
        return loss, recon_loss, reg_loss, kl_loss, orient_loss
    
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

    def get_recon_loss(self, qz, recon, x, edge_index):
        """Feature matrix reconstruction loss & L1 regularization of graph loss"""
        neg_edge_index = negative_sampling(edge_index, force_undirected=True)
        graph_loss = (-torch.log(self.ipd(qz, edge_index, sigmoid=True)+EPS).mean()) + \
                     (-torch.log(1 - self.ipd(qz, neg_edge_index, sigmoid=True)+EPS).mean())
        expr_loss = -Normal(recon.px_loc, recon.px_scale).log_prob(x).sum(-1).mean()

        # DEBUG: why negative reconstruction??
        return expr_loss + graph_loss

    def get_ortho_loss(self, z):
        z_norm = F.normalize(z, dim=0)
        ztz = z_norm.t() @ z_norm
        I = torch.eye(ztz.size(0), device=z.device)
        return F.mse_loss(ztz, I)

    def get_orient_loss(self, qt, pt, origin=0.5):
        u, v = qt.squeeze() - origin, pt.squeeze() - origin
        prod = u * v
        return torch.sum(F.relu(-prod))
