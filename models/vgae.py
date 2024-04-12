import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch.distributions import Normal, Uniform, Gamma, Distribution
from torch.distributions import kl_divergence as kl
from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder
from torch_geometric.utils import to_dense_adj


class Weibull(Distribution):
    def __init__(self, scale, concentration, validate_args=None):
        self.scale = scale
        self.concentration = concentration
        self.uniform = Uniform(0, 1)
        super(Weibull, self).__init__(validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniform_sample = self.uniform.sample(shape)
        weibull_sample = self.scale * (-torch.log(1 - uniform_sample)).pow(1 / self.concentration)
        return weibull_sample

    def log_prob(self, value):
        log_scale = torch.log(self.scale)
        log_prob = (self.concentration - 1) * torch.log(value) - (value / self.scale).pow(self.concentration) + log_scale
        return log_prob

    def entropy(self):
        gamma_const = 0.57721566490153286060  # Euler-Mascheroni constant
        entropy = 1 + torch.log(self.scale) - torch.log(self.concentration) + gamma_const * (1 - 1 / self.concentration)
        return entropy

    def expand(self, batch_shape):
        new = self._get_checked_instance(Weibull, batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(Weibull, new).__init__(validate_args=False)
        new._validate_args = self._validate_args
        return new


# TODO: debug Gamma distribution for z
class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        self.x_to_zloc = GCNConv(configs.c_in, configs.c_hidden)
        self.x_to_zlogscale = GCNConv(configs.c_in, configs.c_hidden)

        self.z_to_uloc = GCNConv(configs.c_hidden, configs.c_latent)
        self.z_to_ulogscale = GCNConv(configs.c_hidden, configs.c_latent)
        self.eps = 1e-10

    def forward(self, x, edge_index, edge_weight):
        qz_loc = F.softmax(self.x_to_zloc(
                x, 
                edge_index=edge_index, 
                # edge_weight=edge_weight
        ), dim=-1) + self.eps

        qz_logscale = F.softplus(self.x_to_zlogscale(
            x,
            edge_index=edge_index, 
            # edge_weight=edge_weight
        )) + self.eps
        
        # qz = self.reparametrize(qz_loc, qz_logscale)
        # qz = Weibull(qz_loc, qz_logscale).rsample()
        qz = Gamma(qz_loc, torch.exp(qz_logscale)).rsample()

        qu_loc = self.z_to_uloc(
            qz, 
            edge_index=edge_index, 
            # edge_weight=edge_weight
        )

        qu_loc = torch.tanh(qu_loc)

        qu_logscale = F.softplus(self.z_to_ulogscale(
            qz,
            edge_index=edge_index, 
            # edge_weight=edge_weight
        )) + self.eps
        
        qu = self.reparametrize(qu_loc, qu_logscale)

        latent = ConfigDict()
        latent.qu_loc = qu_loc
        latent.qu_logscale = qu_logscale
        latent.qu = qu

        latent.qz_loc = qz_loc
        latent.qz_logscale = qz_logscale
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
        self.pz_scale = torch.ones(configs.c_hidden) * configs.pz_scale
        self._px_scale = nn.Parameter(torch.ones(configs.c_in) * configs.px_scale)

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
        pz_loc = self.u_to_zloc(latent.qu) + self.eps
        px_loc = self.z_to_xloc(latent.qz)

        A_hat = F.relu(latent.qz @ latent.qz.t())

        reconst = ConfigDict()
        reconst.pu_scale = self.pu_scale

        reconst.pz_loc = pz_loc
        reconst.pz_scale = self.pz_scale
        
        reconst.A_hat = A_hat
        reconst.px_loc = px_loc
        reconst.px_scale = self.px_scale
        
        return reconst
    
    @property
    def px_scale(self):
        return F.softplus(self._px_scale) + self.eps
    

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
        
        A = to_dense_adj(edge_index=edge_index).squeeze(0)
        A += torch.diag(torch.ones(A.shape[0]))

        recon = self.decoder(latent)
        recon_loss = self.get_recon_loss(recon, x, A)
        reg_loss = self.get_smoothness_loss(latent.qz, A)
        orient_loss = self.get_orient_loss(latent.qu_loc, pu_loc)

        kl_u = kl(
            Normal(latent.qu_loc, torch.exp(latent.qu_logscale)),
            Normal(pu_loc, recon.pu_scale)
        ).sum(dim=1).mean()
        
        # kl_z = kl(
        #     Normal(latent.qz_loc, torch.exp(latent.qz_logscale)),
        #     Normal(recon.pz_loc, recon.pz_scale)
        # ).sum(dim=1).mean()

        kl_z = kl(
            Gamma(latent.qz_loc, torch.exp(latent.qz_logscale)),
            Gamma(recon.pz_loc, 1./recon.pz_scale)
        ).sum(dim=1).mean()

        kl_loss = kl_u + kl_z
        
        loss = recon_loss + self.beta*(kl_loss + reg_loss + orient_loss) 
        return loss, recon_loss, reg_loss, kl_loss, orient_loss

    def get_recon_loss(self, recon, x, A):
        graph_loss = torch.norm(A-recon.A_hat, p=2)
        feature_loss = -Normal(recon.px_loc, recon.px_scale).log_prob(x).sum(-1).mean()
        return feature_loss + graph_loss
    
    def get_smoothness_loss(self, z, A):
        D = torch.diag(torch.sum(A, dim=-1))
        D_prime = torch.sqrt(torch.inverse(D))

        L = D - A
        L_prime = D_prime.t() @ L @ D_prime
        lap_loss = torch.trace(z.t() @ L_prime @ z)

        return lap_loss

    def get_orient_loss(self, qu_mu, pu_mu):
        u, v = qu_mu.squeeze(), pu_mu.squeeze()
        prod = u * v
        sign_loss = torch.sum(F.relu(-prod))
        return sign_loss
    
    def _compute_gamma_kl(self, lam, k, a, b):
        euler_const = 0.57721566490153286060        
        kl = - ( a*torch.log1p(lam) - euler_const*a/k - torch.log1p(k) - 
                 b*lam*torch.exp(torch.lgamma(1+1/k)) + euler_const + 1 + 
                 a*torch.log1p(b) - torch.lgamma(a) )
        return kl.sum(1).mean()

    # def get_orient_loss(self, qu_mu, pu_mu):
    #     cosine_dist = 1 - F.cosine_similarity(qu_mu.squeeze(), 
    #                                           pu_mu.squeeze(), 
    #                                           dim=0)
    #     return cosine_dist
    