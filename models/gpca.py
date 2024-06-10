import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch.distributions import Beta
from torchrl.modules import TruncatedNormal
from torch_sparse import SparseTensor
from torch_sparse import spmm

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.utils import binary_concrete

EPS = 1e-15  # epsilon for positive constraint


class GPCALayer(nn.Module):
    """
    Graph-regularized PCA w/ nonliear activation
    Code reference from:
    https://arxiv.org/pdf/2006.12294
    https://github.com/LingxiaoShawn/GPCANet
    """
    def __init__(self, c_in, c_out, alpha=1.0, 
                 niter=50, act=None, center=False):
        super(GPCALayer, self).__init__()
        self.c_out = c_out
        self.alpha = alpha
        self.niter = niter
        self.center = center
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(1, c_out))
        self.init_weight = True
        
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'softplus':
            self.act = nn.Softplus()
        else:
            self.act = nn.Identity()

        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index):
        n = x.shape[0]
        A = self._get_sparse_adj(edge_index, n)
        if self.center:
            x = x - x.mean(dim=0)

        # Compute F = inv(\psi) * x
        invphi_x = self._approx_f(A, x)

        # Compute orthonormal W
        # if self.init_weight:
        #     _, eig_vec = torch.linalg.eigh(x.t().mm(invphi_x))
        #     eig_vec = torch.real(eig_vec)
        #     self.weight.data = eig_vec[:, -self.c_out:]
        #     self.init_weight = False

        # Non-linear activation
        out = self.act(invphi_x.matmul(self.weight) + self.bias)
        return out

    def freeze(self):
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def _get_sparse_adj(self, edge_index, n):
        """Get sym. normalized adj (sparse format)"""
        row, col = edge_index
        A = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
        A = A.set_diag()
        D = A.sum(dim=1).to(torch.float)
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        return D_inv_sqrt.view(-1, 1) * D_inv_sqrt.view(-1, 1) * A
        
    def _approx_f(self, A, x):
        """Iterative approx. of F ~ inv(I + \alpha*L) * x"""
        invphi_x = x
        for _ in range(self.niter):
            AF = A.matmul(invphi_x)
            invphi_x = self.alpha/(1+self.alpha)*AF + 1/(1+self.alpha)*x
        return invphi_x
            

class GPCAEncoder(nn.Module):
    def __init__(self, configs):
        super(GPCAEncoder, self).__init__()
        self.x_to_c1 = GPCALayer(configs.c_in, configs.c_hidden, configs.alpha, act='softplus')
        self.x_to_c0 = GPCALayer(configs.c_in, configs.c_hidden, configs.alpha, act='softplus')
        
        self.x_to_zloc = GPCALayer(configs.c_in, configs.c_hidden, configs.alpha, act='sigmoid')
        self.x_to_zlogscale = GPCALayer(configs.c_in, configs.c_hidden, configs.alpha)
        
        self.z_to_uloc = GPCALayer(configs.c_hidden, configs.c_latent, configs.alpha)
        
    def forward(self, x, edge_index, edge_weight=None):
        # q(\pi | x, A); q(b | \pi)
        qc1 = self.x_to_c1(x, edge_index) + EPS
        qc0 = self.x_to_c0(x, edge_index) + EPS
        qv = Beta(qc1, qc0).rsample()
        log_pi = self._stick_break_logprob(qv)
        qb = binary_concrete(torch.exp(log_pi))

        # q(z | b, x, A)
        qz_loc = self.x_to_zloc(x, edge_index)
        qz_logscale = self.x_to_zlogscale(x, edge_index)
        qz = TruncatedNormal(qz_loc, torch.exp(qz_logscale)).rsample() * qb

        # q(u | z, A)
        qu = self.z_to_uloc(qz, edge_index)

        return ConfigDict({
            'qc1': qc1,  'qc0': qc0,  'log_pi': log_pi,  'qb': qb,
            'qz_loc': qz_loc, 'qz_logscale': qz_logscale, 'qz': qz,
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

