import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ml_collections import ConfigDict
from tqdm import trange
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl 
from torch_sparse import SparseTensor

EPS = 1e-15  # epsilon for positive constraint


class VAE(nn.Module):
    """
    Baseline VAE
    """
    def __init__(
        self, 
        configs,
        device=torch.device('cpu')
    ):
        super (VAE, self).__init__()
        self.configs = configs
        self.device = device

        # 1-layer
        self.x_to_umu = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_latent),
            nn.BatchNorm1d(configs.c_latent),
            nn.Softplus(),
            nn.Dropout(p=configs.dropout)
        )
        self.x_to_ulogstd = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_latent),
            nn.BatchNorm1d(configs.c_latent),
            nn.Dropout(p=configs.dropout)
        )
        self.u_to_xmu = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_in),
            nn.BatchNorm1d(configs.c_in),
            nn.ReLU()
        )
        self._px_scale = nn.Parameter(torch.rand(configs.c_in))

    def encoder(self, x):
        qu_mu = self.x_to_umu(x)
        qu_logstd = self.x_to_ulogstd(x)
        qu = self._reparametrize(qu_mu, torch.exp(qu_logstd))
        return ConfigDict({'qu_mu': qu_mu, 'qu_logstd': qu_logstd, 'qu': qu})
    
    def decoder(self, qu):
        self.px_mu = self.u_to_xmu(qu)
        self.px = self._reparametrize(self.px_mu, self.px_scale)
        return ConfigDict({'px_mu': self.px_mu, 'px': self.px})
    
    def loss(self, latent, recon, x):
        pu_mu = torch.zeros_like(latent.qu_mu)
        pu_std = torch.ones_like(latent.qu_logstd)

        # TODO: NegBinom parametrization for Xenium, Normal likelihood X fit
        nll = -Normal(recon.px_mu, self.px_scale).log_prob(x).sum(-1).mean()  
        kl_div = kl(
            Normal(latent.qu_mu, torch.exp(latent.qu_logstd)),
            Normal(pu_mu, pu_std)
        ).sum(-1).mean()
        return nll + self.configs.beta*kl_div, nll, kl_div

    @property
    def px_scale(self):
        return F.softplus(self._px_scale) + EPS
        
    def model_train(self, train_configs, dataloader):
        torch.manual_seed(42)
        self.to(self.device)
        self.train()

        losses = []
        nlls = []
        kls = []

        optimizer = optim.Adam(self.parameters(), lr=train_configs.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_configs.gamma)
        pbar = trange(train_configs.n_epochs, desc='Training', leave=True)

        for _ in enumerate(pbar):
            batch_losses = []
            batch_nlls = []
            batch_kls = []

            for x in dataloader:
                x = x.float().to(self.device)
                loss, nll, kl = self.run_one_epoch(optimizer, x)
                batch_losses.append(loss)
                batch_nlls.append(nll)
                batch_kls.append(kl)

            losses.append(np.mean(batch_losses))
            nlls.append(np.mean(batch_nlls))
            kls.append(np.mean(batch_kls))

            scheduler.step()

            pbar.set_postfix({'Training loss': '{:.3f}'.format(losses[-1]),
                              'NLL': '{:.3f}'.format(nlls[-1]),
                              'KL': '{:.3f}'.format(kls[-1])})
        
        pbar.close()
        return losses, nlls, kls
    
    def model_eval(self, feature_mat):
        x = torch.tensor(feature_mat)
        x = x.float().to(self.device)
        self.eval()
        with torch.no_grad():
            latent = self.encoder(x)
            recon = self.decoder(latent.qu)
        return latent, recon 
    
    def run_one_epoch(self, optimizer, x):
        optimizer.zero_grad()
        latent = self.encoder(x)
        recon = self.decoder(latent.qu)
        loss, nll, kl = self.loss(latent, recon, x)
        loss.backward()
        optimizer.step()
        return float(loss), float(nll), float(kl)
    
    def _get_orient_loss(self, q, p):
        return F.binary_cross_entropy_with_logits(q, p, reduction='sum')

    def _reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    

class GPCALayer(nn.Module):
    """
    Graph-regularized PCA w/ nonliear activation
    Code reference from:
    https://arxiv.org/pdf/2006.12294
    https://github.com/LingxiaoShawn/GPCANet
    """
    def __init__(self, c_in, c_out, alpha=1.0, 
                 niter=50, act=None, center=True,
                 init_weight=True, ortho_weight=False):
        super(GPCALayer, self).__init__()
        self.c_out = c_out
        self.alpha = alpha
        self.niter = niter
        self.center = center
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(1, c_out))
        self.init_weight = init_weight
        self.ortho_weight = ortho_weight
        
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU()
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
        if self.init_weight and self.ortho_weight:
            _, eig_vec = torch.linalg.eigh(x.t().mm(invphi_x))
            eig_vec = torch.real(eig_vec)
            self.weight.data = eig_vec[:, -self.c_out:]
            self.init_weight = False

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
