import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ml_collections import ConfigDict
from tqdm import trange
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial
from torch.distributions import kl_divergence
from torch_sparse import SparseTensor


EPS = 1e-6
torch.manual_seed(0)


class VAE(nn.Module):
    """
    Baseline VAE for testing Xenium fitting w/ NB likelihood
    """
    def __init__(
        self,
        configs,
    ):
        super(VAE, self).__init__()
        self.configs = configs
        self.device = configs.device

        self.x_to_hid = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            nn.BatchNorm1d(configs.c_hidden),
            nn.ReLU()
        )

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

        self.z_to_hid = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            nn.BatchNorm1d(configs.c_hidden),
            nn.ReLU()
        )

        self.hid_to_mu = nn.Sequential(
            nn.Linear(configs.c_hidden, configs.c_in),
            nn.Softmax(dim=-1)
        )

        self._theta = nn.Parameter(torch.rand(configs.c_in))

    def encoder(self, x):
        x = torch.log1p(x)
        h = self.x_to_hid(x)
        
        qz_mu = self.hid_to_zmu(h)
        qz_logvar = self.hid_to_zlogvar(h)
        qz = Normal(qz_mu, torch.exp(qz_logvar/2)).rsample()

        return ConfigDict({
            'qz_mu':        qz_mu,
            'qz_logvar':    qz_logvar,
            'qz':           qz
        })
    
    def decoder(self, z, l):
        l = torch.tensor(l, dtype=torch.float, device=self.device)
        h = self.z_to_hid(z)
        x_mu = l * self.hid_to_mu(h)

        return ConfigDict({'px_mu': x_mu})

    def loss(self, x, latent, recon):
        qz_mu = latent.qz_mu
        qz_logvar = latent.qz_logvar

        x_mu = recon.px_mu
        pz_mu = torch.zeros_like(qz_mu)
        pz_std = torch.ones_like(qz_logvar)

        
        logits = torch.log(self.theta + EPS) - torch.log(x_mu + EPS)
        nll = -NegativeBinomial(
            mu=x_mu,
            theta=self.theta
        ).log_prob(x).sum(-1).mean()

        kl_div = kl_divergence(
            Normal(qz_mu, torch.exp(qz_logvar/2)),
            Normal(pz_mu, pz_std)
        ).sum(-1).mean()

        return nll + self.configs.beta*kl_div, nll, kl_div

    @property
    def theta(self):
        return F.softplus(self._theta) + EPS

    def model_train(self, train_configs, dataloader):
        self.to(self.device)
        self.train()

        losses = []
        nlls = []
        kls = []

        optimizer = optim.Adam(
            self.parameters(),
            lr=train_configs.lr,
            weight_decay=1e-3
        )
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
            pbar.set_postfix({
                'Training loss': '{:.3f}'.format(losses[-1]),
                'NLL': '{:.3f}'.format(nlls[-1]),
                'KL': '{:.3f}'.format(kls[-1])
            })
        
        pbar.close()
        return losses, nlls, kls
    
    def model_eval(self, expr, device):
        self.to(device)
        self.eval()

        x = torch.tensor(expr, dtype=torch.float, device=device)
        l = x.sum(-1, keepdim=True)
        with torch.no_grad():
            latent = self.encoder(x)
            recon = self.decoder(latent.qz, l)
        
        return latent, recon

    def run_one_epoch(self, optimizer, x):
        optimizer.zero_grad()
        l = x.sum(-1, keepdim=True)
        latent = self.encoder(x)
        recon = self.decoder(latent.qz, l)
        loss, nll, kl = self.loss(x, latent, recon)
        loss.backward()
        optimizer.step()
        return float(loss), float(nll), float(kl)
    

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
