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
from torch_geometric.utils import to_dense_adj

EPS = 1e-15  # epsilon for positive constraint


class VAE(nn.Module):
    """
    Baseline VAE with empirical t_i
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
        self._pu_scale = nn.Parameter(torch.ones(configs.c_latent) * configs.pu_scale)

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
        return ConfigDict({'px_mu': self.px_mu})
    
    def loss(self, latent, recon, x, pu):
        nll = -Normal(recon.px_mu, self.px_scale).log_prob(x).sum(-1).mean()  
        kl_div = kl(
            Normal(latent.qu_mu, torch.exp(latent.qu_logstd)),
            Normal(pu, self.pu_scale)
        ).sum(-1).mean()
        orient_loss = self._get_orient_loss(latent.qu_mu, pu)
        return nll + self.configs.beta*(kl_div+orient_loss), nll, kl_div, orient_loss
    
    @property
    def pu_scale(self):
        return F.softplus(self._pu_scale) + EPS

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
        orients = []

        optimizer = optim.Adam(self.parameters(), lr=train_configs.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_configs.gamma)
        pbar = trange(train_configs.n_epochs, desc='Training', leave=True)

        for _ in enumerate(pbar):
            batch_losses = []
            batch_nlls = []
            batch_kls = []
            batch_orients = []

            for (x, u_prior) in dataloader:
                x = x.float().to(self.device)
                u_prior = u_prior.float().to(self.device)

                loss, nll, kl, ol = self.run_one_epoch(optimizer, x, u_prior)
                batch_losses.append(loss)
                batch_nlls.append(nll)
                batch_kls.append(kl)
                batch_orients.append(ol)

            losses.append(np.mean(batch_losses))
            nlls.append(np.mean(batch_nlls))
            kls.append(np.mean(batch_kls))
            orients.append(np.mean(batch_orients))

            scheduler.step()

            pbar.set_postfix({'Training loss': '{:.3f}'.format(losses[-1]),
                              'NLL': '{:.3f}'.format(nlls[-1]),
                              'KL': '{:.3f}'.format(kls[-1]),
                              'Sign': '{:.3f}'.format(orients[-1])})
        
        pbar.close()
        return losses, nlls, kls, orients
    
    def model_eval(self, feature_mat):
        x = torch.tensor(feature_mat)
        x = x.float().to(self.device)
        self.eval()
        with torch.no_grad():
            latent = self.encoder(x)
            recon = self.decoder(latent.qu)
        return latent, recon 
    
    def run_one_epoch(self, optimizer, x, u_prior):
        optimizer.zero_grad()
        latent = self.encoder(x)
        recon = self.decoder(latent.qu)
        loss, nll, kl, orient_loss = self.loss(latent, recon, x, u_prior)
        loss.backward()
        optimizer.step()
        return float(loss), float(nll), float(kl), float(orient_loss)
    
    def _get_orient_loss(self, q, p, origin=0.5):
        u, v = q.squeeze() - origin, p.squeeze() - origin
        prod = u * v
        return torch.sum(F.relu(-prod))

    def _reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
    
