import numpy as np
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
    r"""
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
    