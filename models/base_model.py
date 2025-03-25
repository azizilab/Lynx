import random
import numpy as np
import scanpy as sc

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
from scvi.distributions import NegativeBinomial

from ml_collections import ConfigDict
from tqdm import tqdm, trange

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear, GATConv
from torch_geometric import graphgym
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import StepLR, ExponentialLR

# modules for debug
import gc
from scipy.special import comb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import wandb

EPS = 1e-8


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

    def fit(self, train_configs, dataloader):
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
    
    def evaluate(self, expr, device):
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
    
    
class BaseModel(nn.Module, ABC):
    r"""Base class for multi-modal VGAEs"""
    def __init__(
        self, 
        configs: ConfigDict, 
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        self.configs = configs
        self.device = device
        self.to(device)

    @abstractmethod
    def model(self, data: Data):
        r"""Generative model"""
        pass

    @abstractmethod
    def guide(self, data: Data):
        r"""Variational guide"""
        pass

    @abstractmethod
    def predict(self, data: Data, device: torch.device):
        r"""Get latent (z) & reconstructions from batched data object"""
        pass

    @abstractmethod
    def fit(
        self, train_configs: ConfigDict, 
        train_dataloader: DataLoader, val_dataloader: DataLoader, 
        DEBUG: str = False
    ):
        r"""Full model training"""
        pass

    @abstractmethod
    def evaluate(
        self, adata: sc.AnnData, k: int, 
        n_subgraphs: int, device: torch.device
    ):
        r"""Full model inference"""
        pass

    def load_state(self, save_path):
        self.load_state_dict(torch.load(save_path))
    
    def model_train(
        self, model, train_configs: ConfigDict, 
        train_dl: DataLoader, val_dl: DataLoader,
        key: str = None, save_path: str = 'best_model.pth', 
        DEBUG: bool = False, 
        log_wandb: bool = False
    ):
        # Setup optimizer & inference schemes
        svi, scheduler, progress_bar = self.setup(model, train_configs)
        
        # Loss configs
        train_losses, val_losses = [], []
        patience = train_configs.patience
        max_patience = train_configs.patience
        warmup_epochs = train_configs.warmup_epochs
        max_beta = model.configs.beta
        min_val_loss = np.inf

        # Debug configs
        r2, qz_corr_score, pz_corr_score, pz_corr_scores, qz_corr_scores = 0., 0., 0., [], []
    
        for epoch in progress_bar:
            if train_configs.anneal:
                model.configs.beta = self.get_anneal_weight(max_beta, epoch, warmup_epochs)
            train_loss = self.train_step(model, train_dl, svi, key=key, device=train_configs.device)
            scheduler.step()
            val_loss = self.val_step(model, val_dl, svi, key=key, device=train_configs.device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save the best model params
            min_val_loss, patience = self.checkpoint(
                val_loss, min_val_loss, patience, max_patience, save_path
            )
            if patience == 0:
                break

            # DEBUG: disentanglement monitor
            if DEBUG and epoch % 10 == 0:
                data = next(iter(val_dl))
                pz_corr_score, qz_corr_score, r2 = self.monitor_metrics(data, key=key, device=train_configs.device)
                pz_corr_scores.append(pz_corr_score)
                qz_corr_scores.append(qz_corr_score)

            # Log results to wandb
            if log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "r2": r2,
                        "qz_corr_score": qz_corr_score
                    }
                )

            self.set_desc(progress_bar, epoch, train_loss, val_loss, r2, qz_corr_score, pz_corr_score, DEBUG)
            gc.collect()

        # self.load_state_dict(torch.load(save_path))  # Load the best model
        self.plot_latent_corr(pz_corr_scores, qz_corr_scores)
        self.plot_loss(train_losses, val_losses)
        return None
    
    def monitor_metrics(self, data: Data, device: torch.device, key: str = None):
        r"""(Debug-only) Monitor latent factor correlations & reconstruction"""
        res = self.predict(data, device)

        # Latent factor correlations
        pz = res.pz.detach().cpu().numpy()
        qz = res.qz.detach().cpu().numpy()
        px = res.px.detach().cpu().numpy() \
            if 'px' in res.keys() else \
            res.py.detach().cpu().numpy()

        # Compute avg. pariwise factor correlations (lower triangular matrix)
        pz_corr = np.corrcoef(pz.T)
        pz_corr_score = np.abs(np.tril(pz_corr, k=-1)).sum() / comb(pz_corr.shape[0], 2) 
        qz_corr = np.corrcoef(qz.T)
        qz_corr_score = np.abs(np.tril(qz_corr, k=-1)).sum() / comb(qz_corr.shape[0], 2)

        # Reconstruction quality
        r2 = r2_score(
            data.x.detach().cpu().numpy().flatten() \
            if key is None else \
            data[key].x.detach().cpu().numpy().flatten(), 
            px.flatten()
        )
        return pz_corr_score, qz_corr_score, r2
    
    def checkpoint(self, curr_loss, min_loss, patience, max_patience, save_path):
        if curr_loss < min_loss:
            min_loss = curr_loss
            patience = max_patience
            torch.save(self.state_dict(), save_path)
        else:
            patience -= 1
        return min_loss, patience
    
    @property
    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pyro.set_rng_seed(seed)
        return None

    @property
    def init_model_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, GATConv):
                nn.init.xavier_uniform_(m.lin.weight)
            else:
                graphgym.init(m)
    
    @staticmethod
    def setup(model: nn.Module, train_configs: ConfigDict):
        r"""Setup optimizer & inference objects"""
        model.device = train_configs.device
        model.to(train_configs.device)

        optim_params = {
            'lr': train_configs.lr,
            'weight_decay': train_configs.weight_decay,
            'betas': train_configs.betas,
        }
        scheduler_params = {
            'optimizer': torch.optim.AdamW,
            # 'step_size': train_configs.step_size,
            'gamma': train_configs.gamma,
            'optim_args' : optim_params
        }
        # scheduler = StepLR(scheduler_params)
        scheduler = ExponentialLR(scheduler_params)
        elbo = Trace_ELBO()
        svi = SVI(model.model, model.guide, scheduler, elbo)
        pbar = tqdm(range(train_configs.n_epochs))

        return svi, scheduler, pbar
    
    @staticmethod
    def train_step(
        model: nn.Module, dataloader: DataLoader, svi: SVI, 
        device: torch.device, key: str = None
    ):
        r"""Single-epoch training step"""
        model.train()
        total_loss, n_obs = 0., 0.

        batch = next(iter(dataloader))
        batch = batch.to(device)

        for data in dataloader:
            data = data.to(device)
            loss = svi.step(data)
            n_obs += data.x.size(0) if key is None else data[key].x.size(0)
            total_loss += loss

        return total_loss / n_obs
    
    @staticmethod
    def val_step(
        model: nn.Module, dataloader: DataLoader, svi: SVI,
        device: torch.device, key: str = None
    ):
        r"""Single-epoch validation step"""
        model.eval()
        total_loss, n_obs = 0., 0.

        batch = next(iter(dataloader))
        batch = batch.to(device)

        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                loss = svi.evaluate_loss(data)
                n_obs += data.x.size(0) if key is None else data[key].x.size(0)
                total_loss += loss
        return total_loss / n_obs
    
    @staticmethod
    def set_desc(
        pbar: tqdm, epoch: int, train_loss: float, val_loss: float,
        r2: float = 0., qz_corr_score: float = 0., pz_corr_score: float = 0., DEBUG: bool = False
    ):
        if DEBUG:
            pbar.set_description(
                "Epoch {0} train -ELBO: {1}; val -ELBO: {2}; val R2: {3}; val corr: {4}; pz corr: {5}".format(
                    epoch, 
                    np.round(train_loss, 3), 
                    np.round(val_loss, 3), 
                    np.round(r2, 3), 
                    np.round(qz_corr_score, 3),
                    np.round(pz_corr_score, 3)
                )
            ) 
        else:
            pbar.set_description(
                "Epoch {0} train -ELBO: {1}; val -ELBO: {2}".format(
                    epoch, 
                    np.round(train_loss, 3), 
                    np.round(val_loss, 3), 
                )
            )       
        
        return None
    
    @staticmethod
    def lognorm(x):
        l = x.sum(axis=-1, keepdim=True) + EPS
        x = x / l * l.median() 
        return torch.log1p(x)  

    @staticmethod
    def get_anneal_weight(beta, epoch, warmup_epochs):
        return min(beta, (epoch+1)/warmup_epochs)
        
    @staticmethod
    def plot_loss(train_losses, val_losses):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.arange(len(train_losses)), train_losses, label='Train')
        ax.plot(np.arange(len(val_losses)), val_losses, label='Val')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('-ELBO')

        ax.legend()
        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        fig.show()
        
        return None
    
    @staticmethod
    def plot_latent_corr(pz_corr_scores, qz_corr_scores):
        if len(pz_corr_scores) == 0 or len(qz_corr_scores) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.arange(len(pz_corr_scores)), pz_corr_scores, '.--', label='Prior')
        ax.plot(np.arange(len(qz_corr_scores)), qz_corr_scores, '.--', label='Posterior')

        ax.set_xlabel('Epoch checkpoint')
        ax.set_ylabel('Avg. factor correlations')
        ax.legend()

        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        fig.show()

        return None
