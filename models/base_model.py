import random
import wandb
import numpy as np
import scanpy as sc

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from torch.distributions import Normal, kl_divergence
# from scvi.distributions import NegativeBinomial

from ml_collections import ConfigDict
from tqdm import tqdm, trange

from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear, GATConv, GCNConv
from torch_geometric import graphgym
from lightning.pytorch import seed_everything
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ExponentialLR

# modules for debug
import gc
from scipy.special import comb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import scanpy


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

        # Clear existing plates
        pyro.clear_param_store()
        torch.cuda.empty_cache()

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
        self, dataset: Dataset, 
        train_configs: ConfigDict, 
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
        self, model, dataset: Dataset, train_configs: ConfigDict,  
        key: str = None, save_path: str = 'best_model.pth', 
        DEBUG: bool = False, log_wandb: bool = False
    ):
        pyro.clear_param_store()
        torch.cuda.empty_cache()
        
        # Setup optimizer & inference schemes
        svi, scheduler, progress_bar = self.setup(model, train_configs)
        
        # Loss configs
        train_losses, val_losses = [], []
        patience = train_configs.patience
        max_patience = train_configs.patience
        warmup_epochs = train_configs.warmup_epochs
        max_beta = model.configs.beta
        min_val_loss = np.inf

        # Train-test split
        train_data, val_data = random_split(dataset, [0.7, 0.3])
        train_dl, val_dl = DataLoader(train_data, shuffle=True), DataLoader(val_data)

        # Debug configs
        r2, qz_corr_score, pz_corr_score, pz_corr_scores, qz_corr_scores = 0., 0., 0., [], []
        for epoch in progress_bar:
            if train_configs.anneal:
                model.configs.beta = self.get_anneal_weight(max_beta, epoch, warmup_epochs)

            train_loss = self.train_step(model, train_dl, svi, key=key, device=train_configs.device)
            val_loss = self.val_step(model, val_dl, svi, key=key, device=train_configs.device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step()

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

        self.load_state_dict(torch.load(save_path))  # Load the best model
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
    def set_seed(self):
        torch.manual_seed(self.configs.seed)
        seed_everything(self.configs.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        pyro.set_rng_seed(self.configs.seed)
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
                "Epoch {0} train -ELBO: {1}; val -ELBO: {2}; val R2: {3}; q(z) corr: {4}; p(z) corr: {5}".format(
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
        return min(beta, epoch+1/warmup_epochs)
        
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


class SpatialVGAE(BaseModel):
    r"""Ablation baseline: LYNX without auxiliary `u` modality.
    Builds only a spatial KNN graph over `ref` (Xenium) cells and learns
    cell-level latent codes purely from expression + spatial context.
    """

    def __init__(self, configs: ConfigDict, device: torch.device = torch.device('cuda')):
        super().__init__(configs, device)
        self.ref = configs.ref
        self.r2r = (self.ref, 'to', self.ref)
        act = configs.act

        # --- Spatial GCN encoder  x -> hidden -> {z_mu, z_logvar} ---
        self.gcn1 = GCNConv(configs.c_in, configs.c_hidden)
        self.act  = act
        self.to_zmu     = nn.Linear(configs.c_hidden, configs.c_latent)
        self.to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

        # --- Decoder  z -> x  (identical head to HeteroAttnVGAE) ---
        self.decode_x = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            act,
            nn.Linear(configs.c_hidden, configs.c_in)
        )

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.act(self.gcn1(x, edge_index))
        return self.to_zmu(h), self.to_zlogvar(h)

    # ------------------------------------------------------------------
    def model(self, data):
        pyro.module("SpatialVGAE", self)
        x = data[self.ref].x
        l = x.sum(dim=-1, keepdim=True)
        N = x.size(0)

        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        with pyro.plate("cell", N):
            z = pyro.sample(
                "z",
                dist.Normal(
                    torch.zeros(N, self.configs.c_latent, device=self.device),
                    torch.ones(N, self.configs.c_latent, device=self.device)
                ).to_event(1)
            )
            mu  = torch.softmax(self.decode_x(z), dim=-1)
            x_mu = l * mu
            logits = (x_mu + EPS).log() - (theta + EPS).log()
            pyro.sample("x", dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1), obs=x)

    def guide(self, data):
        pyro.module("SpatialVGAE", self)
        x = data[self.ref].x
        edge_index = data.edge_index_dict[self.r2r]
        x_norm = self.lognorm(x)

        z_mu, z_logvar = self._encode(x_norm, edge_index)

        with pyro.plate("cell", x.size(0)):
            pyro.sample(
                "z",
                dist.Normal(z_mu, torch.exp(z_logvar / 2)).to_event(1)
            )

    def predict(self, data, device: torch.device):
        with torch.no_grad():
            data = data.to(device)
            x = data[self.ref].x
            l = x.sum(dim=-1, keepdim=True)
            edge_index = data.edge_index_dict[self.r2r]
            x_norm = self.lognorm(x)

            z_mu, _ = self._encode(x_norm, edge_index)
            mu  = torch.softmax(self.decode_x(z_mu), dim=-1)
            px  = l * mu

            return ConfigDict({
                'qz': z_mu,
                'qs': z_mu,
                'pz': torch.zeros_like(z_mu),
                'px': px,
            })

    def fit(self, dataset, train_configs, DEBUG=False, log_wandb=False):
        super().model_train(
            self, dataset, train_configs, key=self.ref,
            DEBUG=DEBUG, log_wandb=log_wandb
        )
        return None

    def evaluate(
        self,
        adata_ref: sc.AnnData,
        adata_query: sc.AnnData,
        graph_data,
        n_subgraphs: int = 1,
        device: torch.device = torch.device('cuda')
    ):
        r"""Full inference - writes ``X_z`` into ``adata_ref.obsm``."""
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from dataset import HeteroDataset

        self.eval()
        self.device = device
        self.to(device)

        n_cells, n_features = adata_ref.shape
        n_pixels = adata_query.shape[0]
        n_clusters = graph_data.num_clusters

        full_graph_data = HeteroDataset(
            adatas_ref=adata_ref,
            adatas_query=adata_query,
            n_subgraphs=n_subgraphs,
            k=graph_data.k, r=graph_data.r, alpha=getattr(self.configs, 'alpha', 1.0),
            cluster_key=graph_data.cluster_key,
            num_clusters=n_clusters,
            is_weighted=graph_data.is_weighted,
            ref=graph_data.ref, ref_proj_key=graph_data.ref_proj_key,
            query=graph_data.query, query_proj_key=graph_data.query_proj_key,
            is_ref_grid=graph_data.is_ref_grid,
            is_query_grid=graph_data.is_query_grid,
            verbose=False
        )

        dataloader = PyGDataLoader(full_graph_data, shuffle=False)
        qs = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)
        px = np.zeros((n_cells, n_features), dtype=np.float32)

        data = next(iter(dataloader))
        res  = self.predict(data, device)

        ref_indices = data[self.ref].idx.numpy()
        qs[ref_indices] = res.qs.detach().cpu().numpy()
        px[ref_indices] = res.px.detach().cpu().numpy()

        adata_ref.obsm['X_z'] = qs.astype(np.float32)

        return ConfigDict({'qs': qs, 'px': px})
