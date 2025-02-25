import os
import sys
import random
import numpy as np
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from abc import ABC, abstractmethod
from typing import Callable
from ml_collections import ConfigDict
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear, GATConv, GATv2Conv, GCNConv, LGConv
from torch_geometric import graphgym
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import AdamW
from pyro.optim import StepLR
import torch_scatter

# modules for debug
import gc
from scipy.special import comb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from module import Prior
from module import Encoder, GATEncoder, PhenotypeEncoder
from module import Decoder, GATDecoder
from dataset import XeniumDataset, HeteroDataset

EPS = 1e-8


class BaseModel(nn.Module, ABC):
    r"""Base Class for multi-modal VGAEs"""
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
        DEBUG: bool = False
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
            'step_size': train_configs.step_size,
            'gamma': train_configs.gamma,
            'optim_args' : optim_params
        }
        scheduler = StepLR(scheduler_params)
        # optimizer = AdamW(optim_params)
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
        # if hasattr(model, 'init_lazy_modules'):
        #     model.init_lazy_modules(batch)

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
        r2: float = 0., corr_score: float = 0., pz_corr_score: float = 0., DEBUG: bool = False
    ):
        if DEBUG:
            pbar.set_description(
                "Epoch {0} train -ELBO: {1}; val -ELBO: {2}; val R2: {3}; val corr: {4}; pz corr: {5}".format(
                    epoch, 
                    np.round(train_loss, 3), 
                    np.round(val_loss, 3), 
                    np.round(r2, 3), 
                    np.round(corr_score, 3),
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
        plt.show()
        
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
        plt.show()

        return None


class VGAE(BaseModel):
    r"""Learning latent manifold w/ Conditional VGAE
    Generative path: DESI (u) -> Latent (z) -> Xenium (x)
    """
    def __init__(
        self, 
        configs: ConfigDict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(configs, device)
        self.prior = Prior(configs, device=device)
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)

    def model(self, data):
        pyro.module("VGAE", self)
        x = data.x
        u = data.u

        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z_mu, z_logvar = self.prior(u)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))

            mu = self.decode(z)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - (self.theta).log()

            nb_dist = dist.NegativeBinomial(total_count=self.theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, data):
        pyro.module("VGAE", self)
        x = data.x
        u = data.u
        edge_index = data.edge_index

        x = torch.log1p(x)
        z_mu, z_logvar = self.encode(x, u, edge_index)  # Global sample per subgraph

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta): 
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            pyro.sample("z", z_dist.to_event(1)) 

    def get_z(self, x, u, edge_index):
        x = torch.log1p(x)
        return self.encode(x, u, edge_index)
    
    def get_x(self, x, z):
        self.eval()
        l = x.sum(axis=-1, keepdim=True)         
        x_mu = l * self.decode(z)
        return x_mu
    
    def predict(self, data: Data, device: torch.device):
        r"""Get latent representation & predictions from batched data object"""
        self.eval()
        data = data.to(device)
        x = data.x.float()
        u = data.u.float()

        pz_u, _ = self.prior(u)
        qz_xu, _ = self.get_z(x, u, data.edge_index)
        px_z = self.get_x(x, qz_xu)

        return ConfigDict({
            'qz':           qz_xu,
            'pz':           pz_u,
            'px':           px_z
        })
    
    def fit(self, train_configs, train_dl, val_dl: DataLoader, DEBUG=False):  
        super().model_train(self, train_configs, train_dl, val_dl, DEBUG=DEBUG)
        return None
        
    def evaluate(
        self, 
        adata: sc.AnnData,
        k: int = 30, 
        n_subgraphs: int = 8, 
        device: torch.device = torch.device('cuda')
    ):
        r"""Full inference"""
        self.eval()
        self.device = device
        self.to(device)

        graph_data = XeniumDataset(adata, k=k, n_subgraphs=n_subgraphs)

        dataloader = DataLoader(graph_data, shuffle=False)
        qz = np.zeros((adata.shape[0], self.configs.c_latent), dtype=np.float32)
        pz = np.zeros_like(qz)
        px = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)

        # Recover batched predictions in correct spatial orders
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qz = res.qz_params[0].detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()

            for idx, qz_i, pz_i, px_i in zip(data.idx, batch_qz, batch_pz, batch_px):
                qz[idx], pz[idx], px[idx] = qz_i, pz_i, px_i
        
        return ConfigDict({
            'qz':           qz,
            'pz':           pz,
            'px':           px
        })
        

# TODO: add v in both encoder & decoder
class HeteroVGAE(BaseModel):
    r"""Learning latent manifold w/ Conditional VGAE on hetero-graph
    Generative path: DESI (u) -> Latent (z) -> Xenium (x)
    """
    def __init__(
        self,
        configs: ConfigDict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(configs, device)

        self.act = configs.act
        
        # Parse node & edge types
        self.ref = configs.ref
        self.query = configs.query
        self.r2q = (self.ref, 'to', self.query)
        self.q2r = (self.query, 'to', self.ref)
        self.r2r = (self.ref, 'to', self.ref)

        self.prior = Prior(configs)
        self.cluster_embedding = nn.Embedding(configs.num_clusters, configs.c_latent)
        self.num_clusters = configs.num_clusters

        self.x_to_hidden = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            self.act,
            nn.Linear(configs.c_hidden, configs.c_hidden),
            self.act
        )
        self.u_to_hidden = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            self.act,
            nn.Linear(configs.c_hidden, configs.c_hidden),
            self.act
        )


        

        self.z_encoder = GATConv(
            (configs.c_hidden, configs.c_hidden),
            configs.c_hidden,
            heads=1,
            concat=False,
            add_self_loops=False,
            residual=False
        )
        self.qz_mu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.qz_logvar =  nn.Linear(configs.c_hidden, configs.c_latent)

        self.z_decoder = GATConv(
            (configs.c_latent, configs.c_latent),
            configs.c_latent,
            heads=1,
            concat=False,
            add_self_loops=False,
            residual=True
        )

        self.unpool_z = GATConv(
            (configs.c_latent, configs.c_in), configs.c_latent,
            heads=1, concat=False, add_self_loops=False, residual=False
        ) 

        self.pv_mu = nn.Sequential(
            # nn.Linear(configs.c_latent, configs.c_latent),
            self.act,
            nn.Linear(2*configs.c_latent, configs.c_latent),
        )
        self.pv_logvar = nn.Sequential(
            # nn.Linear(configs.c_latent, configs.c_latent),
            self.act,
            nn.Linear(2*configs.c_latent, configs.c_latent),
        )

        self.out = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            self.act,
            nn.Linear(configs.c_hidden, configs.c_in)
        )

        self.gamma_attn = nn.Sequential(
            nn.Linear(configs.c_latent*2+configs.c_latent*2, configs.c_latent),
            configs.act,     
            nn.Linear(configs.c_latent, configs.c_latent),
            configs.act,            
            nn.Linear(configs.c_latent, 2)  
        )

        self.c_2_v = nn.Sequential(
            nn.Linear(2*configs.c_latent, 2*configs.c_latent),
            # configs.act,                 
            # nn.Linear(configs.c_latent, configs.c_latent)  
        )

        self.lognorm_proj = nn.Sequential(
            nn.Linear((configs.c_hidden*2+configs.c_latent)*2+1, configs.c_hidden),
            configs.act,                 
            nn.Linear(configs.c_hidden, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, 2)  
        )

        self.qv_encoder = nn.Sequential(
            nn.Linear(configs.c_hidden*2+configs.c_latent, configs.c_hidden),
            self.act,
            nn.Linear(configs.c_hidden, configs.c_hidden),
            self.act,
        )
        self.qv_mu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.qv_logvar =  nn.Linear(configs.c_hidden, configs.c_latent)

    def model(self, data):
        pyro.module("VAE", self)

        u = data[self.query].x
        x = data[self.ref].x
        clusters = data[self.ref].cluster
        l = x.sum(axis=-1, keepdim=True)

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        c = self.cluster_embedding(clusters).to(self.device)
        # c = F.one_hot(clusters, num_classes=self.num_clusters).float().to(self.device)
        # -------------------------------------------------------------------------
        #  SAMPLE z FROM p(z | u)
        # -------------------------------------------------------------------------

        # Conditional prior: query-dim
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar = self.prior(u)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))


        # -------------------------------------------------------------------------
        #  2) SAMPLE alpha (S_r2r) FROM p(alpha | c, z)
        # -------------------------------------------------------------------------

        z_avg = self.z_decoder((z, c), edge_index_dict[self.q2r], edge_attr_dict[self.q2r])
        hires_feats = torch.cat([c, z_avg], dim=1)

        edge_index = edge_index_dict[self.r2r]       # shape [2, E]
        edge_distances = edge_attr_dict[self.r2r]    # shape [E]
        src, dst = edge_index

        #Concatenate the cell-type embeddings for src & dst
        node_src = hires_feats[src]   # shape [E, 2*c_latent]
        node_dst = hires_feats[dst]   # shape [E, 2*c_latent]
        node_ij = torch.cat([node_src, node_dst], dim=-1)  # shape [E, 4*c_latent]

        #Pass to gamma
        out = self.gamma_attn(node_ij).exp() + EPS  # shape [E, 2]
        alpha_ij = out[:, 0] * edge_distances #scale by distance for prior
        beta_ij = out[:, 1]

        with pyro.plate("r2r_edges", alpha_ij.size(0)):
            S_ij = pyro.sample(
                "S_r2r", 
                dist.Gamma(concentration=alpha_ij, rate=beta_ij)
            )


        # -------------------------------------------------------------------------
        #  SAMPLE v FROM p(v | c, z, alpha)
        # -------------------------------------------------------------------------

        with pyro.plate("hires", x.size(0)):
            W_ij = self.normalize_edges(S_ij, dst, x.size(0))

            v_feats = self.c_2_v(hires_feats)
            v_feats_src = v_feats[src]
            weighted_edges = W_ij.unsqueeze(-1) * v_feats_src  # shape [E, c_latent]

            pv = torch_scatter.scatter_add(weighted_edges, dst, dim=0, dim_size=x.size(0)) + v_feats #residual

            pv_mu = self.pv_mu(pv)
            pv_logvar = self.pv_logvar(pv)
            v = pyro.sample("v", dist.Normal(pv_mu, torch.exp(pv_logvar/2)).to_event(1))

            # -------------------------------------------------------------------------
            #  SAMPLE x FROM p(x | v)
            # -------------------------------------------------------------------------

            mu = self.out(v)
            mu = torch.softmax(mu, dim=-1)

            x_mu = l * mu
            logits = (x_mu+EPS).log() - theta.log()

            nb_dist = dist.NegativeBinomial(total_count=theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, data):
        pyro.module("VAE", self)

        x = data[self.ref].x          # [num_hires, in_dim]
        u = data[self.query].x        # [num_lowres, aux_dim]
        clusters = data[self.ref].cluster
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        edge_index_r2r = edge_index_dict[self.r2r]        # [2, E_r2r]
        edge_distances = edge_attr_dict[self.r2r]         # [E_r2r]
        src, dst = edge_index_r2r

        c = self.cluster_embedding(clusters).to(self.device)
        # c = F.one_hot(clusters, num_classes=self.num_clusters).float().to(self.device)

        x = self.lognorm(x)  

        x = self.x_to_hidden(x)
        u = self.u_to_hidden(u)

        #aggregate u by average
        u_neighbors = u[edge_index_dict[self.q2r][0]]
        u_avg = torch_scatter.scatter_mean(u_neighbors, edge_index_dict[self.q2r][1], dim=0, dim_size=x.shape[0])

        hires_feats = torch.cat([c, x, u_avg], dim=-1)

        # -------------------------------------------------------------------------
        #  SAMPLE z FROM q(z | x, u)
        # -------------------------------------------------------------------------
        with pyro.plate("lowres", u.size(0)):
            qz = self.z_encoder((x, u), edge_index_dict[self.r2q])
            qz = self.act(qz)
            z_mu = self.qz_mu(qz)
            z_logvar = self.qz_logvar(qz)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar / 2)).to_event(1)
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("z", z_dist)

        # -------------------------------------------------------------------------
        #  SAMPLE v FROM q(v | c, x, u)
        # -------------------------------------------------------------------------
        qv = self.qv_encoder(hires_feats)  # shape [num_hires, hidden]

        v_mu = self.qv_mu(qv)
        v_logvar = self.qv_logvar(qv)
        with pyro.plate("hires", x.size(0)):
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("v", dist.Normal(v_mu, torch.exp(v_logvar / 2)).to_event(1))

        # -------------------------------------------------------------------------
        #  SAMPLE alpha (S_r2r) FROM q(alpha | c, x, u)
        # -------------------------------------------------------------------------
        x_src = hires_feats[src]                         
        x_dst = hires_feats[dst]
        dist_col = edge_distances.unsqueeze(-1)          

        edge_feats = torch.cat([x_src, x_dst, dist_col], dim=-1)  # shape [E_r2r, hires_feats + 1]
        alpha_params = self.lognorm_proj(edge_feats)  
        loc_alpha = alpha_params[:, 0]
        scale_alpha = alpha_params[:, 1].exp() + EPS

        with pyro.plate("r2r_edges", edge_feats.size(0)):
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("S_r2r", dist.LogNormal(loc_alpha, scale_alpha))



    def aggregator_lowres(self, u_tensor, edge_index_q2r, num_hires):
        """
        For each high-resolution node i, gather the u-values of all 
        lower-resolution nodes j that connect (j->i) in edge_index_q2r,
        and compute the mean. Returns [num_hires, u_dim].
        """
        src, dst = edge_index_q2r  # shape [2, E_q2r]
        u_src = u_tensor[src]      # shape [E_q2r, u_dim]
        # scatter mean by hi-res index
        u_agg = torch_scatter.scatter_mean(u_src, dst, dim=0, dim_size=num_hires)
        return u_agg

    def normalize_edges(self, S, indices, size):
        S_sums = torch_scatter.scatter_add(S, indices, dim=0, dim_size=size)  

        W = S / (S_sums[indices] + EPS)  # shape [E]

        return W

    def predict(self, data, device):
        with torch.no_grad():
            data = data.to(device)
            
            # Observed data
            x = data[self.ref].x
            l = x.sum(axis=-1, keepdim=True)

            x = self.lognorm(x)
            u = data[self.query].x

            x = self.x_to_hidden(x)
            u = self.u_to_hidden(u)

            clusters = data[self.ref].cluster
            # c = F.one_hot(clusters, num_classes=self.num_clusters).float().to(device)
            c = self.cluster_embedding(clusters).to(self.device)
            
            edge_index_dict = data.edge_index_dict
            edge_attr_dict = data.edge_attr_dict

            edge_index_r2r = edge_index_dict[self.r2r]        # [2, E_r2r]
            edge_distances = edge_attr_dict[self.r2r]         # [E_r2r]
            src, dst = edge_index_r2r

            #aggregate u by average
            u_neighbors = u[edge_index_dict[self.q2r][0]]
            u_avg = torch_scatter.scatter_mean(u_neighbors, edge_index_dict[self.q2r][1], dim=0, dim_size=x.shape[0])

            hires_feats = torch.cat([c, x, u_avg], dim=-1)

            # ---------- z from p(z| u ) -----------
            pz, _ = self.prior(data[self.query].x)  # e.g. shape [num_lowres, latent_dim]

            # ---------- z from q(z| x,u ) -----------
            qz, attn_score = self.z_encoder((x, u), edge_index_dict[self.r2q], return_attention_weights=True)
            qz = self.act(qz)
            qz = self.qz_mu(qz)

            # ---------- v from q(v | c, x, u ) ----------
            qv = self.qv_encoder(hires_feats)  # shape [num_hires, hidden]
            qv = self.qv_mu(qv)
            
            # ---------- alpha from q(alpha | c, x, u) ----------

            x_src = hires_feats[src]                         
            x_dst = hires_feats[dst]
            dist_col = edge_distances.unsqueeze(-1)          

            edge_feats = torch.cat([x_src, x_dst, dist_col], dim=-1)  # shape [E_r2r, hires_feats + 1]
            alpha_params = self.lognorm_proj(edge_feats)  
            loc_alpha = alpha_params[:, 0]
            scale_alpha = alpha_params[:, 1].exp() + EPS

            alpha_mean = torch.exp(loc_alpha + 0.5 * (scale_alpha**2))
            qa = self.normalize_edges(alpha_mean, dst, x.size(0))

            # ---------- Predict x p(x | v) -----------
            mu = self.out(qv)
            mu = torch.softmax(mu, dim=-1)

            px = l * mu
            

            return ConfigDict({
                "qz": qz,
                "pz": pz,
                "qa": qa,              
                "qv": qv,                    
                "px": px, 
                "attn_score": attn_score
            })


    def fit(self, train_configs, train_dl, val_dl, DEBUG=False):
        super().model_train(self, train_configs, train_dl, val_dl, key=self.ref, DEBUG=DEBUG)
        return None
    
    def evaluate(
        self, 
        adata_ref: sc.AnnData,
        adata_query: sc.AnnData,
        graph_data: HeteroDataset,
        device: torch.device = torch.device('cuda')
    ):
        self.eval()
        self.device = device
        self.to(device)

        n_cells, n_features = adata_ref.shape
        n_pixels, _ = adata_query.shape

        full_graph_data = HeteroDataset(
            adatas_ref=adata_ref, 
            adatas_query=adata_query, 
            n_subgraphs=1,
            k=graph_data.k, r=graph_data.r, is_weighted=graph_data.is_weighted,
            ref=graph_data.ref, ref_proj_key=graph_data.ref_proj_key,
            query=graph_data.query, query_proj_key=graph_data.query_proj_key
        )

        dataloader = DataLoader(full_graph_data, shuffle=False)
        qzu = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)    # lowres latent
        qzx = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)   # hires latent x
        qv = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)    # hires latent v
        pz = np.zeros_like(qzu)
        px = np.zeros((n_cells, n_features), dtype=np.float32)
        attn = np.zeros(n_cells, dtype=np.float32)
        num_clusters = len(np.unique(adata_ref.obs.leiden))
        v_attn = np.zeros((n_cells, num_clusters))

        # Temporary accumulators for weighted averages
        qzx_weighted_sum = np.zeros_like(qzx)
        qzx_attention_sum = np.zeros((n_cells), dtype=np.float32)
        qzx_attention_counter = np.zeros((n_cells), dtype=np.float32)

        # Recover batched predictions in correct spatial orders
        for data in dataloader:
            res = self.predict(data, device)

            batch_qzu = res.qz.detach().cpu().numpy()  # dim: [L, K]
            batch_qv = res.qv.detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()
            batch_edges = res.attn_score[0].detach().cpu().numpy().T  # dim: [edges, 2]
            batch_attn = res.attn_score[1].detach().cpu().numpy()    # dim: [edges, 1]


            ###v attn
            # batch_v_edges = res.attn_score[0].detach().cpu().numpy().T  # dim: [edges, 2]
            # batch_v_attn = res.attn_score[1].detach().cpu().numpy()    # dim: [edges, 1]

            # v_attn_sum = np.zeros((n_cells, num_clusters), dtype=np.float32)

            # ref_idx = data[self.ref].idx[batch_v_edges[:, 1]]
            # clusters = data[self.ref].cluster[batch_v_edges[:, 0]]

            # np.add.at(v_attn_sum, (ref_idx, clusters), batch_v_attn.squeeze())

            # #Count How Many Cells Are in Each Cluster
            # counts_by_cluster = adata_ref.obs['leiden'].value_counts().sort_index()

            # counts_cluster = counts_by_cluster.to_numpy()

            # v_attn = v_attn_sum.copy()
            
            # #Divide by the Total Number of Cells in Each Cluster
            # for c in range(num_clusters):
            #     denom = counts_cluster[c]
            #     if denom > 0:
            #         v_attn[:, c] /= denom
            #     else:
            #         v_attn[:, c] = 0


            #################


            query_indices = data[self.query].idx.numpy()
            qzu[query_indices] = batch_qzu
            pz[query_indices] = batch_pz

            ref_indices = data[self.ref].idx.numpy()
            qv[ref_indices] = batch_qv
            px[ref_indices] = batch_px

            # Compute highres latent representations via attention assignments
            for edge, a in zip(batch_edges, batch_attn):
                ref_idx = data[self.ref].idx[edge[0]]
                
                # Update accumulators for highres
                attn[ref_idx] += a
                qzx_weighted_sum[ref_idx] += a * batch_qzu[edge[1]]  # [N, latent_dim]
                qzx_attention_sum[ref_idx] += a  # [N]
                qzx_attention_counter[ref_idx] += 1

        # Average highres latent representations
        valid = qzx_attention_counter > 0
        qzx[valid.squeeze()] = qzx_weighted_sum[valid.squeeze()] / qzx_attention_sum[valid.squeeze(), None]
        attn[valid.squeeze()] = attn[valid.squeeze()] / qzx_attention_counter[valid.squeeze()]


        return ConfigDict({
            'qv':           qv,
            'qzu':          qzu,
            'qzx':          qzx, 
            'pz':           pz,
            'px':           px,
            'attn':         attn,
            # 'qa':       qa
        })
    