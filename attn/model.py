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
from torch_geometric.nn import Linear, GATConv, GATv2Conv, GCNConv, LGConv, GCN
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


    def load_state(self, save_path):
        self.load_state_dict(torch.load(save_path))
    
    
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

class scKI(BaseModel):
    r"""
    Testing cell communcation
    """
    def __init__(
        self,
        configs: ConfigDict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()

        self.configs = configs
        self.device = device

        self.act = configs.act
        
        # Parse node & edge types
        self.ref = configs.ref
        self.query = configs.query
        self.r2q = (self.ref, 'to', self.query)
        self.q2r = (self.query, 'to', self.ref)
        self.r2r = (self.ref, 'to', self.ref)
        self.num_ligands = configs.num_ligands
        self.num_receptors = configs.num_receptors
        self.k = configs.k


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
            residual=False
        )

        self.unpool_z = GATConv(
            (configs.c_latent, configs.c_in), configs.c_latent,
            heads=1, concat=False, add_self_loops=False, residual=False
        ) 

        self.pv_mu = nn.Sequential(
            # nn.Linear(configs.c_latent, configs.c_latent),
            self.act,
            nn.Linear(configs.c_latent, configs.c_latent),
        )
        self.pv_logvar = nn.Sequential(
            # nn.Linear(configs.c_latent, configs.c_latent),
            self.act,
            nn.Linear(configs.c_latent, configs.c_latent),
        )

        self.out = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            self.act,
            nn.Linear(configs.c_hidden, configs.c_in)
        )

        self.gamma_attn = nn.Sequential(
            nn.Linear(configs.c_latent*2, configs.c_latent),
            # configs.act,     
            # nn.Linear(configs.c_latent, configs.c_latent),
            configs.act,            
            nn.Linear(configs.c_latent, 2)  
        )

        self.c_2_v = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_latent),
            # configs.act,                 
            # nn.Linear(configs.c_latent, configs.c_latent)  
        )

        self.lognorm_proj = nn.Sequential(
            nn.Linear((configs.c_hidden)*2+1, configs.c_hidden),
            configs.act,                 
            nn.Linear(configs.c_hidden, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, 2)  
        )

        self.qv_encoder = nn.Sequential(
            nn.Linear(configs.c_hidden, configs.c_hidden),
            # self.act,
            # nn.Linear(configs.c_hidden, configs.c_hidden),
            self.act,
        )
        self.qv_mu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.qv_logvar =  nn.Linear(configs.c_hidden, configs.c_latent)

        self.c_summary = GCNConv(configs.c_latent, configs.c_latent, add_self_loops=False)

        self.g_2_r = nn.Sequential(
            nn.Linear(configs.c_in, configs.num_receptors),
            self.act,
            nn.Linear(configs.num_receptors, configs.num_receptors),
            self.act,
        )

        self.g_2_l = nn.Sequential(
            nn.Linear(configs.c_in, configs.num_receptors),
            self.act,
            nn.Linear(configs.num_receptors, configs.num_ligands),
            self.act,
        )

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

        # c = self.c_summary(c, edge_index_dict[self.r2r], edge_weight=1/edge_attr_dict[self.r2r]) + c
        # -------------------------------------------------------------------------
        #  SAMPLE z FROM p(z | u)
        # -------------------------------------------------------------------------

        # Conditional prior: query-dim
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar = self.prior(u)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            # z_dist = dist.Normal(torch.zeros_like(z_mu, device=self.device), torch.zeros_like(z_logvar, device=self.device))
            z = pyro.sample("z", z_dist.to_event(1))


        # -------------------------------------------------------------------------
        #  SAMPLE alpha (S_r2r) FROM p(alpha | c, z)
        # -------------------------------------------------------------------------

        z_avg = self.z_decoder((z, c), edge_index_dict[self.q2r])
        # z_avg = torch_scatter.scatter_mean(z, dst, dim=0, dim_size=x.size(0))
        hires_feats = torch.cat([z_avg], dim=1)

        edge_index = edge_index_dict[self.r2r]       # shape [2, E]
        edge_distances = edge_attr_dict[self.r2r]    # shape [E]
        src, dst = edge_index

        #Concatenate the cell-type embeddings for src & dst
        node_src = hires_feats[src]   # shape [E, 2*c_latent]
        node_dst = hires_feats[dst]   # shape [E, 2*c_latent]
        node_ij = torch.cat([node_src, node_dst], dim=-1)  # shape [E, 4*c_latent]

        out = self.gamma_attn(node_ij)  # shape [E, 2]
        alpha_ij = out[:, 0] * edge_distances #scale by distance for prior
        beta_ij = out[:, 1].exp() + EPS

        with pyro.plate("r2r_edges", alpha_ij.size(0)):
            S_ij = pyro.sample(
                "S_r2r", 
                dist.LogNormal(loc=alpha_ij, scale=beta_ij)
            )


        # -------------------------------------------------------------------------
        #  SAMPLE v FROM p(v | c, z, alpha)
        # -------------------------------------------------------------------------

        with pyro.plate("hires", x.size(0)):
            W_ij = self.normalize_edges(S_ij, dst, x.size(0))

            v_feats = self.c_2_v(z_avg)
            v_feats_src = v_feats[src]
            weighted_edges = W_ij.unsqueeze(-1) * v_feats_src  # shape [E, c_latent]

            pv = torch_scatter.scatter_add(weighted_edges, dst, dim=0, dim_size=x.size(0)) #+ v_feats #residual

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


            # -------------------------------------------------------------------------
            #  SAMPLE w FROM p(w)
            # -------------------------------------------------------------------------
            # alpha = torch.ones(self.num_ligands, self.num_receptors, device=self.device) * 0.7
            # beta = torch.ones(self.num_ligands, self.num_receptors, device=self.device)

            # pyro.sample(
            #     "W",
            #     dist.Beta(alpha, beta).to_event(2)
            # )

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

        # c = self.cluster_embedding(clusters).to(self.device)
        # c = F.one_hot(clusters, num_classes=self.num_clusters).float().to(self.device)

        x = self.lognorm(x) 

        raw_x = x 

        x_receptors = x[:, data.receptors]
        x_ligands = x[:, data.ligands]

        x = self.x_to_hidden(x)
        u = self.u_to_hidden(u)

        #aggregate u by average
        # u_neighbors = u[edge_index_dict[self.q2r][0]]
        # u_avg = torch_scatter.scatter_mean(u_neighbors, edge_index_dict[self.q2r][1], dim=0, dim_size=x.shape[0])
        

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
        #  SAMPLE v FROM q(v | x)
        # -------------------------------------------------------------------------
        # u_pooled = self.pool_u(x, u, edge_index_dict, attn_scores, weighted=False)
        hires_feats = torch.cat([x], dim=-1)

        qv = self.qv_encoder(hires_feats)  # shape [num_hires, hidden]

        v_mu = self.qv_mu(qv)
        v_logvar = self.qv_logvar(qv)
        with pyro.plate("hires", x.size(0)):
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("v", dist.Normal(v_mu, torch.exp(v_logvar / 2)).to_event(1))

                # -------------------------------------------------------------------------
                #  SAMPLE w FROM q(w | x)
                # -------------------------------------------------------------------------

                # neighbors_padded[i, :] = up to k source nodes that point to node i
                neighbors_padded = self.pad_edges_vectorized(src, dst, x.size(0), self.k)

                # Gather ligand features from the neighbors; shape (N, k, g)
                source_ligands_raw = raw_x[neighbors_padded]

                # shape (N, g) -> (N, 1, g) -> broadcast -> (N, k, g)
                node_indices = torch.arange(x.size(0), device=raw_x.device)
                target_receptors_raw = raw_x[node_indices].unsqueeze(1).expand(-1, self.k, -1)

                # Flatten for MLP, shape (N*k, g)
                lig_flat = source_ligands_raw.reshape(-1, source_ligands_raw.shape[-1])
                rec_flat = target_receptors_raw.reshape(-1, target_receptors_raw.shape[-1])

                x_ligands_hidden = self.g_2_l(lig_flat)  # (N*k, l)
                x_receptors_hidden = self.g_2_r(rec_flat)  # (N*k, r)

                # Reshape back to (N, k, l) and (N, k, r)
                # -> so we can do a bmm for each node
                num_ligands = x_ligands_hidden.shape[-1]
                num_receptors = x_receptors_hidden.shape[-1]

                lig_reshaped = x_ligands_hidden.view(x.shape[0], self.k, num_ligands)  # (N, k, l)
                rec_reshaped = x_receptors_hidden.view(x.shape[0], self.k, num_receptors)  # (N, k, r)

                # We want bmm: (N, l, k) x (N, k, r) -> (N, l, r)
                lig_bmm = lig_reshaped.transpose(1, 2)  # (N, l, k)

                prob = torch.bmm(lig_bmm, rec_reshaped)  # (N, l, r)

                prob = F.sigmoid(prob)

                W = prob

                # W = pyro.sample( # L x R
                #     "W",
                #     dist.Normal(prob, 1).to_event(2)
                # )


        # -------------------------------------------------------------------------
        #  SAMPLE alpha (S_r2r) FROM q(alpha | x)
        # -------------------------------------------------------------------------
        

        x_src = x_ligands[src]  
        x_dst = x_receptors[dst]

        W_edge = W[dst]

        # For bilinear form alpha[e] = x_src[e]^T * W[dst[e]] * x_dst[e]:

        # Reshape for bmm:
        # x_src -> (E, 1, l)
        # W_edge -> (E, l, r)
        # x_dst -> (E, r, 1)
        x_src_ = x_src.unsqueeze(1)    # (E, 1, l)
        x_dst_ = x_dst.unsqueeze(2)    # (E, r, 1)

        # Multiply in two steps:
        alpha = torch.bmm(x_src_, W_edge)   # (E, 1, r)
        alpha = torch.bmm(alpha, x_dst_)    # (E, 1, 1)

        # Squeeze away the trailing singleton dimensions -> shape (E,)
        alpha = alpha.squeeze(-1).squeeze(-1)
        alpha = F.softplus(alpha) + EPS



        x_src = hires_feats[src]                         
        x_dst = hires_feats[dst]
        dist_col = edge_distances.unsqueeze(-1)   

        edge_feats = torch.cat([x_src, x_dst, dist_col], dim=-1)  # shape [E_r2r, hires_feats + 1]
        alpha_params = self.lognorm_proj(edge_feats)  
        loc_alpha = alpha_params[:, 0]
        scale_alpha = alpha_params[:, 1].exp() + EPS

        with pyro.plate("r2r_edges", x_src.size(0)):
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("S_r2r", dist.LogNormal(loc_alpha, scale_alpha))
                # pyro.sample("S_r2r", dist.LogNormal(alpha, 0.1))

    
    def pad_edges_vectorized(self, src: torch.Tensor, dst: torch.Tensor, N: int, k: int) -> torch.Tensor:
        """
        Build an (N x k) padded neighbor matrix, where row i has up to k distinct src nodes
        pointing to i, in ascending order of dst. If a node has more than k in the subgraph,
        we truncate. If it has fewer, we fill with 0.
        """
        # 1) Sort edges by 'dst' ascending
        perm = torch.argsort(dst)  # E
        dst_sorted = dst[perm]     # (E,)
        src_sorted = src[perm]     # (E,)

        # 2) Count how many edges each node has
        unique_nodes, node_counts = torch.unique(dst_sorted, return_counts=True)

        # 3) Build segment boundaries for each node
        #    cum_counts[i] = sum of node_counts up to (but not including) index i
        #    shape is (#unique_nodes + 1,)
        cumsums = torch.cat([
            torch.zeros(1, device=dst.device, dtype=node_counts.dtype),
            node_counts.cumsum(dim=0)
        ])

        # 4) row_index picks which node each edge belongs to, col_index is the edge position within that node
        row_index = dst_sorted  # shape (E,)
        # For edge e, col_index[e] = e - cumsums[node], i.e. the offset in that node's edge list
        col_index = torch.arange(dst_sorted.shape[0], device=dst.device) - cumsums[row_index]

        # 5) Initialize neighbor matrix as all zeros (or -1), then scatter
        neighbors_padded = torch.zeros(N, k, dtype=torch.long, device=dst.device)

        # We only scatter edges whose col_index < k (i.e., truncate extras if a node has > k edges)
        valid_mask = col_index < k
        neighbors_padded[row_index[valid_mask], col_index[valid_mask]] = src_sorted[valid_mask]

        return neighbors_padded

    def pool_u(self, x, u, edge_index_dict, att_scores, weighted=False):
         
        u_neighbors = u[edge_index_dict[self.q2r][0]]
        att_scores = att_scores[1]

        # ---------------------------
        # Weighted Aggregation Option
        # ---------------------------

        if weighted:
            # Multiply each neighbor feature by its attention score
            weighted_neighbors = u_neighbors * att_scores

            # Sum the weighted features for each x node
            weighted_sum = torch_scatter.scatter_add(
                weighted_neighbors,
                edge_index_dict[self.q2r][1],
                dim=0,
                dim_size=x.shape[0]
            )

            # Sum the attention scores for each x node
            att_sum = torch_scatter.scatter_add(
                att_scores,
                edge_index_dict[self.q2r][1],
                dim=0,
                dim_size=x.shape[0]
            )

            # Compute the weighted average (normalizing by the total attention per node)

            # assert(torch.all(att_sum != 0))
            u_weighted_avg = weighted_sum / (att_sum+EPS)


            return u_weighted_avg

        # ---------------------------
        # Mean Aggregation Option
        # ---------------------------
        # Simply take the mean of all neighbor features for each x node
        u_mean = torch_scatter.scatter_mean(
            u_neighbors,
            edge_index_dict[self.q2r][1],
            dim=0,
            dim_size=x.shape[0]
        )

        return u_mean



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

            raw_x = x 

            x_receptors = x[:, data.receptors]
            x_ligands = x[:, data.ligands]

            x = self.x_to_hidden(x)
            u = self.u_to_hidden(u)

            clusters = data[self.ref].cluster
            # c = F.one_hot(clusters, num_classes=self.num_clusters).float().to(device)
            # c = self.cluster_embedding(clusters).to(self.device)
            
            edge_index_dict = data.edge_index_dict
            edge_attr_dict = data.edge_attr_dict

            edge_index_r2r = edge_index_dict[self.r2r]        # [2, E_r2r]
            edge_distances = edge_attr_dict[self.r2r]         # [E_r2r]
            src, dst = edge_index_r2r

            # -------------------------------------------------------------------------
            #  SAMPLE w FROM q(w | x)
            # -------------------------------------------------------------------------

            # neighbors_padded[i, :] = up to k source nodes that point to node i
            neighbors_padded = self.pad_edges_vectorized(src, dst, x.size(0), self.k)

            # Gather ligand features from the neighbors; shape (N, k, g)
            source_ligands_raw = raw_x[neighbors_padded]

            # shape (N, g) -> (N, 1, g) -> broadcast -> (N, k, g)
            node_indices = torch.arange(x.size(0), device=raw_x.device)
            target_receptors_raw = raw_x[node_indices].unsqueeze(1).expand(-1, self.k, -1)

            # Flatten for MLP, shape (N*k, g)
            lig_flat = source_ligands_raw.reshape(-1, source_ligands_raw.shape[-1])
            rec_flat = target_receptors_raw.reshape(-1, target_receptors_raw.shape[-1])

            x_ligands_hidden = self.g_2_l(lig_flat)  # (N*k, l)
            x_receptors_hidden = self.g_2_r(rec_flat)  # (N*k, r)

            # Reshape back to (N, k, l) and (N, k, r)
            # -> so we can do a bmm for each node
            num_ligands = x_ligands_hidden.shape[-1]
            num_receptors = x_receptors_hidden.shape[-1]

            lig_reshaped = x_ligands_hidden.view(x.shape[0], self.k, num_ligands)  # (N, k, l)
            rec_reshaped = x_receptors_hidden.view(x.shape[0], self.k, num_receptors)  # (N, k, r)

            # We want bmm: (N, l, k) x (N, k, r) -> (N, l, r)
            lig_bmm = lig_reshaped.transpose(1, 2)  # (N, l, k)

            prob = torch.bmm(lig_bmm, rec_reshaped)  # (N, l, r)

            prob = F.sigmoid(prob)

            # prob[prob >= 0.5] = 1.0
            # prob[prob < 0.5] = 0

            W = prob
            

            #aggregate u by average
            # u_neighbors = u[edge_index_dict[self.q2r][0]]
            # u_avg = torch_scatter.scatter_mean(u_neighbors, edge_index_dict[self.q2r][1], dim=0, dim_size=x.shape[0])

            # ---------- z from p(z| u ) -----------
            pz, _ = self.prior(data[self.query].x)  # e.g. shape [num_lowres, latent_dim]

            # ---------- z from q(z| x,u ) -----------
            qz, attn_score = self.z_encoder((x, u), edge_index_dict[self.r2q], return_attention_weights=True)
            qz = self.act(qz)
            qz = self.qz_mu(qz)

            u_pooled = self.pool_u(x, u, edge_index_dict, attn_score, weighted=False)

            hires_feats = torch.cat([x], dim=-1)


            # ---------- v from q(v | c, x, u ) ----------
            qv = self.qv_encoder(hires_feats)  # shape [num_hires, hidden]
            qv = self.qv_mu(qv)
            
            # ---------- alpha from q(alpha | c, x, u) ----------

            x_src = x_ligands[src]  
            x_dst = x_receptors[dst]

            W_edge = W[dst]

            # For bilinear form alpha[e] = x_src[e]^T * W[dst[e]] * x_dst[e]:

            # Reshape for bmm:
            # x_src -> (E, 1, l)
            # W_edge -> (E, l, r)
            # x_dst -> (E, r, 1)
            x_src_ = x_src.unsqueeze(1)    # (E, 1, l)
            x_dst_ = x_dst.unsqueeze(2)    # (E, r, 1)

            # Multiply in two steps:
            alpha = torch.bmm(x_src_, W_edge)   # (E, 1, r)
            alpha = torch.bmm(alpha, x_dst_)    # (E, 1, 1)

            # Squeeze away the trailing singleton dimensions -> shape (E,)
            alpha = alpha.squeeze(-1).squeeze(-1)
            alpha = F.softplus(alpha)+EPS



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
                "qa": (edge_index_r2r, qa),    
                # "qa_params": (edge_index_r2r, loc_alpha, scale_alpha),         
                "qv": qv,                    
                "px": px, 
                "attn_score": attn_score
            })
        
    def setup(self, train_configs: ConfigDict):
        r"""Setup optimizer & inference objects"""
        self.device = train_configs.device
        self.to(train_configs.device)

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
        svi = SVI(self.model, self.guide, scheduler, elbo)
        pbar = tqdm(range(train_configs.n_epochs))

        return svi, scheduler, pbar


    def fit(self, train_configs, train_dl, val_dl, DEBUG=False):
        # Setup optimizer & inference schemes
        svi, scheduler, progress_bar = self.setup(train_configs)
        
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