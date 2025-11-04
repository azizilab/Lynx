import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from ml_collections import ConfigDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv
import torch_scatter

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from base_model import BaseModel
from module import Prior, StructuralPrior, ConvPrior
from module import Encoder, XtoZEncoder, ConvXtoZEncoder, XtoVEncoder, XtoOmegaCluEncoder
from module import Decoder, ZtoSDecoder
from dataset import XeniumDataset, HeteroDataset

EPS = 1e-8


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
        self.prior = Prior(configs)
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
            z_mu, z_logvar = self.prior(u, None)
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

        pz_u, _ = self.prior(u, None)
        qz_xu, _ = self.get_z(x, u, data.edge_index)
        px_z = self.get_x(x, qz_xu)

        return ConfigDict({
            'qz':           qz_xu,
            'pz':           pz_u,
            'px':           px_z
        })
    
    def fit(self, train_configs, train_dl, val_dl: DataLoader, DEBUG=False, log_wandb=False):  
        super().model_train(self, train_configs, train_dl, val_dl, DEBUG=DEBUG, log_wandb=log_wandb)
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
            batch_qz = res.qz.detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()

            for idx, qz_i, pz_i, px_i in zip(data.idx, batch_qz, batch_pz, batch_px):
                qz[idx], pz[idx], px[idx] = qz_i, pz_i, px_i
        
        return ConfigDict({
            'qz':           qz,
            'pz':           pz,
            'px':           px
        })


class HeteroAttnVGAE(BaseModel):
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
        
        # Whether to use conv. prior / posterior for `u` (i.e. histology patches)
        self.patch_size = configs.patch_size if hasattr(configs, 'patch_size') else -1 
        self.num_clusters = configs.num_clusters

        self.prior = StructuralPrior(configs) if self.patch_size < 0 else ConvPrior(configs)
        self.encode_z = XtoZEncoder(configs) if self.patch_size < 0 else ConvXtoZEncoder(configs)
        self.encode_v = XtoVEncoder(configs)
        self.encode_omega = XtoOmegaCluEncoder(configs)
        self.cluster_embedding = nn.Embedding(configs.num_clusters, configs.c_latent)

        self.decode_s = ZtoSDecoder(configs)        
        self.decode_x = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, configs.c_in)
        )

    def model(self, data):
        pyro.module("VAE", self)

        u = data[self.query].x
        x = data[self.ref].x
        l = x.sum(axis=-1, keepdim=True)
        
        # Reshape image patches if paired with histology
        if self.patch_size > 0:
            u = self._reshape_patches(u)

        # Graph properties
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        clusters = data[self.ref].cluster
        
        edge_index = edge_index_dict[self.r2r]
        edge_distances = edge_attr_dict[self.r2r]
        src, dst = edge_index
        src_clusters = clusters[src]    
        n_edges = edge_index.size(1)

        # --- Global parameters ---
        # \theta: gene-specific dispersion
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        # \kappa: cluster-specific embedding
        kappa = self.cluster_embedding(clusters).to(self.device)
        
        # --------------------------
        #  Sample z from p(z | u)
        # --------------------------
        with pyro.plate("patch", u.size(0)):
            z_mu, z_logvar = self.prior(u, edge_index_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))

        # Deterministic unpool z (patch-level) -> s (cell-level)
        s = self.decode_s(
            z, kappa, edge_index_dict, 
            celltype_aware=self.configs.celltype_aware
        )
            
        # -------------------------------------------------------------
        #  Sample edge weights omega from p(omega | d ; alpha, gamma)
        # -------------------------------------------------------------
        if self.configs.infer_cell_interaction:
            log_d = torch.log1p(edge_distances)
            alpha = self.configs.alpha   # Distance-aware dispersion
            gamma = self._compute_gamma_shift(clusters)[src_clusters]  # Cluster abundance-aware shift
            scale = 1.0 / (edge_index.size(1) / x.size(0))

            # DEBUG: error, unique(dst) to create empty nodes!
            with pyro.plate("r2r_edges", log_d.size(0)):
                with poutine.scale(scale=scale):
                    # Append empty edge priors
                    unique_dst = torch.unique(dst)
                    log_rate = torch.cat([
                        -alpha*log_d + gamma,
                        torch.zeros_like(unique_dst).to(self.device) # Append empty edge priors
                    ], dim=0)

                    gumbel_dist = dist.Gumbel(
                        log_rate, torch.ones_like(log_rate)
                    )
                    omega_raw = pyro.sample('omega', gumbel_dist) / self.configs.temperature

                    # Append "empty" edge per cell to allow relaxed softmax
                    omega = torch_scatter.scatter_softmax(
                        omega_raw, 
                        torch.cat([dst, unique_dst])
                    )[:n_edges]

            # Update s' with linear combination of nbr & cluster identities
            neighbor_effect = self._weighted_sum(edge_index, omega, s)
            cluster_effect = kappa[clusters]
            s = neighbor_effect + cluster_effect

        # --------------------------------------
        #  Reconstruct x from p(x | s, omega)
        # --------------------------------------
        mu = torch.softmax(self.decode_x(s), dim=-1)
        x_mu = l * mu
        logits = logits = (x_mu+EPS).log() - (theta+EPS).log()

        with pyro.plate("cell", x.size(0)):
            nb_dist = dist.NegativeBinomial(total_count=theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, data):
        pyro.module("VAE", self)

        x = data[self.ref].x    # [num_cells, in_dim]
        u = data[self.query].x  # [num_patches, aux_dim]
        x = self.lognorm(x)  

        # Reshape image patches if paired with histology
        if self.patch_size > 0:
            u = self._reshape_patches(u)

        edge_index_dict = data.edge_index_dict
        edge_index = edge_index_dict[self.r2r]

        # -----------------------------------------------
        #  Sample cluster embedding c from q(c | x^hat)
        # -----------------------------------------------
        # bulk_clu = torch.log1p(data[self.ref].bulk_clu)
        # with pyro.plate("clusters", self.configs.num_clusters):
        #     clu_emb_mu, clu_emb_logvar = self.clu_encoder(bulk_clu).chunk(2, dim=-1) 
        #     clu_emb_dist = dist.Normal(clu_emb_mu, torch.exp(clu_emb_logvar/2))
        #     pyro.sample("clu_emb", clu_emb_dist.to_event(1))

        # ------------------------------
        #  Sample z from q(z | x, u)
        # ------------------------------
        with pyro.plate("patch", u.size(0)):
            z_mu, z_logvar, _ = self.encode_z(x, u, edge_index_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            pyro.sample("z", z_dist.to_event(1))

        # ----------------------------------
        #  Sample omega from q(omega | x)
        # ----------------------------------
        if self.configs.infer_cell_interaction:
            x_empty = torch.zeros(x.size(0), self.configs.c_hidden).to(self.device)
            omega_loc = self.encode_omega(x, x_empty, edge_index_dict)
            scale = 1.0 / (edge_index.size(1) / x.size(0))
            with pyro.plate("r2r_edges", omega_loc.size(0)):
                with poutine.scale(scale=scale):
                    pyro.sample("omega", dist.Delta(omega_loc))

    def predict(self, data, device):
        with torch.no_grad():
            data = data.to(device)
            
            # Observed data
            x = data[self.ref].x
            l = x.sum(axis=-1, keepdim=True)
            x = self.lognorm(x)
            u = data[self.query].x

            # Reshape image patches if paired with histology
            if self.patch_size > 0:
                u = self._reshape_patches(u)

            clusters = data[self.ref].cluster           
            edge_index_dict = data.edge_index_dict
            edge_index = edge_index_dict[self.r2r]
            src, dst = edge_index
            n_edges = edge_index.size(1)
            
            # ---------- p(z | u) ------------
            pz, _ = self.prior(u, edge_index_dict)

            # ---------- q(z | x, u) -----------
            qz, _, _ = self.encode_z(x, u, edge_index_dict)
            
            # ---------- deterministic z (patch) -> s (cell) -----------
            # bulk_clu = torch.log1p(data[self.ref].bulk_clu[clusters]) 
            # clu_effect, _ = self.clu_encoder(bulk_clu[clusters]).chunk(2, dim=-1) 
            kappa = self.cluster_embedding(clusters).to(device)
            qs = self.decode_s(
                qz, kappa, edge_index_dict, 
                celltype_aware=self.configs.celltype_aware
            )

            # ---------- q(\omega | x) & p(x | s, \omega) ----------
            q_omega = None
            q_omega_raw = None

            if self.configs.infer_cell_interaction:
                x_empty = torch.zeros(x.size(0), self.configs.c_hidden).to(self.device)
                q_omega_raw = self.encode_omega(x, x_empty, edge_index_dict)
                q_omega_raw /= self.configs.temperature
                q_omega = torch_scatter.scatter_softmax(
                    q_omega_raw, 
                    torch.cat([dst, torch.unique(dst)])
                )[:n_edges]  # Remove empty edges

                neighbor_effect = self._weighted_sum(edge_index, q_omega, qs)
                cluster_effect = kappa[clusters]
                qv = neighbor_effect + cluster_effect
                mu = torch.softmax(self.decode_x(qv), dim=-1)
            else:
                mu = torch.softmax(self.decode_x(qs), dim=-1)

            px = l * mu
                
            return ConfigDict({
                # Latent & reconstructions
                "qz": qz,
                "qs": qs,
                "pz": pz,    
                "px": px,

                # cell-cell attention (edge weight) 
                "omega": q_omega,            
                "omega_logits": q_omega_raw[:n_edges]     
            })

    def fit(self, train_configs, train_dl, val_dl, DEBUG=False, log_wandb=False):
        super().model_train(self, train_configs, train_dl, val_dl, key=self.ref, DEBUG=DEBUG, log_wandb=log_wandb)
        return None
    
    def evaluate(
        self, 
        adata_ref: sc.AnnData,
        adata_query: sc.AnnData,
        graph_data: HeteroDataset,
        n_subgraphs: int = 1,
        device: torch.device = torch.device('cuda')
    ):
        self.eval()
        self.device = device
        self.to(device)

        n_cells, n_features = adata_ref.shape
        n_pixels, _ = adata_query.shape
        n_clusters = adata_ref.obs.leiden.max()+1

        full_graph_data = HeteroDataset(
            adatas_ref=adata_ref, 
            adatas_query=adata_query, 
            n_subgraphs=n_subgraphs,
            k=graph_data.k, r=graph_data.r, alpha=0.,
            cluster_key=graph_data.cluster_key,
            num_clusters=graph_data.num_clusters,
            is_weighted=graph_data.is_weighted,
            ref=graph_data.ref, ref_proj_key=graph_data.ref_proj_key,
            query=graph_data.query, query_proj_key=graph_data.query_proj_key,
            verbose=True
        )

        dataloader = DataLoader(full_graph_data, shuffle=False)
        qz = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)    # lowres latent (patch-res.)
        qs = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)   # hires latent (cell-res.)
        pz = np.zeros_like(qz)
        px = np.zeros((n_cells, n_features), dtype=np.float32)

        # Summarized cell-type specific attention scores
        qomega_scores = np.zeros((n_cells, n_clusters), dtype=np.float32)  

        # Recover batched predictions 
        for data in dataloader:
            res = self.predict(data, device)

            # Extract batched subgraph outputs
            batch_pz = res.pz.detach().cpu().numpy()  # dim: [L, K]
            batch_qz = res.qz.detach().cpu().numpy()  # dim: [L, K]
            batch_qs = res.qs.detach().cpu().numpy()  # dim: [N, K]
            batch_px = res.px.detach().cpu().numpy()  # dim: [N, G]

            # Recover correct global spatial orders from batched predictions
            query_indices = data[self.query].idx.numpy()
            qz[query_indices] = batch_qz
            pz[query_indices] = batch_pz

            ref_indices = data[self.ref].idx.numpy()
            qs[ref_indices] = batch_qs
            px[ref_indices] = batch_px

            if self.configs.infer_cell_interaction:
                batch_omega = res.omega.detach().cpu().numpy()  # dim: [E]
                batch_omega_logits = res.omega_logits.detach().cpu().numpy()  # dim: [E]
  
                # Cell-type specific attention scores (normalized from raw `omega` logits)
                src, dst = data.edge_index_dict[self.r2r].cpu().numpy()
                clusters = data[self.ref].cluster.cpu().numpy()  # [N]
                src_cluster_edges = clusters[src]  # [E]
                cell_idx = data[self.ref].idx[dst].cpu().numpy()  # [E]

                logit_sum   = np.zeros((n_cells, n_clusters), dtype=np.float32)
                logit_count = np.zeros((n_cells, n_clusters), dtype=np.int32)
                np.add.at(logit_sum, (cell_idx, src_cluster_edges), batch_omega_logits)
                np.add.at(logit_count, (cell_idx, src_cluster_edges), 1)

                mean_logits = np.full_like(logit_sum, -1e9, dtype=np.float32)  # -inf where no edges
                mask = logit_count > 0
                mean_logits[mask] = logit_sum[mask] / logit_count[mask]

                row_max = np.max(mean_logits, axis=1, keepdims=True)
                exps = np.exp(mean_logits - row_max)
                exps[~np.isfinite(exps)] = 0.0
                row_sum = exps.sum(axis=1, keepdims=True) + EPS
                qomega_scores[:] = exps / row_sum  # per-cell softmax across clusters

        # In-place storage to adatas
        adata_query.obsm['X_z'] = qz.astype(np.float32)  # Latent (z) for patches
        adata_ref.obsm['X_z'] = qs.astype(np.float32)  # Latent (z) for cells
    
        # Save edge index & weights for visualization
        # TODO: By default, inference-stage only has 1 subgraph
        if self.configs.infer_cell_interaction:
            adata_ref.obsm['omega'] = qomega_scores  # Attention scores summarized per cell
            adata_ref.uns['omega'] = batch_omega   
            adata_ref.uns['edge_index'] = data.edge_index_dict[self.r2r].cpu().numpy()

        return ConfigDict({
            'qzu':          qz,
            'qzx':          qs, 
            'pz':           pz,
            'px':           px,
        })

    def _reshape_patches(self, u):
        r"""Reshape flattened patches to proper image format"""
        batch_size = u.shape[0]
        expected_size = 3 * self.patch_size * self.patch_size
        if u.shape[1] != expected_size:
            raise ValueError(f"Expected flattened patch size {expected_size}, got {u.shape[1]}")
        u_reshaped = u.view(batch_size, 3, self.patch_size, self.patch_size)
        return u_reshaped

    @staticmethod
    def _weighted_sum(edge_index, omega, z):
        r"""Compute weighted neighboring scores per node"""
        src, dst = edge_index
        neighbor_contrib = torch_scatter.scatter_add(
            omega.unsqueeze(-1)*z[src], 
            dst, 
            dim=0, dim_size=z.size(0)
        )	
        return neighbor_contrib

    @staticmethod
    def _compute_gamma_shift(labels):
        r"""Compute cluster-specific edge strength shift
        to account for differential cluster abundance"""
        counts = torch.bincount(labels)
        freq = counts.float() / counts.sum().float()   
        gamma_shift = -torch.log(freq + EPS)
        return gamma_shift
    
