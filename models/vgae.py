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
from module import Encoder, XtoZEncoder, ConvXtoZEncoder, XtoVEncoder, XtoOmegaCluEncoder, XtoKappaEncoder
from module import Decoder, ZtoSDecoder, StoXDecoder
from module import hsic
from dataset import XeniumDataset, HeteroDataset

EPS = 1e-8
EULER_MASCHERONI = 0.5772156649


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

    def fit(self, dataset, train_configs, DEBUG=False, log_wandb=False):  
        super().model_train(self, dataset, train_configs, DEBUG=DEBUG, log_wandb=log_wandb)
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
        self.prior = StructuralPrior(configs) if self.patch_size < 0 else ConvPrior(configs)
        self.encode_z = XtoZEncoder(configs) if self.patch_size < 0 else ConvXtoZEncoder(configs)
        self.encode_kappa = XtoKappaEncoder(configs)
        self.encode_omega = XtoOmegaCluEncoder(configs)
        self.kappa_mu = nn.Embedding(configs.n_cluster, configs.c_latent)
        self.kappa_logvar = nn.Embedding(configs.n_cluster, configs.c_latent)

        self.decode_s = ZtoSDecoder(configs)        
        self.decode_x = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, configs.c_in)
        )
        # self.decode_x = StoXDecoder(configs)

    def model(self, data):
        pyro.module("VAE", self)

        u = data[self.query].x
        x = data[self.ref].x
        clusters = data[self.ref].cluster
        l = x.sum(axis=-1, keepdim=True)
        
        # Reshape image patches if paired with histology
        if self.patch_size > 0:
            u = self._reshape_patches(u)

        # Graph properties
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        
        edge_index = edge_index_dict[self.r2r]
        edge_distances = edge_attr_dict[self.r2r]
        src, dst = edge_index

        # q2r edges for unpooling z -> cells
        q2r_src, q2r_dst = edge_index_dict[self.q2r]

        # --- Global parameters ---
        # \theta: gene-specific dispersion
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        # Plates
        cell_plate = pyro.plate("cell", x.size(0))

        # -----------------------
        #  Sample z ~ p(z | u)
        # -----------------------
        with pyro.plate("patch", u.size(0)):
            z_mu, z_logvar = self.prior(u, edge_index_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))

        # --------------------------------------------
        #  Deterministic unpooling z -> z_cell
        # --------------------------------------------
        z_cell = torch_scatter.scatter_mean(
            z[q2r_src], q2r_dst, dim=0, dim_size=x.size(0)
        )

        if self.configs.infer_cell_interaction:

            # ---------------------------------------------
            #  Sample omega_raw ~ p(omega_raw | distance)
            #  omega = exp(-omega_raw)
            # ---------------------------------------------
            log_d = torch.log1p(edge_distances)
            alpha = self.configs.alpha
            scale = 1.0 / (edge_index.size(1) / x.size(0))

            with pyro.plate("r2r_edges", log_d.size(0)):
                with poutine.scale(scale=scale):
                    log_rate = alpha * log_d #EULER_MASCHERONI removed since the normalization c and otherwise makes negative rate
                    gumbel_dist = dist.Gumbel(log_rate, torch.ones_like(log_rate))
                    omega_raw = pyro.sample("omega", gumbel_dist)
                    omega_raw = omega_raw.squeeze(-1) if omega_raw.dim() > 1 else omega_raw
                    omega = torch.exp(-omega_raw)

            # --------------------------
            #  Sample kappa ~ p(kappa | cluster)
            #  (type-anchored intrinsic state; per-cell latent)
            # --------------------------
            with cell_plate:
                C = int(clusters.max().item()) + 1

                counts = torch.bincount(clusters, minlength=C).float().to(self.device)
                w = 1.0 / (counts[clusters] + 1e-8)

                with poutine.scale(scale=w):
                    kappa = pyro.sample(
                        "kappa",
                        dist.Normal(
                            self.kappa_mu(clusters),
                            torch.exp(self.kappa_logvar(clusters) / 2)
                        ).to_event(1)
                    )

            # -----------------------------------------------------
            #  delta = z_cell - kappa  (what neighbors transmit)
            # -----------------------------------------------------
            delta = z_cell - kappa
            delta = delta - delta.mean(dim=0, keepdim=True)
            msg = self._weighted_sum(edge_index, omega, delta)
            s_prime = kappa + msg

        else:
            s_prime = z_cell

        # --------------------------------------
        #  Reconstruct x from p(x | s_prime)
        # --------------------------------------
        with cell_plate:
            mu = torch.softmax(self.decode_x(s_prime), dim=-1)
            x_mu = l * mu
            logits = (x_mu + EPS).log() - (theta + EPS).log()

            nb_dist = dist.NegativeBinomial(total_count=theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)


    def guide(self, data):
        pyro.module("VAE", self)

        x = data[self.ref].x
        u = data[self.query].x
        x = self.lognorm(x)

        # Reshape image patches if paired with histology
        if self.patch_size > 0:
            u = self._reshape_patches(u)

        edge_index_dict = data.edge_index_dict
        edge_index = edge_index_dict[self.r2r]
        src, dst = edge_index

        # q2r edges for unpooling z -> cells
        q2r_src, q2r_dst = edge_index_dict[self.q2r]

        # Plates
        cell_plate = pyro.plate("cell", x.size(0))

        # -------------------------
        #  Sample z ~ q(z | x, u)
        # -------------------------
        with pyro.plate("patch", u.size(0)):
            z_mu, z_logvar, _ = self.encode_z(x, u, edge_index_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar / 2))
            z = pyro.sample("z", z_dist.to_event(1))

        # deterministic unpool z -> z_cell
        z_cell = torch_scatter.scatter_mean(
            z[q2r_src], q2r_dst, dim=0, dim_size=x.size(0)
        )

        if self.configs.infer_cell_interaction:

            clusters = data[self.ref].cluster
            C = int(clusters.max().item()) + 1

            # ----------------------------
            #  Sample kappa ~ q(kappa | x)
            # ----------------------------
            counts = torch.bincount(clusters, minlength=C).float().to(self.device)
            w = 1.0 / (counts[clusters] + 1e-8)

            with cell_plate:
                with poutine.scale(scale=w):
                    kappa_mu, kappa_logvar = self.encode_kappa(x)
                    kappa = pyro.sample(
                        "kappa",
                        dist.Normal(kappa_mu, torch.exp(kappa_logvar / 2)).to_event(1)
                    )

            # -------------------------------
            #  Sample omega_raw ~ q(omega_raw | x, z_cell)
            # -------------------------------
            omega_loc = self.encode_omega(x, z_cell, edge_index_dict)
            omega_loc = omega_loc.squeeze(-1) if omega_loc.dim() > 1 else omega_loc

            scale = 1.0 / (edge_index.size(1) / x.size(0))

            with pyro.plate("r2r_edges", omega_loc.size(0)):
                with poutine.scale(scale=scale):
                    omega_raw = pyro.sample("omega", dist.Delta(omega_loc))
                    omega_raw = omega_raw.squeeze(-1) if omega_raw.dim() > 1 else omega_raw
                    omega = torch.exp(-omega_raw)

            # -----------------------------------------------------
            #  delta = z_cell - kappa   (transmittable component)
            # -----------------------------------------------------
            delta = z_cell - kappa
            delta = delta - delta.mean(dim=0, keepdim=True)   # global centering

            # neighbor message = weighted neighborhood mean(delta)
            msg = self._weighted_sum(edge_index, omega, delta)

            k0 = kappa - kappa.mean(dim=0, keepdim=True)
            m0 = msg   - msg.mean(dim=0, keepdim=True)

            hsic_loss = hsic(m0, k0.detach())  # detach kappa for stability

            pyro.factor(
                "hsic_indep",
                1e-3 * hsic_loss,
                has_rsample=True
            )

            # small l1 on omega for stability (helps large weights with nearby neighbors)
            omega_l1 = omega.mean()  # (E,) -> scalar

            pyro.factor(
                "omega_l1",
                1e-3 * omega_l1, 
                has_rsample=True
            )




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

            edge_index_dict = data.edge_index_dict
            edge_index = edge_index_dict[self.r2r]
            n_edges = edge_index.size(1)

            # q2r edges for unpooling patch z -> cell z
            q2r_src, q2r_dst = edge_index_dict[self.q2r]

            # ---------- p(z | u) ----------
            pz_mu, _ = self.prior(u, edge_index_dict)

            # ---------- q(z | x, u) ----------
            qz_mu, _, _ = self.encode_z(x, u, edge_index_dict)

            # ---------- unpool to cells ----------
            z_cell = torch_scatter.scatter_mean(
                qz_mu[q2r_src], q2r_dst, dim=0, dim_size=x.size(0)
            )

            infer_cci = self.configs.infer_cell_interaction
            if infer_cci:
                # ---------- omega (posterior mean / location) ----------
                q_omega_raw = self.encode_omega(x, z_cell, edge_index_dict)
                q_omega_raw = q_omega_raw.squeeze(-1) if q_omega_raw.dim() > 1 else q_omega_raw
                q_omega = torch.exp(-q_omega_raw)

                # ---------- kappa (posterior mean) ----------
                q_kappa_mu, q_kappa_logvar = self.encode_kappa(x)
                kappa = q_kappa_mu

                clusters = data[self.ref].cluster
                C = int(clusters.max().item()) + 1

                # ---------- delta = z_cell - kappa ----------
                delta = z_cell - kappa
                delta = delta - delta.mean(dim=0, keepdim=True)
                # ---------- msg ----------
                msg = self._weighted_sum(edge_index, q_omega, delta)

                # ---------- intrinsic + extrinsic ----------
                qs = z_cell 

                s_prime = kappa + msg

                mu = torch.softmax(self.decode_x(s_prime), dim=-1)

            else:
                qs = z_cell
                mu = torch.softmax(self.decode_x(qs), dim=-1)
                q_omega = None
                q_omega_raw = None

            px = l * mu

            return ConfigDict({
                "qz": qz_mu,
                "qs": qs,
                "pz": pz_mu,
                "px": px,

                # cell-cell attention (edge weight) 
                "omega": q_omega[:n_edges] if infer_cci else None,
                "omega_logits": q_omega_raw[:n_edges] if infer_cci else None
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
            k=graph_data.k, r=graph_data.r, alpha=self.configs.alpha,
            cluster_key=graph_data.cluster_key,
            num_clusters=graph_data.num_clusters,
            is_weighted=graph_data.is_weighted,
            ref=graph_data.ref, ref_proj_key=graph_data.ref_proj_key,
            query=graph_data.query, query_proj_key=graph_data.query_proj_key,
            is_ref_grid=graph_data.is_ref_grid,
            is_query_grid=graph_data.is_query_grid,
            verbose=False
        )

        dataloader = DataLoader(full_graph_data, shuffle=False)
        qz = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)  # lowres latent 
        qs = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)   # hires latent
        pz = np.zeros_like(qz)
        px = np.zeros((n_cells, n_features), dtype=np.float32)

        # Summarized cell-type specific attention scores
        qomega_scores = np.zeros((n_cells, n_clusters), dtype=np.float32)  

        # assume always one batch
        data = next(iter(dataloader))
        res = self.predict(data, device)

        batch_pz = res.pz.detach().cpu().numpy()
        batch_qz = res.qz.detach().cpu().numpy()
        batch_qs = res.qs.detach().cpu().numpy()
        batch_px = res.px.detach().cpu().numpy()

        query_indices = data[self.query].idx.numpy()
        qz[query_indices] = batch_qz
        pz[query_indices] = batch_pz

        ref_indices = data[self.ref].idx.numpy()
        qs[ref_indices] = batch_qs
        px[ref_indices] = batch_px

        if self.configs.infer_cell_interaction:
            eps = 1e-8
            batch_omega = res.omega.detach().cpu().numpy()  # (E,)

            src, dst = data.edge_index_dict[self.r2r].cpu().numpy()
            clusters = data[self.ref].cluster.cpu().numpy()  # (N_subgraph,)

            cell_indices = data[self.ref].idx[dst].cpu().numpy()  # global target cell ids, (E,)

            # -------------------------------------------------------
            # den[i] = sum_{j->i} omega_{j->i}   (per target cell)
            # -------------------------------------------------------
            den = np.zeros((n_cells,), dtype=np.float32)
            np.add.at(den, cell_indices, batch_omega)

            # -------------------------------------------------------
            # normalized omega per edge (what the model actually uses)
            # -------------------------------------------------------
            omega_norm = batch_omega / (den[cell_indices] + eps)   # (E,)

            # -------------------------------------------------------
            # MEAN aggregation of normalized omega by (cell, source-type)
            # -------------------------------------------------------
            attn_sum = np.zeros((n_cells, n_clusters), dtype=np.float32)
            attn_count = np.zeros((n_cells, n_clusters), dtype=np.int32)

            np.add.at(attn_sum, (cell_indices, clusters[src]), omega_norm)
            np.add.at(attn_count, (cell_indices, clusters[src]), 1)

            qomega_scores = np.divide(
                attn_sum, attn_count,
                out=np.zeros_like(attn_sum),
                where=attn_count != 0
            ).astype(np.float32)

            # ---------
            # Abundance null per target cell (probabilities)
            # ---------
            edge_dist = data[self.r2r].edge_attr
            edge_dist = edge_dist.squeeze(-1) if edge_dist.dim() > 1 else edge_dist
            edge_dist = edge_dist.detach().cpu().numpy()

            alpha = float(self.configs.alpha)
            w_abun = (1.0 + edge_dist).astype(np.float32) ** (-alpha)  # (E,)

            abun_den = np.zeros((n_cells,), dtype=np.float32)
            np.add.at(abun_den, cell_indices, w_abun)

            abun_norm = w_abun / (abun_den[cell_indices] + eps)  # (E,)

            abundance_sum = np.zeros((n_cells, n_clusters), dtype=np.float32)
            abundance_count = np.zeros((n_cells, n_clusters), dtype=np.int32)

            np.add.at(abundance_sum, (cell_indices, clusters[src]), abun_norm)
            np.add.at(abundance_count, (cell_indices, clusters[src]), 1)

            abundance_count = np.divide(
                abundance_sum, abundance_count,
                out=np.zeros_like(abundance_sum),
                where=abundance_count != 0
            ).astype(np.float32)


        adata_query.obsm['X_z'] = qz.astype(np.float32)  # Latent (z) for patches
        adata_ref.obsm['X_z'] = qs.astype(np.float32)  # Latent (z) for cells
    
        # Save edge index & weights for visualization
        if self.configs.infer_cell_interaction:
            adata_ref.obsm['omega'] = qomega_scores  # Attention scores summarized per cell
            adata_ref.obsm['abundance'] = abundance_count   # Cell-type abundance per cell
            adata_ref.uns['omega'] = batch_omega   
            adata_ref.uns['edge_index'] = data.edge_index_dict[self.r2r].cpu().numpy()

        return ConfigDict({
            'qz':           qz,
            'qs':           qs, 
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
    def _weighted_sum(edge_index, edge_weights, x):
        r"""Compute weighted neighboring scores per node"""
        src, dst = edge_index
        N = x.size(0)

        num = torch_scatter.scatter_add(
            edge_weights.unsqueeze(-1) * x[src],
            dst,
            dim=0,
            dim_size=N,         
        )

        den = torch_scatter.scatter_add(
            edge_weights,
            dst,
            dim=0,
            dim_size=N,          
        ).unsqueeze(-1)

        return num / (den + 1e-8)
