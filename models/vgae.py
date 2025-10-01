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
from module import Encoder, XtoZEncoder, ConvXtoZEncoder, XtoVEncoder, XtoOmegaEncoder, XtoOmegaCluEncoder
from module import Decoder, ZtoSDecoder, ZtoXDecoder
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

        # Whether to use conv. prior / posterior for `u` (i.e. histology patches)
        self.patch_size = configs.patch_size if hasattr(configs, 'patch_size') else -1 

        self.cluster_embedding = nn.Embedding(configs.num_clusters, configs.c_latent)
        self.num_clusters = configs.num_clusters

        self.prior = StructuralPrior(configs) if self.patch_size < 0 else ConvPrior(configs)
        self.encode_z = XtoZEncoder(configs) if self.patch_size < 0 else ConvXtoZEncoder(configs)
        self.encode_v = XtoVEncoder(configs)
        self.encode_omega = XtoOmegaEncoder(configs)
        self.decode_omega = ZtoSDecoder(configs)        
        self.decode_x = ZtoXDecoder(configs)

    def model(self, data):
        pyro.module("VAE", self)

        u = data[self.query].x
        x = data[self.ref].x
        l = x.sum(axis=-1, keepdim=True)

        # Reshape image patches if paired with histology
        if self.patch_size > 0:
            u = self._reshape_patches(u)

        # Sample gene-specific dispersion
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        
        # --------------------------
        #  Sample z from p(z | u)
        # --------------------------
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar = self.prior(u, edge_index_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))
            
        # ---------------------------------------------
        #  Sample omega (S_r2r) from p(omega | c, z)
        # ---------------------------------------------
        clusters = data[self.ref].cluster
        c = self.cluster_embedding(clusters).to(self.device)
        
        s, omega_loc, omega_scale = self.decode_omega(z, c, edge_index_dict, edge_attr_dict)
        with pyro.plate("r2r_edges", omega_loc.size(0)):
            omega_ij = pyro.sample(
                "omega", 
                dist.LogNormal(omega_loc, omega_scale)
            )

        # ------------------------------------
        #  Sample v from p(v | z, c, omega)
        # ------------------------------------
        _, dst = edge_index_dict[self.r2r]  # source & target edges
        W_ij = self.normalize_edges(omega_ij, dst, x.size(0))
        mu = self.decode_x(s, W_ij, edge_index_dict)
        x_mu = l * mu
        logits = logits = (x_mu+EPS).log() - theta.log()

        with pyro.plate("hires", x.size(0)):
            nb_dist = dist.NegativeBinomial(total_count=theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, data):
        pyro.module("VAE", self)

        x = data[self.ref].x    # [num_hires, in_dim]
        u = data[self.query].x  # [num_lowres, aux_dim]
        x = self.lognorm(x)  
        # x = torch.log1p(x)

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        # Reshape image patches if paired with histology
        if self.patch_size > 0:
            u = self._reshape_patches(u)

        # ------------------------------
        #  Sample z from q(z | x, u)
        # ------------------------------
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar, _ = self.encode_z(x, u, edge_index_dict, edge_attr_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("z", z_dist.to_event(1))

        # ----------------------------------
        #  Sample omega from q(omega | x)
        # ----------------------------------
        omega_loc, omega_scale = self.encode_omega(x, edge_index_dict, edge_attr_dict)        
        with pyro.plate("r2r_edges", omega_loc.size(0)):
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("omega", dist.LogNormal(omega_loc, omega_scale))

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
            c = self.cluster_embedding(data[self.ref].cluster).to(device)
            
            edge_index_dict = data.edge_index_dict
            edge_attr_dict = data.edge_attr_dict
            _, dst= edge_index_dict[self.r2r]

            # Reshape image patches if paired with histology
            if self.patch_size > 0:
                u = self._reshape_patches(u)


            # ---------- z from p(z | u ) -----------
            pz, _ = self.prior(u, edge_index_dict)

            # ---------- z from q(z | x, u) -----------
            qz, _, attn_score = self.encode_z(x, u, edge_index_dict, edge_attr_dict)
            
            # ---------- omega from q(\omega | x) ----------
            omega_loc, omega_scale = self.encode_omega(x, edge_index_dict, edge_attr_dict)
            omega_mean = torch.exp(omega_loc + 0.5*(omega_scale**2))
            W_ij = self.normalize_edges(omega_mean, dst, x.size(0))

            # ---------- Reconstruct x from p(x | s, c, \omega)
            s, _, _ = self.decode_omega(qz, c, edge_index_dict, edge_attr_dict)
            mu = self.decode_x(s, W_ij, edge_index_dict)
            px = l * mu

            return ConfigDict({
                "qz": qz,
                "pz": pz,
                # "qa": (edge_index_dict[self.r2r], W_ij),                               
                "px": px, 
                "attn_score": attn_score
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

        full_graph_data = HeteroDataset(
            adatas_ref=adata_ref, 
            adatas_query=adata_query, 
            n_subgraphs=n_subgraphs,
            k=graph_data.k, r=graph_data.r, 
            is_weighted=graph_data.is_weighted, use_radius=graph_data.use_radius,
            ref=graph_data.ref, ref_proj_key=graph_data.ref_proj_key,
            query=graph_data.query, query_proj_key=graph_data.query_proj_key,
            verbose=False
        )

        dataloader = DataLoader(full_graph_data, shuffle=False)
        qzu = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)    # lowres latent
        qzx = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)   # hires latent x
        pz = np.zeros_like(qzu)
        px = np.zeros((n_cells, n_features), dtype=np.float32)

        # # Temporary accumulators for weighted averages
        qzx_weighted_sum = np.zeros_like(qzx)
        qzx_attention_sum = np.zeros((n_cells), dtype=np.float32)
        qzx_attention_counter = np.zeros((n_cells), dtype=np.float32)

        # Recover batched predictions in correct spatial orders
        for data in dataloader:
            res = self.predict(data, device)

            batch_qzu = res.qz.detach().cpu().numpy()  # dim: [L, K]
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()
            batch_edges = res.attn_score[0].detach().cpu().numpy().T  # dim: [edges, 2]
            batch_attn = res.attn_score[1].detach().cpu().numpy()    # dim: [edges, 1]

            #################
            query_indices = data[self.query].idx.numpy()
            qzu[query_indices] = batch_qzu
            pz[query_indices] = batch_pz

            ref_indices = data[self.ref].idx.numpy()
            px[ref_indices] = batch_px

            # Compute highres latent representations via attention assignments
            for edge, a in zip(batch_edges, batch_attn):
                ref_idx = data[self.ref].idx[edge[0]]
                
                # Update accumulators for highres
                qzx_weighted_sum[ref_idx] += a * batch_qzu[edge[1]]  # [N, latent_dim]
                qzx_attention_sum[ref_idx] += a   # [N]
                qzx_attention_counter[ref_idx] += 1

        # Average highres latent representations
        valid = qzx_attention_sum > 0
        qzx[valid.squeeze()] = qzx_weighted_sum[valid.squeeze()] / qzx_attention_sum[valid.squeeze(), None]

        # In-place storage to adatas
        adata_query.obsm['X_z'] = qzu
        adata_ref.obsm['X_z'] = qzx

        return ConfigDict({
            'qzu':          qzu,
            'qzx':          qzx, 
            'pz':           pz,
            'px':           px,
        })

    def _reshape_patches(self, u):
        """Reshape flattened patches to proper image format"""
        batch_size = u.shape[0]
        expected_size = 3 * self.patch_size * self.patch_size
        if u.shape[1] != expected_size:
            raise ValueError(f"Expected flattened patch size {expected_size}, got {u.shape[1]}")
        u_reshaped = u.view(batch_size, 3, self.patch_size, self.patch_size)
        return u_reshaped


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
        self.clu_encoder = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            nn.LayerNorm(configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, configs.c_latent * 2)
        )
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

        # --- Global parameters ---

        # \pi: cluster "bulk" weights
        pi = pyro.param(
            "pi",
            self.configs.clu_weight*torch.ones(self.configs.num_clusters, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        # \theta: gene-specific dispersion
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        # --- Sparse edge-edge weight priors ---
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        edge_index = edge_index_dict[self.r2r]
        src, dst = edge_index
        d_edge     = edge_attr_dict[self.r2r]
        clusters = data[self.ref].cluster
  
        abundances = data[self.ref].abundance
        src_clusters = clusters[src]    

        # TODO: [DEBUG] Gamma prior pre-softmax
        # alpha = torch.ones_like(d_edge)
        # beta  = self.configs.base_sparsity + d_edge + abundances[src_clusters]*self.configs.abundance_penalization
        concentration = self.configs.base_sparsity
        rate = self.configs.base_sparsity + d_edge + self.configs.abundance_penalization*abundances[src_clusters]

        # ----------------------------------
        #  Sample omega from p(c)
        # ----------------------------------
        with pyro.plate("clusters", self.configs.num_clusters):
            # cluster embeddings
            clu_emb = pyro.sample("clu_emb",
                dist.Normal(
                    torch.zeros(self.configs.c_latent, device=self.device),
                    torch.ones (self.configs.c_latent, device=self.device)).to_event(1)
            )  # (C, c_latent)
        
        # --------------------------
        #  Sample z from p(z | u)
        # --------------------------
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar = self.prior(u, edge_index_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))
            
        # ---------------------------------------------------------
        #  Sample high-res embedding s from p(s | z);
        #  Sample edge weights omega from p(omega ; alpha, beta)
        # ---------------------------------------------------------
        s = self.decode_s(z, x.size(0), edge_index_dict, edge_attr_dict, only_s=True)
        # TODO: [DEBUG] Gamma prior pre-softmax
        with pyro.plate("r2r_edges", edge_index.size(1)):
            # omega = pyro.sample("omega", dist.Beta(alpha, beta))  # (E,)
            omega = pyro.sample("omega", dist.Gamma(concentration, rate))  # (E,)

        # --------------------------------------
        #  Reconstruct x from p(x | s, omega, pi)
        # --------------------------------------
        with pyro.plate("cells", x.size(0)):
            omega_normed = torch_scatter.scatter_softmax(omega, dst)
            neighbor_effect = self._weighted_sum(edge_index, omega_normed, s)
            pyro.deterministic("neigh_eff", neighbor_effect)
            clu_effect = torch.einsum("x,xy->xy", pi[clusters], clu_emb[clusters])
            pyro.deterministic("clu_eff", clu_effect)
            v = neighbor_effect + clu_effect  # (N, c_latent)
            pyro.deterministic("v", v)
        
        mu = torch.softmax(self.decode_x(v), dim=-1)
        x_mu = l * mu
        logits = logits = (x_mu+EPS).log() - (theta+EPS).log()

        with pyro.plate("hires", x.size(0)):
            nb_dist = dist.NegativeBinomial(total_count=theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, data):
        pyro.module("VAE", self)

        x = data[self.ref].x    # [num_hires, in_dim]
        u = data[self.query].x  # [num_lowres, aux_dim]
        x = self.lognorm(x)  

        # Reshape image patches if paired with histology
        if self.patch_size > 0:
            u = self._reshape_patches(u)

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        # -----------------------------------------------
        #  Sample cluster embedding c from q(c | x^hat)
        # -----------------------------------------------
        bulk_clu = torch.log1p(data[self.ref].bulk_clu)
        with pyro.plate("clusters", self.configs.num_clusters):
            clu_emb_loc, _ = self.clu_encoder(bulk_clu).chunk(2, dim=-1)  # (C, c_latent*2)
            pyro.sample("clu_emb", dist.Delta(clu_emb_loc).to_event(1))

        # ------------------------------
        #  Sample z from q(z | x, u)
        # ------------------------------
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar, _ = self.encode_z(x, u, edge_index_dict, edge_attr_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("z", z_dist.to_event(1))

        # ----------------------------------
        #  Sample omega from q(omega | x)
        # ----------------------------------
        # TODO: [DEBUG] use Gamma instead of Delta
        q_omega = self.encode_omega(x, edge_index_dict, edge_attr_dict)  
        with pyro.plate("r2r_edges", q_omega.size(0)):
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("omega", dist.Delta(q_omega))  # (E,)
                
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
            bulk_clu = torch.log1p(data[self.ref].bulk_clu[clusters])            
            edge_index_dict = data.edge_index_dict
            edge_attr_dict = data.edge_attr_dict
            _, dst = edge_index_dict[self.r2r]

            # Reshape image patches if paired with histology
            if self.patch_size > 0:
                u = self._reshape_patches(u)

            # ---------- p(z | u) ------------
            pz, _ = self.prior(u, edge_index_dict)

            # ---------- q(z | x, u) -----------
            qz, _, attn_score = self.encode_z(x, u, edge_index_dict, edge_attr_dict)
            
            # ---------- q(\omega | x) ---------
            q_omega = self.encode_omega(x, edge_index_dict, edge_attr_dict) 
            q_omega_normed = torch_scatter.scatter_softmax(q_omega, dst)
            s = self.decode_s(qz, x.size(0), edge_index_dict, edge_attr_dict, only_s=True)
            neighbor_effect = self._weighted_sum(edge_index_dict[self.r2r], q_omega_normed, s)

            pi = pyro.param("pi").to(device)
            kappa, _ = self.clu_encoder(bulk_clu[clusters]).chunk(2, dim=-1)
            clu_effect = torch.einsum("x,xy->xy", pi[clusters], kappa)
            v = neighbor_effect + clu_effect  # (N, c_latent)

            # ---------- Reconstruct x from p(x | s, \omega)
            mu = torch.softmax(self.decode_x(v), dim=-1)
            px = l * mu
    
            return ConfigDict({
                "qz": qz,
                "pz": pz,
                "omega": q_omega,  # cell-cell attn edge weights            
                "px": px, 
                "attn_score": attn_score,  # cell-patch attn weights
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

        full_graph_data = HeteroDataset(
            adatas_ref=adata_ref, 
            adatas_query=adata_query, 
            n_subgraphs=n_subgraphs,
            k=graph_data.k, r=graph_data.r, 
            is_weighted=graph_data.is_weighted, use_radius=graph_data.use_radius,
            ref=graph_data.ref, ref_proj_key=graph_data.ref_proj_key,
            query=graph_data.query, query_proj_key=graph_data.query_proj_key,
            verbose=False
        )

        dataloader = DataLoader(full_graph_data, shuffle=False)
        qzu = np.zeros((n_pixels, self.configs.c_latent), dtype=np.float32)    # lowres latent
        qzx = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)   # hires latent x
        pz = np.zeros_like(qzu)
        px = np.zeros((n_cells, n_features), dtype=np.float32)

        # # Temporary accumulators for weighted averages
        qzx_weighted_sum = np.zeros_like(qzx)
        qzx_attention_sum = np.zeros((n_cells), dtype=np.float32)
        qzx_attention_counter = np.zeros((n_cells), dtype=np.float32)
         
        # Retrive inference Attn scores
        qomega_scores = np.zeros((n_cells, adata_ref.obs.leiden.max()+1), dtype=np.float32)

        # Recover batched predictions in correct spatial orders
        for data in dataloader:
            res = self.predict(data, device)

            batch_qzu = res.qz.detach().cpu().numpy()  # dim: [L, K]
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()
            batch_edges = res.attn_score[0].detach().cpu().numpy().T  # dim: [E, 2]
            batch_attn = res.attn_score[1].detach().cpu().numpy()    # dim: [E, 1]
            batch_omega = res.omega.detach().cpu().numpy() # dim: [E]
  
            # Cell-type specific attention scores
            ref_idx = data[self.ref].idx
            assert np.all(qomega_scores[ref_idx] == 0)  # making sure cells are called twice

            src, dst = data.edge_index_dict[self.r2r].cpu().numpy()
            clusters = data[self.ref].cluster.cpu().numpy()  # [N]
            cluster_edges = clusters[src]  # [E]
            np.add.at(qomega_scores, (data[self.ref].idx[dst], cluster_edges), batch_omega)

            query_indices = data[self.query].idx.numpy()
            qzu[query_indices] = batch_qzu
            pz[query_indices] = batch_pz

            ref_indices = data[self.ref].idx.numpy()
            px[ref_indices] = batch_px

            # Compute highres latent representations via attention assignments
            for edge, a in zip(batch_edges, batch_attn):
                ref_idx = data[self.ref].idx[edge[0]]
                
                # Update accumulators for highres
                qzx_weighted_sum[ref_idx] += a * batch_qzu[edge[1]]  # [N, latent_dim]
                qzx_attention_sum[ref_idx] += a   # [N]
                qzx_attention_counter[ref_idx] += 1

        # Average highres latent representations
        valid = qzx_attention_sum > 0
        qzx[valid.squeeze()] = qzx_weighted_sum[valid.squeeze()] / qzx_attention_sum[valid.squeeze(), None]

        # In-place storage to adatas
        adata_query.obsm['X_z'] = qzu
        adata_ref.obsm['X_z'] = qzx
        adata_ref.obsm['omega'] = qomega_scores

        return ConfigDict({
            'qzu':          qzu,
            'qzx':          qzx, 
            'pz':           pz,
            'px':           px,
        })

    def _weighted_sum(self, edge_index, omega, z):
        """Compute weighted neighboring scores per node"""
        src, dst = edge_index
        neighbor_contrib = torch_scatter.scatter_add(omega.unsqueeze(-1)*z[src], dst, dim=0, dim_size=z.size(0))	
        return neighbor_contrib

    def _reshape_patches(self, u):
        """Reshape flattened patches to proper image format"""
        batch_size = u.shape[0]
        expected_size = 3 * self.patch_size * self.patch_size
        if u.shape[1] != expected_size:
            raise ValueError(f"Expected flattened patch size {expected_size}, got {u.shape[1]}")
        u_reshaped = u.view(batch_size, 3, self.patch_size, self.patch_size)
        return u_reshaped
