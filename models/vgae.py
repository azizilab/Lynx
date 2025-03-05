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

from ml_collections import ConfigDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv
import torch_scatter

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from base_model import BaseModel
from module import Prior
from module import Encoder, XtoZEncoder, XtoVEncoder, XtoOmegaEncoder
from module import Decoder, ZtoOmegaDecoder, ZtoVDecoder
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
            configs.act
        )      

        self.u_to_hidden = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            configs.act
        )      

        self.encode_z = XtoZEncoder(configs)
        self.encode_v = XtoVEncoder(configs)
        self.encode_omega = XtoOmegaEncoder(configs)

        self.decode_omega = ZtoOmegaDecoder(configs)
        self.decode_v = ZtoVDecoder(configs)
        self.v_to_x = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            self.act,
            nn.Linear(configs.c_hidden, configs.c_in),
        )

    def model(self, data):
        pyro.module("VAE", self)

        u = data[self.query].x
        x = data[self.ref].x
        l = x.sum(axis=-1, keepdim=True)

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        clusters = data[self.ref].cluster
        c = self.cluster_embedding(clusters).to(self.device)
        
        theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        # --------------------------
        #  Sample z from p(z | u)
        # --------------------------
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar = self.prior(u)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2))
            z = pyro.sample("z", z_dist.to_event(1))

        # ---------------------------------------------
        #  Sample omega (S_r2r) from p(omega | c, z)
        # ---------------------------------------------
        # TODO: compare lognormal-lognormal vs. weibull-gamma
        # z_unpool, omega_loc, omega_scale = self.decode_omega(z, c, edge_index_dict, edge_attr_dict)
        # with pyro.plate("r2r_edges", omega_loc.size(0)):
        #     omega_ij = pyro.sample(
        #         "omega", 
        #         dist.LogNormal(omega_loc, omega_scale)
        #     )

        z_unpool, omega_alpha, omega_beta = self.decode_omega(z, c, edge_index_dict, edge_attr_dict)
        with pyro.plate("r2r_edges", omega_alpha.size(0)):
            omega_ij = pyro.sample(
                "omega", 
                dist.Gamma(omega_alpha, omega_beta)
            )

        # ------------------------------------
        #  Sample v from p(v | z, c, omega)
        # ------------------------------------
        _, dst = edge_index_dict[self.r2r]  # source & target edges
        W_ij = self.normalize_edges(omega_ij, dst, x.size(0))
        with pyro.plate("hires", x.size(0)):
            v_mu, v_logvar = self.decode_v(z_unpool, W_ij, edge_index_dict)
            v = pyro.sample("v", dist.Normal(v_mu, torch.exp(v_logvar/2)).to_event(1))

            # --------------------------
            #  Sample x from p(x | v)
            # --------------------------
            mu = self.v_to_x(v)  # softmax inside module
            mu = torch.softmax(mu, dim=-1)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - theta.log()

            nb_dist = dist.NegativeBinomial(total_count=theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def guide(self, data):
        pyro.module("VAE", self)

        x = data[self.ref].x    # [num_hires, in_dim]
        u = data[self.query].x  # [num_lowres, aux_dim]
        x = self.lognorm(x)  

        # Project observations to a joint hidden-dim
        x = self.x_to_hidden(x)
        u = self.u_to_hidden(u)

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        # ------------------------------
        #  Sample z from q(z | x, u)
        # ------------------------------
        with pyro.plate("lowres", u.size(0)):
            z_mu, z_logvar, _ = self.encode_z(x, u, edge_index_dict, edge_attr_dict)
            z_dist = dist.Normal(z_mu, torch.exp(z_logvar/2)).to_event(1)
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("z", z_dist)

        # --------------------------
        #  Sample v from q(v | x)
        # --------------------------
        with pyro.plate("hires", x.size(0)):
            v_mu, v_logvar = self.encode_v(x)
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("v", dist.Normal(v_mu, torch.exp(v_logvar / 2)).to_event(1))

        # ----------------------------------
        #  Sample omega from q(omega | x)
        # ----------------------------------
        # TODO: compare lognormal-lognormal vs. weibull-gamma
        # omega_loc, omega_scale = self.encode_omega(x, edge_index_dict, edge_attr_dict)
        # with pyro.plate("r2r_edges", omega_loc.size(0)):
        #     with poutine.scale(scale=self.configs.beta):
        #         pyro.sample("omega", dist.LogNormal(omega_loc, omega_scale))

        omega_lambda, omega_k = self.encode_omega(x, edge_index_dict, edge_attr_dict)
        with pyro.plate("r2r_edges", omega_lambda.size(0)):
            with poutine.scale(scale=self.configs.beta):
                pyro.sample("omega", dist.Weibull(omega_lambda, omega_k))

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
            
            edge_index_dict = data.edge_index_dict
            edge_attr_dict = data.edge_attr_dict
            _, dst= edge_index_dict[self.r2r]

            # ---------- z from p(z | u ) -----------
            pz, _ = self.prior(data[self.query].x)  # e.g. shape [num_lowres, latent_dim]

            # ---------- z from q(z | x, u) -----------
            x = self.x_to_hidden(x)
            u = self.u_to_hidden(u)

            qz, _, attn_score = self.encode_z(x, u, edge_index_dict, edge_attr_dict)

            # ---------- v from q(v | x) ----------
            qv, _ = self.encode_v(x)
            
            # ---------- omega from q(omega | x) ----------
            # TODO: compare lognormal-lognormal vs. gamma-weibull
            # omega_loc, omega_scale = self.encode_omega(x, edge_index_dict, edge_attr_dict)
            # omega_mean = torch.exp(omega_loc + 0.5*(omega_scale**2))
            omega_lambda, omega_k = self.encode_omega(x, edge_index_dict, edge_attr_dict)
            omega_mean = omega_lambda * torch.special.digamma(1+1/omega_k).exp()

            qa = self.normalize_edges(omega_mean, dst, x.size(0))

            # ---------- Reconstruct x from p(x | v) -----------
            mu = self.v_to_x(qv)
            mu = torch.softmax(mu, dim=-1)
            px = l * mu
            

            return ConfigDict({
                "qz": qz,
                "pz": pz,
                "qa": (edge_index_dict[self.r2r], qa),              
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
            batch_v_edges = res.qa[0].detach().cpu().numpy().T  # dim: [edges, 2]
            batch_v_attn = res.qa[1].detach().cpu().numpy()    # dim: [edges, 1]

            v_attn_sum = np.zeros((n_cells, num_clusters), dtype=np.float32)

            ref_idx = data[self.ref].idx[batch_v_edges[:, 1]]
            clusters = data[self.ref].cluster[batch_v_edges[:, 0]]

            np.add.at(v_attn_sum, (ref_idx, clusters), batch_v_attn.squeeze())

            # Min-Max normalization per cluster (column)
            v_attn = v_attn_sum.copy()
            v_attn_min = v_attn.min(axis=0, keepdims=True)
            v_attn_max = v_attn.max(axis=0, keepdims=True)

            # Avoid division by zero
            assert(np.all((v_attn_max - v_attn_min) != 0))
            v_attn = (v_attn - v_attn_min) / (v_attn_max - v_attn_min)

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
                qzx_attention_sum[ref_idx] += a   # [N]
                qzx_attention_counter[ref_idx] += 1

        # Average highres latent representations
        valid = qzx_attention_sum > 0
        qzx[valid.squeeze()] = qzx_weighted_sum[valid.squeeze()] / qzx_attention_sum[valid.squeeze(), None]
        attn[valid.squeeze()] = attn[valid.squeeze()] / qzx_attention_counter[valid.squeeze()]

        # In-place storage to adatas
        adata_ref.obsm['X_z'] = qzx
        adata_ref.obsm['X_v'] = qv
        adata_ref.obsm['v_attn'] = v_attn
        adata_query.obsm['X_z'] = qzu

        return ConfigDict({
            'qv':           qv,
            'qzu':          qzu,
            'qzx':          qzx, 
            'pz':           pz,
            'px':           px,
            'attn':         attn,
            'v_attn':       v_attn
        })
    