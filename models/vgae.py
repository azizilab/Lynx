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
from torch_geometric.utils import softmax as edge_softmax
import torch_scatter

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from base_model import BaseModel
from module import Prior, StructuralPrior, ConvPrior, VolumePrior
from module import Encoder, XtoZEncoder, ConvXtoZEncoder, XtoVEncoder, XtoOmegaCluEncoder, XtoKappaEncoder
from module import Decoder, ZtoSDecoder, StoXDecoder
from module import hsic
from dataset import XeniumDataset, HeteroDataset

EPS = 1e-8


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
        if configs.get('volume', None) is not None:  # 3D prior over the registered stack of the auxiliary modality
            self.prior = VolumePrior(configs)
        self.encode_z = XtoZEncoder(configs) if self.patch_size < 0 else ConvXtoZEncoder(configs)
        self.encode_kappa = XtoKappaEncoder(configs)
        self.encode_omega = XtoOmegaCluEncoder(configs)
        if configs.niche_path:
            self.niche_proj = nn.Linear(configs.c_latent, configs.c_latent, bias=False)
        self.kappa_mu = nn.Embedding(configs.n_cluster, configs.c_latent)
        self.kappa_logvar = nn.Embedding(configs.n_cluster, configs.c_latent)

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
            z_mu, z_logvar = self._prior(data, u)
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
            # ---------------------------------------------
            # log_d = torch.log1p(edge_distances)
            alpha = self.configs.alpha
            scale = 1.0 / (edge_index.size(1) / x.size(0))

            with pyro.plate("r2r_edges", edge_distances.size(0)):
                with poutine.scale(scale=scale):
                    if self.configs.omega_prior == 'exponential':
                        exponential_dist = dist.Exponential(
                            (1+edge_distances).pow(alpha)
                        )
                        omega = pyro.sample("omega", exponential_dist).squeeze(-1)
                    else:  # log-normal: w_ij ∝ (1+d_ij)^-alpha exp(g_ij)
                        g = pyro.sample("omega_logit", dist.Normal(
                            torch.zeros_like(edge_distances), self.configs.omega_sigma))
                        omega = self._edge_weights(g, edge_distances, dst, x.size(0))

            # --------------------------
            #  Sample kappa ~ p(kappa | cluster)
            #  (type-anchored intrinsic state; per-cell latent)
            # --------------------------
            with cell_plate:
                w = self._kappa_scale(clusters)

                kappa_mu = self.kappa_mu(clusters)
                kappa_std = torch.ones(self.configs.c_latent, dtype=torch.float).to(self.device)
                with poutine.scale(scale=w):
                    kappa = pyro.sample(
                        "kappa",
                        dist.Normal(kappa_mu, kappa_std).to_event(1)
                    )

            # -----------------------------------------------------
            #  delta = z_cell - kappa  (what neighbors transmit)
            # -----------------------------------------------------
            delta = z_cell - kappa
            delta = delta - delta.mean(dim=0, keepdim=True)
            msg = self._weighted_sum(edge_index, omega, delta)
            s_prime = kappa + msg if not self.configs.niche_path else kappa + self.niche_proj(z_cell) + msg

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
            solve = self.configs.kappa_guide == 'posterior_solve'

            # ----------------------------
            #  Sample kappa ~ q(kappa | x)
            # ----------------------------
            w = self._kappa_scale(clusters)

            with cell_plate:
                with poutine.scale(scale=w):
                    kappa_mu, kappa_logvar = self.encode_kappa(x)
                    if not solve:
                        kappa = pyro.sample(
                            "kappa",
                            dist.Normal(kappa_mu, torch.exp(kappa_logvar / 2)).to_event(1)
                        )

            # -------------------------------
            #  Sample omega_raw ~ q(omega_raw | x, z_cell)
            # -------------------------------
            omega = self._guide_omega(x, z_cell, edge_index_dict, data.edge_attr_dict[self.r2r], sample=True)

            # posterior_solve: q(kappa) at the posterior mean of kappa given the encoded state (kappa_mu)
            if solve:
                kappa_loc = self._kappa_solve(edge_index, omega, z_cell, kappa_mu, clusters)
                with cell_plate:
                    with poutine.scale(scale=w):
                        kappa = pyro.sample(
                            "kappa",
                            dist.Normal(kappa_loc, torch.exp(kappa_logvar / 2)).to_event(1)
                        )

            # Regularization w/ HSIC btw kappa & m (original model; off by default)
            if self.configs.hsic_weight > 0:
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
                    self.configs.hsic_weight * hsic_loss,
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
            pz_mu, _ = self._prior(data, u)

            # ---------- q(z | x, u) ----------
            qz_mu, _, _ = self.encode_z(x, u, edge_index_dict)

            # ---------- unpool to cells ----------
            z_cell = torch_scatter.scatter_mean(
                qz_mu[q2r_src], q2r_dst, dim=0, dim_size=x.size(0)
            )

            infer_cci = self.configs.infer_cell_interaction
            if infer_cci:
                # ---------- omega (posterior mean / location) ----------
                q_omega = self._guide_omega(x, z_cell, edge_index_dict, data.edge_attr_dict[self.r2r])

                # ---------- kappa (posterior mean) ----------
                q_kappa_mu, _ = self.encode_kappa(x)
                clusters = data[self.ref].cluster
                kappa = q_kappa_mu if self.configs.kappa_guide == 'amortized' else \
                    self._kappa_solve(edge_index, q_omega, z_cell, q_kappa_mu, clusters)

                # ---------- delta = z_cell - kappa ----------
                delta = z_cell - kappa
                delta = delta - delta.mean(dim=0, keepdim=True)
                # ---------- msg ----------
                msg = self._weighted_sum(edge_index, q_omega, delta)

                # ---------- intrinsic + niche + extrinsic ----------
                qs = z_cell 
                s_prime = kappa + self._niche(z_cell) + msg
                mu = torch.softmax(self.decode_x(s_prime), dim=-1)

            else:
                qs = z_cell
                s_prime = kappa = msg = q_kappa_mu = None
                mu = torch.softmax(self.decode_x(qs), dim=-1)
                q_omega = None

            px = l * mu
            z_prior = torch_scatter.scatter_mean(pz_mu[q2r_src], q2r_dst, dim=0, dim_size=x.size(0))

            return ConfigDict({
                "qz": qz_mu,
                "qs": qs,
                "pz": pz_mu,
                "px": px,
                "omega": q_omega,
                "z_prior": z_prior,   # prior niche latent per cell (auxiliary modality only)
                "s_hat": q_kappa_mu,  # encoded state
                "kappa": kappa,
                "msg": msg,
                "state": s_prime,
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
        n_clusters = graph_data.num_clusters
        
        full_graph_data = HeteroDataset(
            adatas_ref=adata_ref, 
            adatas_query=adata_query, 
            n_subgraphs=n_subgraphs,
            k=graph_data.k, r=graph_data.r, r_bigraph=graph_data.r_bigraph, alpha=self.configs.alpha,
            cluster_key=graph_data.cluster_key,
            num_clusters=n_clusters,
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

        query_indices = data[self.query].idx.cpu().numpy()
        qz[query_indices] = batch_qz
        pz[query_indices] = batch_pz

        ref_indices = data[self.ref].idx.cpu().numpy()
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

            # State decomposition and what the post-hoc tests read (util/test_assoc.py)
            to_np = lambda v: v.detach().cpu().numpy().astype(np.float32)
            for key in ('z_prior', 's_hat', 'kappa', 'msg'):
                arr = np.zeros((n_cells, self.configs.c_latent), dtype=np.float32)
                arr[ref_indices] = to_np(res[key])
                adata_ref.obsm['X_' + key] = arr
            state = to_np(res.state)
            adata_ref.uns['lynx'] = dict(
                version=self.configs.get('version', 'revised'),
                edge_dist=edge_dist, alpha=alpha, radius=float(graph_data.r), cluster_key=graph_data.cluster_key,
                kappa_mu=to_np(self.kappa_mu.weight),
                jacobian=np.stack([self._decoder_jacobian(state[clusters == c].mean(0)) if (clusters == c).any()
                                   else np.zeros((n_features, self.configs.c_latent), np.float32)
                                   for c in range(n_clusters)]),
            )
            if self.configs.niche_path:
                adata_ref.uns['lynx']['niche_proj'] = to_np(self.niche_proj.weight)

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

    def _prior(self, data, u):
        r"""p(z | u) from the section's own graph, or (3D prior) from the registered volume at each pixel's voxel"""
        if isinstance(self.prior, VolumePrior):
            return self.prior(data[self.query].vox)
        return self.prior(u, data.edge_index_dict)

    def _niche(self, z_cell):
        r"""Direct effect of the cell's own niche latent on its state (niche_path)"""
        return self.niche_proj(z_cell) if self.configs.niche_path else torch.zeros_like(z_cell)

    def _kappa_scale(self, clusters):
        r"""Per-cell weight of the kappa prior: 'balanced' mean n_c / n_c, 'inverse' 1 / n_c"""
        C = int(clusters.max().item()) + 1
        counts = torch.bincount(clusters, minlength=C).float().to(self.device)
        if self.configs.kappa_prior_scale == 'balanced':
            return counts[counts > 0].mean() / counts[clusters]
        return 1.0 / (counts[clusters] + 1e-8)

    def _edge_weights(self, g, edge_distances, dst, n):
        r"""Log-normal edge prior: receiver-normalised w_ij ∝ (1+d_ij)^-alpha exp(g_ij)"""
        d = edge_distances.squeeze(-1) if edge_distances.dim() > 1 else edge_distances
        return edge_softmax(-self.configs.alpha * torch.log1p(d) + g, dst, num_nodes=n)

    def _guide_omega(self, x, z_cell, edge_index_dict, edge_distances, sample=False):
        r"""Edge weights at the guide: the edge network's output (exponential prior) or the log-normal weights at its
        offsets g; `sample` records the guide site"""
        n, exp = x.size(0), self.configs.omega_prior == 'exponential'
        loc = self.encode_omega(x, z_cell, edge_index_dict, positive=exp).squeeze(-1)
        if sample:
            scale = 1.0 / (edge_index_dict[self.r2r].size(1) / n)
            with pyro.plate("r2r_edges", loc.size(0)):
                with poutine.scale(scale=scale):
                    if exp:
                        return pyro.sample("omega", dist.Delta(loc)).squeeze(-1)
                    loc = pyro.sample("omega_logit", dist.Delta(loc))
        return loc if exp else self._edge_weights(loc, edge_distances, edge_index_dict[self.r2r][1], n)

    def _kappa_solve(self, edge_index, omega, z_cell, s_hat, clusters):
        r"""Posterior mean of kappa given the encoded state s_hat, a noisy observation (variance lam) of the state
            s' = kappa + niche(z) + A (z - kappa) - r mean(z - kappa) = L kappa + b,  kappa ~ N(mu_c, I):
            (L^T L / lam + I) kappa = L^T (s_hat - b) / lam + mu_c,
        with A the receiver-normalised edge weights and r their row sums; conjugate gradient from s_hat"""
        src, dst = edge_index
        n = s_hat.size(0)
        a = (omega / (torch_scatter.scatter_add(omega, dst, dim=0, dim_size=n)[dst] + 1e-8)).unsqueeze(-1)
        r = torch_scatter.scatter_add(a, dst, dim=0, dim_size=n)
        A = lambda v: torch_scatter.scatter_add(a * v[src], dst, dim=0, dim_size=n)
        At = lambda v: torch_scatter.scatter_add(a * v[dst], src, dim=0, dim_size=n)
        L = lambda v: v - (A(v) - r * v.mean(0, keepdim=True))
        Lt = lambda u: u - At(u) + (r * u).mean(0, keepdim=True)
        lam, mu = self.configs.kappa_solve_lambda, self.kappa_mu(clusters)
        b = A(z_cell) - r * z_cell.mean(0, keepdim=True)
        if self.configs.niche_path:
            b = b + self.niche_proj(z_cell)
        rhs = Lt(s_hat - b) / lam + mu
        return self._kappa_cg(lambda v: Lt(L(v)) / lam + v, rhs, s_hat, mu, edge_index, n)

    def _kappa_cg(self, H, rhs, x, mu, edge_index, n):
        r"""Conjugate gradient for H kappa = rhs from x (H = I + L^T L / lam, eigenvalues >= 1). With kappa_solve_tol > 0
        it stops at the first step whose true residual certifies ||kappa - kappa*|| <= ||r|| <= tol ||kappa* - mu||
        (checked as ||r|| <= tol / (1 + tol) ||kappa - mu||); kappa_solve_iters caps the steps (0: the tissue bound).
        Counts in kappa_solve_stats"""
        tol = float(self.configs.kappa_solve_tol)

        def cert(res, k):
            with torch.no_grad():
                return bool(res.norm() <= tol / (1 + tol) * (k - mu).norm())

        res = rhs - H(x)
        p, rs, used, certified = res, (res * res).sum(), 0, False
        for _ in range(self._kappa_solve_cap(edge_index, n, tol)):
            if tol > 0 and cert(res, x):
                with torch.no_grad():  # the recursive residual drifts: confirm with the true one
                    certified = cert(rhs - H(x), x)
                if certified:
                    break
            Hp = H(p)
            step = rs / ((p * Hp).sum() + 1e-12)
            x = x + step * p
            res = res - step * Hp
            rs_new = (res * res).sum()
            p = res + (rs_new / (rs + 1e-12)) * p
            rs = rs_new
            used += 1
        if tol > 0 and not certified:
            with torch.no_grad():
                certified = cert(rhs - H(x), x)
        if not hasattr(self, 'kappa_solve_stats'):
            self.kappa_solve_stats = dict(solves=0, steps=0, max_steps=0, uncertified=0)
        st = self.kappa_solve_stats
        st['solves'] += 1; st['steps'] += used; st['max_steps'] = max(st['max_steps'], used)
        st['uncertified'] += int(tol > 0 and not certified)
        return x

    def _kappa_solve_cap(self, edge_index, n, tol):
        r"""kappa_solve_iters, or (0) the steps that reach tol of the starting error for any weights on this graph:
        cond(H) <= c = 1 + (2 + sqrt(D))^2 / lam, D the most receivers of one sender, 2 ((sqrt(c)-1)/(sqrt(c)+1))^k <= tol"""
        if self.configs.kappa_solve_iters > 0:
            return int(self.configs.kappa_solve_iters)
        assert tol > 0, "kappa_solve_iters = 0 (the tissue bound) needs kappa_solve_tol > 0"
        D = int(torch.bincount(edge_index[0], minlength=n).max()) if edge_index.numel() else 0
        c = 1 + (2 + np.sqrt(D)) ** 2 / self.configs.kappa_solve_lambda
        rho = (np.sqrt(c) - 1) / (np.sqrt(c) + 1)
        return max(1, int(np.ceil(np.log(2 / tol) / -np.log(rho))))

    def _decoder_jacobian(self, s0):
        r"""Jacobian (G, D) of the decoded log-proportions at state s0, centred over genes"""
        s0 = torch.as_tensor(s0, dtype=torch.float32, device=self.device)
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(lambda s: torch.log_softmax(self.decode_x(s[None]), -1)[0], s0)
        return (J - J.mean(0, keepdim=True)).detach().cpu().numpy().astype(np.float32)

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
