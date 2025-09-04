# scKI_clean.py ─────────────────────────────────────────────────────────────
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, AdamW
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add, scatter_softmax, scatter
from torch_geometric.data import Data, DataLoader
import pyro.poutine as poutine
from tqdm import tqdm
from sklearn.metrics import r2_score

from custom_dist import TruncatedNormal

EPS = 1e-6
MAX_SCALE = 30.0          # 🔹 cap std-devs used in Normal


# ───────────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    c_in: int
    num_clusters: int
    # architecture
    c_hidden: int = 16
    c_latent: int = 4
    c_spatial: int = 32
    act: Callable = nn.ReLU
    base_prob: int = 2    
    spread_prob: int = 1
    hsic_weight: float = 1e3
    entropy_weight: float = 1e-3
    cluster_penalty: float = 1e2
    # shrinkage hyper-priors

    # optimisation
    lr: float = 1e-3
    n_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



# ───────────────────────────────────────────────────────────────────────────
# 3.  Main model
# ───────────────────────────────────────────────────────────────────────────
class scKI(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        self.tau_hi    = 5.0
        self.tau_lo    = 0.1
        self.alpha = 1.0
        self.beta  = 10.0
        self.use_gate = True
        self.warmup = 5

        # enc/dec --------------------------------------------------------
        self.encoder_head = nn.Sequential(
            nn.Linear(cfg.c_in, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_latent * 2)  # mean + std-dev
        )
        # self.v_decoder = nn.Sequential(
        #     nn.Linear(cfg.c_latent, cfg.c_spatial),
        # )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.c_latent, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_in)
        )
        # self.z_encoder = GATConv(
        #     in_channels = cfg.c_hidden,
        #     out_channels= cfg.c_hidden,
        #     heads       = 1,
        #     concat      = False,
        #     add_self_loops=False, 
        #     residual    = False,      
        #     edge_dim=1   
        #    )   
        self.x_encoder = nn.Sequential(
            nn.Linear(cfg.c_in, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_hidden)
        )
        self.clu_encoder = nn.Sequential(
            nn.Linear(cfg.c_in, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_hidden),
            nn.LayerNorm(cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_latent * 2)  # mean + std-dev
        )
        self.clu_mean = nn.Parameter(torch.zeros(cfg.num_clusters, cfg.c_latent))
        self.clu_logstd = nn.Parameter(torch.zeros(cfg.num_clusters, cfg.c_latent))
        
        # self.z_encoder = nn.Sequential(
        #     nn.Linear(cfg.c_in, cfg.c_hidden),
        #     cfg.act,
        #     nn.Linear(cfg.c_hidden, cfg.c_hidden),
        #     cfg.act,
        #     nn.Linear(cfg.c_hidden, cfg.c_latent*2)
        # )
        
        # self.z_encoder = nn.Sequential(
        #     nn.Linear(cfg.c_spatial, cfg.c_spatial),
        #     cfg.act,
        #     nn.Linear(cfg.c_spatial, cfg.c_latent * 2)
        # )
            
        # self.clu_emb = nn.Embedding(cfg.num_clusters, cfg.c_latent)

        # λ̂ₑ amortiser
        self.pi_mlp = nn.Sequential(
            nn.Linear(cfg.c_hidden + cfg.c_hidden + 1, cfg.c_latent),
            cfg.act,
            nn.Linear(cfg.c_latent, 2),
        )
        self.source_mlp = nn.Sequential(
            nn.Linear(cfg.c_in, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_hidden),
        )
        self.target_mlp = nn.Sequential(
            nn.Linear(cfg.c_in, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_hidden),
        )
        self.target_bulk_mlp = nn.Sequential(
            nn.Linear(cfg.c_in, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_hidden),
        )
        self.bulk_mlp = nn.Sequential(
            nn.Linear(cfg.c_in, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_hidden),
        )
        self.pi_mlp = nn.Sequential(
            nn.Linear(cfg.c_hidden + cfg.c_hidden + 1, cfg.c_hidden),
            cfg.act,
            nn.Linear(cfg.c_hidden, cfg.c_latent),
            cfg.act,
            nn.LayerNorm(cfg.c_latent),
            nn.Linear(cfg.c_latent, 1),
        )
        # self.pi_raw_mlp = nn.Sequential(
        #     nn.Linear(cfg.c_hidden + cfg.c_hidden + 1, cfg.c_hidden),
        #     cfg.act,
        #     nn.Linear(cfg.c_hidden, cfg.c_latent),
        #     cfg.act,
        #     nn.LayerNorm(cfg.c_latent),
        #     nn.Linear(cfg.c_latent, 1),
        # )
        # self.pi_bulk_mlp = nn.Sequential(
        #     nn.Linear(cfg.c_hidden + cfg.c_hidden, cfg.c_hidden),
        #     cfg.act,
        #     nn.Linear(cfg.c_hidden, cfg.c_latent),
        #     cfg.act,
        #     nn.LayerNorm(cfg.c_latent),
        #     nn.Linear(cfg.c_latent, 1),
        # )

        self.init_weights()

        # for m in self.lambda_mlp.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, a=1.)
        #         nn.init.zeros_(m.bias)

        # for m in self.pi_bulk_mlp.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.zeros_(m.bias)
        # for m in self.pi_raw_mlp.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.zeros_(m.bias)

        # init_val = 1.0                       # ~ median-scaled distance
        # nn.init.constant_(self.lambda_mlp[-1].bias,
        #                 init_val)  # inverse of exp
        
        # self.ln_raw = nn.LayerNorm(cfg.c_hidden + cfg.c_hidden + 1)
        # self.ln_bulk = nn.LayerNorm(cfg.c_hidden + cfg.c_hidden)

        # global scale (single scalar, stays in MAP)
        # self.tau_log_c = pyro.param("tau_log_c",
            #                     torch.zeros(cfg.num_clusters))
    def init_weights(self):
        # Xavier for all Linear weights; zero bias
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # π heads last layer: small scale so logits aren’t huge at start
        def _shrink_last_linear(seq):
            # find last Linear in a Sequential and shrink it
            last_lin = None
            for layer in reversed(seq):
                if isinstance(layer, nn.Linear):
                    last_lin = layer; break
            if last_lin is not None:
                with torch.no_grad():
                    last_lin.weight.mul_(0.5)  # shrink
                    if last_lin.bias is not None:
                        last_lin.bias.zero_()

        # _shrink_last_linear(self.pi_raw_mlp)
        # _shrink_last_linear(self.pi_bulk_mlp)
        _shrink_last_linear(self.pi_mlp)

        # Cluster param tensors (if you use them)
        if hasattr(self, "clu_mean"):
            nn.init.normal_(self.clu_mean, mean=0.0, std=0.02)
        if hasattr(self, "clu_logstd"):
            nn.init.constant_(self.clu_logstd, -1.0)  # ~ e^-1 std




    # -------------------------------------------------------------------
    def _rbf_weighted_sum(self, edge_index, d2, lam, z):
        src, dst = edge_index
        w = torch.exp( -d2 / (2.0 * lam ** 2 + EPS) )
        # w = w * gate
        # return scatter_add(w.unsqueeze(-1) * z[src],
        #                    dst, dim=0, dim_size=z.size(0))
        num = scatter_add(w.unsqueeze(-1)*z[src], dst, dim=0, dim_size=z.size(0))			
        den = scatter_add(w, dst, dim=0, dim_size=z.size(0)).unsqueeze(-1)+EPS			
        return num/den # ← degree-normalised
        # return scatter_add(w.unsqueeze(-1) * z[src],
        #                    torch.zeros_like(dst, device=self.device), dim=0, dim_size=z.size(0))
        # num = scatter_add(w.unsqueeze(-1) * z[src],            # (N,d)
        #               dst, dim=0, dim_size=z.size(0))
        # den = scatter_add(w, dst, dim=0, dim_size=z.size(0))   # (N,)
        # return num / (den.unsqueeze(-1) + EPS)
    def _weighted_sum(self, edge_index, pi, z):
        src, dst = edge_index
        neighbor_contrib = scatter_add(pi.unsqueeze(-1)*z[src], dst, dim=0, dim_size=z.size(0))	
        return neighbor_contrib # ← degree-normalised

    # -------------------------------------------------------------------
    def model(self, data):
        pyro.module("scKI", self)
        x          = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        d_edge     = data.edge_attr.squeeze().to(self.device)
        # d2         = d_edge.pow(2) 
        lib        = x.sum(-1, keepdim=True) + EPS

        theta = pyro.param("theta",
            torch.ones(self.cfg.c_in, device=self.device),
            constraint=dist.constraints.positive).to(self.device)  # (D,)
        
        with pyro.plate("clusters", self.cfg.num_clusters, dim=-1):
            # cluster embeddings
            clu_emb = pyro.sample("clu_emb",
                dist.Normal(torch.zeros(self.cfg.c_latent, device=self.device),
                            1.*torch.ones (self.cfg.c_latent, device=self.device)).to_event(1)).to(self.device)  # (C, c_latent)
        

        # λ̃ₑ ------------------------------------------------------------
        with pyro.plate("edges", edge_index.size(1), dim=-1):
             # Beta–Bernoulli gate  ----------------------------
            # alpha = torch.tensor(self.alpha, device=self.device)   # broader prior
            # beta  = torch.tensor(self.beta, device=self.device)
            # tau   = pyro.param("tau_temp",
            #                     torch.tensor(self.tau_hi, device=self.device),
            #                     constraint=dist.constraints.positive)
            # if self.use_gate:
            #     pi = pyro.sample("pi", dist.Beta(alpha, beta)) 
            #     gate = pyro.sample("gate", dist.RelaxedBernoulliStraightThrough(temperature=tau, probs=pi))
            # else:
            #     gate = 1.

            # lam = pyro.sample('lam', dist.Normal(torch.tensor(0., device=self.device),
            #                 torch.tensor (1., device=self.device)))
            # lam = F.relu(lam)
            # lam = pyro.sample('lam', TruncatedNormal(loc=0.0, scale=1., lower=0.))
            # lam = pyro.sample('lam', dist.HalfNormal(torch.tensor(0.1, device=self.device)))  # (E
            prob = pyro.sample('pi', dist.Beta(torch.ones_like(d_edge), self.cfg.base_prob + self.cfg.spread_prob*d_edge))  # (E,)

        # pyro.deterministic("lam_gated", gate*lam)

        # latent plates -------------------------------------------------
        with pyro.plate("cells", x.size(0)):
            z = pyro.sample("z",
                dist.Normal(torch.zeros(self.cfg.c_latent, device=self.device),
                            torch.ones (self.cfg.c_latent, device=self.device)).to_event(1))

            neighbor_effect = self._weighted_sum(edge_index, prob, z)
            clu_weight = pyro.sample("clu_weight", dist.Beta(torch.ones(x.size(0), device=self.device),
                                                              torch.ones(x.size(0), device=self.device)))  # (N,)
            clu_effect = clu_weight.unsqueeze(-1)*clu_emb[data.cluster.to(self.device)]


            pyro.deterministic("neigh_eff", neighbor_effect)
            pyro.deterministic("clu_eff", clu_effect)
            # v_mean = self.v_decoder(v_mean)                          # (N, c_spatial)
            v = neighbor_effect + clu_effect  # (N, c_latent)
            pyro.deterministic("v", v)
            # v_mean = z
            
            # v = pyro.sample("v",
            #     dist.Normal(v_mean, torch.ones_like(v_mean)).to_event(1))

            # ---- Negative Binomial likelihood  ----
            p_x = torch.softmax(self.decoder(v), dim=-1)       # (N,D)
            mu = (lib * p_x).clamp(min=EPS)                       # 🔹 avoid 0
            logits = (mu + EPS).log() - (theta + EPS).log()

            # logits = logits.clamp(min=-20.0, max=20.0)              # 🔹

            pyro.sample(
                "x",
                dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1),
                obs=x
            )

    def ent_loss(self, q_pi: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        ent = -(q_pi * (q_pi.clamp(min=1e-12).log()))   # (E,)
        ent_per_dst = scatter(ent, dst, dim=0, reduce="sum")  # (N_nodes,)
        entropy = ent_per_dst.mean()  # scalar
        return entropy

    def hsic_loss(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        N = X.size(0)
        device = X.device

        # Gram matrices (linear kernels)
        Kx = X @ X.T
        Ky = Y @ Y.T

        # Centering matrix
        H = torch.eye(N, device=device) - (1.0 / N) * torch.ones((N, N), device=device)

        # Centered kernels
        Kx_c = H @ Kx @ H
        Ky_c = H @ Ky @ H

        # Numerator
        num = (Kx_c * Ky_c).sum()

        # Robust denominator
        norm_x = torch.linalg.norm(Kx_c, "fro")
        norm_y = torch.linalg.norm(Ky_c, "fro")
        denom = (norm_x * norm_y).clamp(min=eps)

        return num / denom



    # -------------------------------------------------------------------
    def guide(self, data):
        pyro.module("scKI", self)
        x = data.x.to(self.device)
        clusters = data.cluster.to(self.device)

        l = x.sum(axis=-1, keepdim=True) + EPS
        x = x / l * l.median() 
        x = torch.log1p(x)  

        bulk_clu = data.bulk_clu.to(self.device)
    
        edge_index = data.edge_index.to(self.device)
        edge_attr = torch.log1p(data.edge_attr.squeeze().to(self.device))
        src, dst   = edge_index

        with pyro.plate("clusters", self.cfg.num_clusters, dim=-1):

            clu_qmu, clu_qsigma = self.clu_encoder(bulk_clu).chunk(2, dim=-1)  # (C, c_latent*2)
            clu_qsigma = torch.exp(clu_qsigma).clamp(max=MAX_SCALE) + EPS  # 🔹 avoid 0

            # q_clu_std = torch.exp(self.clu_logstd).clamp(max=MAX_SCALE) + EPS
            # print(self.clu_mean.min(), self.clu_mean.max(), q_clu_std.min(), q_clu_std.max())
            clus = pyro.sample("clu_emb",
                dist.Normal(clu_qmu, clu_qsigma).to_event(1))

        #raw feat
        edge_feat = torch.cat([self.target_mlp(x[dst]), self.source_mlp(x[src]), edge_attr.unsqueeze(-1)], dim=-1)  # (E, c_hidden + c_hidden + 1 + c_latent)

        # edge_feat = self.ln_raw(edge_feat)
        # edge_feat = self.pi_mlp(edge_feat)  # (E, 2)
        #bulk feat
        bulk_dst = torch.arange(x.size(0), device=self.device)
        dst_all = torch.cat([dst, bulk_dst])
        
        bulk_edge_feat = torch.cat([self.target_bulk_mlp(x[bulk_dst]), self.bulk_mlp(bulk_clu[clusters]), torch.zeros(x.size(0), 1, device=self.device)], dim=-1)
        # bulk_edge_feat = self.ln_bulk(bulk_edge_feat)
        # bulk_edge_feat = self.pi_bulk_mlp(bulk_edge_feat)  # (N, 2)

        edge_feat_ext = torch.cat([edge_feat, bulk_edge_feat], dim=0)  # (E+N, d)
        
        logits = self.pi_mlp(edge_feat_ext).squeeze(-1)  # (E,)
        # logits = torch.cat([edge_feat, bulk_edge_feat], dim=0).squeeze(-1)  # (E+N, d)
        logits = logits.clamp(-30, 30)
        assert torch.isfinite(logits).all(), \
            f"Non-finite logits detected: {logits}"
        # q_pi = scatter_softmax(logits, dst)
        # deg_dst = scatter(torch.ones_like(dst_all, dtype=torch.float), dst_all, dim=0, reduce="sum")
        # assert (deg_dst[dst_all] > 0).all(), \
        #     f"Zero indegree detected for some dst nodes: {deg_dst[dst_all]}"

        
        # logits_norm = logits / (deg_dst[dst_all].sqrt() + 1e-6)
        # assert torch.isfinite(logits_norm).all(), \
        #     f"Non-finite logits_norm detected: {logits_norm}"
        
        # probs = scatter_softmax(logits_norm, dst_all)
        probs = scatter_softmax(logits, dst_all)
        assert torch.isfinite(probs).all(), \
            f"NaN/Inf detected in probs: {probs}"
        
        # sums = scatter(probs, dst_all, dim=0, reduce="sum")
        # mask = ~torch.isclose(sums, torch.ones_like(sums), atol=1e-5)
        # if mask.any():
        #     bad_idx = mask.nonzero(as_tuple=True)[0]
        #     bad_vals = sums[bad_idx]
        #     bad_degs = deg_dst[bad_idx]
        #     raise AssertionError(
        #         f"Softmax not normalized. Nodes={bad_idx.tolist()} "
        #         f"sum={bad_vals.tolist()} indeg={bad_degs.tolist()}"
        #     )


        q_pi = probs[:edge_index.size(1)]
        q_clu_weight = probs[edge_index.size(1):]

        
        with pyro.plate("edges", edge_index.size(1), dim=-1):

            pyro.sample('pi',
                dist.Delta(q_pi)
            )

            entropy = self.ent_loss(probs, dst_all)
            assert torch.isfinite(entropy), \
                f"NaN in entropy_reg: {entropy}"
            w_ent = getattr(self, "current_weights", {}).get("entropy", self.cfg.entropy_weight)
            pyro.factor("entropy_reg", w_ent * entropy, has_rsample=True)

            src_clu = clusters[src]   # (E,)
            dst_clu = clusters[dst]   # (E,)
            same_cluster_mask = (src_clu == dst_clu).float()    # (E,)
            same_mass_per_dst = scatter(q_pi * same_cluster_mask, dst, dim=0, reduce="sum")  # (N,)
            # normalize by total mass per node (should be ~1, but safe)
            total_mass_per_dst = scatter(q_pi, dst, dim=0, reduce="sum") + 1e-8
            frac_same = same_mass_per_dst / total_mass_per_dst  # (N,)
            # average fraction of within-cluster mass
            penalty = frac_same.mean()
            w_pen = getattr(self, "current_weights", {}).get("penalty", self.cfg.cluster_penalty)
            pyro.factor("same_cluster_penalty", w_pen * penalty, has_rsample=True)


            

        # z -------------------------------------------------------------
        with pyro.plate("cells", x.size(0)):
            
            pyro.sample('clu_weight',
                dist.Delta(q_clu_weight)
            )
            
            mu_z, sigma_z = self.encoder_head(x).chunk(2, dim=-1)

            sigma_z = sigma_z.exp().clamp(max=MAX_SCALE) + EPS
            qz = pyro.sample("z", dist.Normal(mu_z, sigma_z).to_event(1))

            q_neigh_eff = self._weighted_sum(edge_index, q_pi, qz)

            clus_emb = clus[clusters]
            hsic = self.hsic_loss(q_neigh_eff, clus_emb)
            w_hsic = getattr(self, "current_weights", {}).get("hsic", self.cfg.hsic_weight)
            # pyro.factor("hsic_independence", w_hsic * hsic, has_rsample=True)
            


    def _anneal_weight(self, epoch: int, start: float, end: float, max_epoch: int) -> float:
        """
        Linearly anneal from `start` → `end` over `max_epoch` epochs.
        After max_epoch, stays at `end`.
        """
        t = min(epoch / max_epoch, 1.0)
        return start + t * (end - start)


    # -------------------------------------------------------------------
    def fit(self, train_loader: DataLoader,
            val_loader:   Optional[DataLoader] = None,
            ):
        self.to(self.device)

        # svi = SVI(self.model, self.guide,
        #           Adam({"lr": self.cfg.lr}), Trace_ELBO())
        loss_fn = lambda model, guide, batch: pyro.infer.Trace_ELBO().differentiable_loss(model, guide, batch)
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = loss_fn(self.model, self.guide, next(iter(train_loader)).to(self.device))
        params = set(site["value"].unconstrained()
                for site in param_capture.trace.nodes.values())
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr)



        pbar = tqdm(range(1, self.cfg.n_epochs + 1), desc="Training")

        train_elbo, val_elbo, r2_hist = [], [], []
        self.use_gate = False

        for epoch in pbar:
            self.train()
            loss_sum, n_obs = 0.0, 0

            entropy_w = self._anneal_weight(epoch, -1e3, self.cfg.entropy_weight, max_epoch=100)
            penalty_w = self._anneal_weight(epoch, 0.0, self.cfg.cluster_penalty, max_epoch=2)
            hsic_w    = self._anneal_weight(epoch, 0.0, self.cfg.hsic_weight,    max_epoch=100)

            self.current_weights = {
                "entropy": entropy_w,
                "penalty": penalty_w,
                "hsic": hsic_w,
            }

            # adjust tau
            # linear tau: τ_hi → τ_lo over remaining epochs
            # ----- switch gate ON exactly at gate_start ------------
            # if (not self.use_gate) and epoch >= self.warmup:
            #     self.use_gate = True
            #     print(f"▶  Gate training ENABLED at epoch {epoch}")

            # if self.use_gate:
            #     t = (epoch - self.warmup) / max(1, self.cfg.n_epochs - self.warmup)
            #     tau_now = self.tau_hi * (self.tau_lo / self.tau_hi) ** t
            #     pyro.get_param_store()["tau_temp"] = torch.tensor(tau_now,
            #                                                        device=self.device)
            # else:
            #     tau_now = self.tau_hi

            for batch in train_loader:
                batch = batch.to(self.device)
                loss = loss_fn(self.model, self.guide, batch)
                loss_sum += loss.item()
                # loss_sum += svi.step(batch)
                n_obs    += batch.x.size(0)

                loss.backward()
                # if epoch % 10 == 0:                       # every 10 epochs is enough
                #     grad_norm = self.lambda_mlp[-1].weight.grad.norm().item()
                    # print("‖∇θ‖ for λ-head:", grad_norm)
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()


            train_elbo.append(loss_sum / n_obs)

            # ----- optional validation --------------------------------
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss, m_obs = 0.0, 0
                    preds, targs, pis, hsics, ents   = [], [], [], [], []
                    penalties = []
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        val_loss += loss_fn(self.model, self.guide, batch)
                        m_obs    += batch.x.size(0)

                        g_tr  = poutine.trace(self.guide).get_trace(batch)
                        m_tr  = poutine.trace(
                                    poutine.replay(self.model, trace=g_tr)
                                ).get_trace(batch)
                        v_mu  = m_tr.nodes["v"]["value"]
                        pi = g_tr.nodes['pi']['value']
                        lib   = batch.x.sum(-1, keepdim=True)
                        px    = F.softmax(self.decoder(v_mu), dim=-1) * lib
                        preds.append(px.cpu())
                        targs.append(batch.x.cpu())
                        pis.append(pi.cpu())

                        clu_eff = m_tr.nodes["clu_eff"]["value"]
                        neigh_eff = m_tr.nodes["neigh_eff"]["value"]
                        hsic = self.hsic_loss(neigh_eff, clu_eff)

                        hsics.append(hsic.cpu())

                        bulk_dst = torch.arange(batch.x.size(0), device=self.device)
                        dst_all = torch.cat([batch.edge_index[1], bulk_dst])
                        clu_weight = g_tr.nodes['clu_weight']['value']
                        probs = torch.cat([pi, clu_weight], dim=0)
                        ent = self.ent_loss(probs, dst_all)
                        ents.append(ent.cpu())


                        src, dst = batch.edge_index
                        src_clu = batch.cluster[src]
                        dst_clu = batch.cluster[dst]
                        same_cluster_mask = (src_clu == dst_clu).float()    # (E,)
                        same_mass_per_dst = scatter(pi * same_cluster_mask, dst, dim=0, reduce="sum")  # (N,)
                        total_mass_per_dst = scatter(pi, dst, dim=0, reduce="sum") + 1e-8
                        frac_same = same_mass_per_dst / total_mass_per_dst  # (N,)
                        penalty = frac_same.mean()
                        penalties.append(penalty.cpu())


                        # if 'pi' in g_tr.nodes.keys():
                        #     probs.append(g_tr.nodes["pi"]["value"].cpu())
                        # else:
                        #     probs.append(torch.ones_like(batch.edge_attr).cpu())

                    val_elbo.append(val_loss / m_obs)
                    r2 = r2_score(torch.cat(targs).numpy().flatten(),
                                  torch.cat(preds).numpy().flatten())
                    r2_hist.append(r2)

                    pis = torch.cat(pis).flatten()
                    hsics = torch.stack(hsics).flatten()
                    ents = torch.stack(ents).flatten()
                    penalties = torch.stack(penalties).flatten()


                    # q_pi = torch.cat(probs).flatten()
                    # pi25 = torch.quantile(q_pi, 0.25).item()
                    # pi95 = torch.quantile(q_pi, 0.95).item()


            
            # progress bar text
            txt = f"E{epoch:03d}  trainELBO={train_elbo[-1]:.3f}"
            if val_loader is not None:
                # txt += f"  valELBO={val_elbo[-1]:.3f}  R²={r2_hist[-1]:.3f} pi25={pi25:.2f} pi95={pi95:.2f} v={v_mu.mean().item():.2f}"
                txt += f"  valELBO={val_elbo[-1]:.3f}  R²={r2_hist[-1]:.3f}"
                txt += f" v={v_mu.mean().item():.2f}"
                txt += f" pi 25, 95 = {torch.quantile(pis, 0.25).item():.3f} {torch.quantile(pis, 0.95).item():.3f}"
                txt += f" hsic={hsics.mean():.3f} {hsics.std():.3f}"
                txt += f" ent={ents.mean():.3f}"
                txt += f" penalty={penalties.mean():.3e}"
            pbar.set_description(txt)

        return train_elbo, val_elbo, r2_hist
    
    # ---------------------------------------------------------- helpers
    @torch.no_grad()
    def get_params(self, data: Data, device: torch.device = 'cpu'):
        dl = DataLoader(data, shuffle=True)
        data = next(iter(dl)).to(device)

        tmp_device = self.device

        self.device = device
        self.to(self.device)

        guide_tr = poutine.trace(self.guide).get_trace(data)
        model_tr = poutine.trace(poutine.replay(self.model, trace=guide_tr)).get_trace(data)
        
        # lam = guide_tr.nodes['lam_local']['value'].cpu() * guide_tr.nodes['tau']['value'].cpu()
        pi = model_tr.nodes['pi']['value'].cpu()
        # pi = model_tr.nodes['pi']['value'].cpu()
        # lam = None

        z_mu = guide_tr.nodes["z"]["value"]

        # v_dist: dist.Normal = guide_tr.nodes["v"]["fn"].base_dist
        # v_mu = v_dist.loc
        v_mu = model_tr.nodes["v"]["value"]

        lib = data.x.sum(dim=-1, keepdim=True)
        px = F.softmax(self.decoder(v_mu), dim=-1) * lib

        self.device = tmp_device
        self.to(self.device)

        return {"qz": z_mu.cpu(), "pi": pi, "px": px.cpu(), "guide": guide_tr, "model": model_tr}