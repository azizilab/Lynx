import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from ml_collections import ConfigDict
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch_geometric.nn import SGConv, Sequential
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dataset import XeniumGraphDataset

EPS = 1e-8


class VGAE(nn.Module):
    """
    Conditional VGAE to learn Latent Manifold 
    """
    def __init__(self, configs, device='cpu'):
        super(VGAE, self).__init__()
        self.configs = configs
        self.device = device

        self.pz_u = ConditionalPrior(configs)
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)
        self.to(device)

    def model(self, x, u, s, edge_index):
        pyro.module("VGAE", self)
        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)
        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z_mu = self.pz_u(u)
            z_std = torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device)
            z = pyro.sample(
                "z",
                dist.Normal(z_mu, z_std).to_event(1)
            )

            mu = self.decode(z, s, edge_index)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - (self.theta).log()

            nb_dist = dist.NegativeBinomial(
                total_count=self.theta,
                logits=logits
            )

            pyro.sample(
                "x",
                nb_dist.to_event(1),
                obs=x
            )

    def guide(self, x, u, s, edge_index):
        pyro.module("VGAE", self)

        if self.configs.embed_option == 'attn':
            l = x.sum(axis=-1, keepdim=True)
            x = x / l * l.median()
        x = torch.log(x+EPS)
        z_param = self.encode(x, u, s, edge_index)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z_mu, z_logvar, _ = z_param
            z_std = torch.exp(z_logvar/2)
            pyro.sample(
                "z", 
                dist.Normal(z_mu, z_std).to_event(1)
            )

    def get_cond_prior(self, u, device='cpu'):
        u = torch.tensor(u).to(device)
        return self.pz_u(u)

    def get_z(self, x, u, s, edge_index, device='cpu'):
        if self.configs.embed_option == 'attn':
            l = x.sum(axis=-1, keepdim=True)
            x = x / l * l.median()
        x = torch.log(x+EPS).to(device) 
        u = u.to(device)
        s = s.to(device)
        edge_index = edge_index.to(device)

        z_mu, z_logvar, attn_weights = self.encode(x, u, s, edge_index)
        return z_mu, z_logvar, attn_weights
    
    def sample_z(self, x, u, s, edge_index, n_samples=100):
        z_mu, z_logvar, _ = self.get_z(x, u, s, edge_index)
        z_samples = dist.Normal(z_mu, torch.exp(z_logvar//2)).sample((n_samples,))
        return z_samples
    
    def get_x(self, x, s, edge_index, z_param, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
        l = x.sum(axis=-1, keepdim=True)
        edge_index = edge_index.to(device)

        z_mu = z_param[0].to(device)
        z_logvar = z_param[1].to(device)
        z = dist.Normal(z_mu, torch.exp(z_logvar/2)).sample()
            
        mu  = self.decode(z, s, edge_index)
        px_mu = l * mu
        return px_mu
    
    def sample_x(self, x, u, edge_index, n_samples=100, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
        x = torch.log(x + EPS)
        u = torch.tensor(u).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        predictive = pyro.infer.Predictive(self, self.guide, n_samples)
        pxs = predictive(x, u, ei)
        return pxs["x"]
    
    def predict(self, data, device=torch.device('cpu')):
        """
        Predict latent representation & reconstructions 
        on full data
        """
        self.eval()
        x = data.x.to(device).float()
        u = data.u.to(device).float()
        s = data.s.to(device).float()
        edge_index = data.edge_index.to(device)

        pz = self.get_cond_prior(u)
        qz_params = self.get_z(x, u, s, edge_index)
        x_mu = self.get_x(x, s, edge_index, qz_params)

        return ConfigDict({
            'qz_params':    qz_params,
            'pz':           pz,
            'px':           x_mu
        })
    
    def evaluate(self, adata, k=30, n_subgraphs=8, device=torch.device('cpu')):
        """
        Predict latent representation & reconstructions 
        on mini-batched subgraphs
        """
        self.to(device)
        self.device = device
        self.eval()

        position_map = {
            tuple(pos): i
            for i, pos in enumerate(
                adata.obs[['x_centroid', 'y_centroid']].values.astype(np.float32)
            )
        }
        graph_data = XeniumGraphDataset(
            k=k, n_subgraphs=n_subgraphs
        ).load_graphs([adata])

        dataloader = DataLoader(graph_data, shuffle=False)
        qz = np.zeros((adata.shape[0], self.configs.c_latent), dtype=np.float32)
        attn_weights = []
        pz = np.zeros_like(qz)
        px = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qz = res.qz_params[0].detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()

            if self.configs.embed_option == 'attn':
                attn_weights.append(
                    res.qz_params[-1].detach().cpu().numpy()
                )

            for pos, qz_i, pz_i, px_i in zip(data.pos, batch_qz, batch_pz, batch_px):
                idx = position_map[tuple(pos.detach().cpu().numpy().astype(np.float32))]
                qz[idx], pz[idx], px[idx] = qz_i, pz_i, px_i
        
        if self.configs.embed_option == 'attn':
            attn_weights = np.vstack(attn_weights).mean(0)

        return ConfigDict({
            'qz':           qz,
            'attn_weights': attn_weights,
            'pz':           pz,
            'px':           px
        })
        
    def _PD_approx(self, cov, UPLO='L'):
        eigvals, Q = torch.linalg.eigh(cov @ cov.T) if UPLO == 'L' else torch.linalg.eigh(cov)
        Qt = Q.T
        Lambda = torch.diag(torch.tensor([torch.max(v, EPS) for v in eigvals]))
        return Q @ Lambda @ Qt


class ConditionalPrior(nn.Module):
    def __init__(self, configs):
        super(ConditionalPrior, self).__init__()
        activation = configs.act
        c_hidden = min(configs.c_aux, configs.c_hidden)
        self.layer = nn.Sequential(
            nn.Linear(configs.c_aux, c_hidden),
            activation,
            nn.Linear(c_hidden, configs.c_latent),
        )

    def forward(self, x):
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.embed_option = configs.embed_option
        activation = configs.act
        self.num_heads = configs.num_heads
        self.c_embedding = configs.c_embedding
        
        c_obs = configs.c_in + configs.c_aux
        self.obs_to_hid = Sequential('obs, edge_index', [
            (SGConv(c_obs, configs.c_hidden, K=configs.k_hop), 'obs, edge_index -> h'),
            activation, 
        ])

        self.mixed_x_signal = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            activation,
            nn.Linear(configs.c_hidden, configs.c_embedding//2),
        )
        self.mixed_u_signal = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            activation,
            nn.Linear(configs.c_hidden, configs.c_embedding//2),
        )

        self.W_x = nn.Parameter(torch.empty(configs.c_in, configs.c_embedding//2))
        self.W_u = nn.Parameter(torch.empty(configs.c_aux, configs.c_embedding//2))

        # self.layer_norm_q = nn.LayerNorm(configs.c_embedding)
        # self.layer_norm_k = nn.LayerNorm(configs.c_embedding)
        # self.layer_norm_v = nn.LayerNorm(configs.c_embedding)

        self.q_proj_weight = nn.Parameter(torch.empty(configs.c_embedding, configs.c_embedding))
        self.k_proj_weight = nn.Parameter(torch.empty(configs.c_embedding, configs.c_embedding))
        self.v_proj_weight = nn.Parameter(torch.empty(configs.c_embedding, configs.c_embedding))

        self.out_proj_weight = nn.Parameter(torch.empty(configs.c_embedding, configs.c_embedding))
        self.out_proj_bias = nn.Parameter(torch.randn(configs.c_embedding))

        xavier_uniform_(self.W_x)
        xavier_uniform_(self.W_u)
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.out_proj_weight)

        self.attn_to_hid = Sequential('x, edge_index', [
            (SGConv(configs.c_in, configs.c_hidden), 'x, edge_index -> h'),
            activation,
        ])
        self.hid_to_zmu = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zlogvar = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        
    def forward(self, x, u, s, edge_index):
        attn_weights = None

        if self.embed_option == 'cat':
            obs = torch.cat([x, u], dim=-1)
            h = self.obs_to_hid(obs, edge_index)
            z_mu = self.hid_to_zmu(h, edge_index)
            z_logvar = self.hid_to_zlogvar(h, edge_index)
            return z_mu, z_logvar, attn_weights

        elif self.embed_option == 'attn':
            gene_embedding = self._signal_transform(x[..., None] * self.W_x)
            metabolite_embedding = self._signal_transform(u[..., None] * self.W_u)

            x_mixed = self._signal_transform(self.mixed_x_signal(x))
            u_mixed = self._signal_transform(self.mixed_u_signal(u))
            gene_embedding -= x_mixed[:, None, :]
            metabolite_embedding -= u_mixed[:, None, :]

            # DEBUG: test layerNorm
            # batch first -> sequence first; dim: [G, N, E]
            Q = gene_embedding.transpose(0, 1)
            K = metabolite_embedding.transpose(0, 1)
            V = metabolite_embedding.transpose(0, 1)

            # Q = self.layer_norm_q(gene_embedding).transpose(0, 1)
            # K = self.layer_norm_k(metabolite_embedding).transpose(0, 1)
            # V = self.layer_norm_v(metabolite_embedding).transpose(0, 1)

            attn_output, attn_weights = F.multi_head_attention_forward(
                query=Q, key=K, value=V,
                embed_dim_to_check=Q.shape[-1],
                num_heads=self.num_heads,
                in_proj_weight=None, in_proj_bias=None,        
                bias_k=None, bias_v=None,
                add_zero_attn=False,
                dropout_p=0.0,
                out_proj_weight=self.out_proj_weight, out_proj_bias=self.out_proj_bias,      
                training=self.training,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=True
            )       
            attn_output = attn_output.transpose(0, 1)  # dim: [N, G, E]
            attn_output = F.avg_pool1d(attn_output, kernel_size=self.c_embedding).squeeze()

            h = self.attn_to_hid(attn_output, edge_index)
            z_mu = self.hid_to_zmu(h, edge_index)
            z_logvar = self.hid_to_zlogvar(h, edge_index)

            return z_mu, z_logvar, attn_weights

        else:
            raise NotImplementedError(
                'Integration option {} not implemented in Encoder'.format(self.integrate_option)
            )
    
    @staticmethod
    def _signal_transform(x):
        assert x.shape[-1] % 2 == 0
        
        # Split the tensor into two halves along the last dimension
        x_cos = torch.cos(x)  # Apply cosine to the even indices
        x_sin = torch.sin(x)  # Apply sine to the same indices

        transformed = torch.concat([x_cos, x_sin], dim=-1)
        return transformed / np.sqrt(x.shape[-1])


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()        
        activation = configs.act

        self.z_to_hid = Sequential('z, edge_index', [
            (SGConv(configs.c_latent, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
            activation,
            nn.Dropout(p=configs.dropout)
        ])

        self.hid_to_xmu = nn.Sequential(
            nn.Linear(configs.c_hidden, configs.c_in),
            activation,
            nn.Dropout(p=configs.dropout),
            nn.Linear(configs.c_in, configs.c_in),
            nn.Softmax(-1)
        )

    def forward(self, z, s, edge_index):
        h = self.z_to_hid(z, edge_index)
        hs = torch.cat([h, s], dim=-1)
        mu = self.hid_to_xmu(hs) + EPS
        return mu
