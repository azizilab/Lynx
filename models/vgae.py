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
from pyro.infer.reparam import ProjectedNormalReparam
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

        self.prior_dist = configs.prior 
        self.pz_u = ConditionalPrior(configs)
        self.encode = Encoder(configs)
        self.decode = Decoder(configs)

        self.to(device)

        assert self.prior_dist == 'normal' or self.prior_dist == 'vMF', \
            """Prior distribution type {} not implemented yet\n
               Please choose from `normal` & `vMF`""".format(self.prior_dsit)

    def model(self, x, u, edge_index):
        pyro.module("VGAE", self)
        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float),
            constraint=dist.constraints.positive
        ).to(self.device)

        l = x.sum(axis=-1, keepdim=True)
        if self.prior_dist == 'vMF':
            z_concentration = self.pz_u(u)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            if self.prior_dist == 'normal':
                z_mu = self.pz_u(u)
                z_std = torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device)
                z = pyro.sample(
                    "z",
                    dist.Normal(z_mu, z_std).to_event(1)
                )
            else:
                z = self._sample_von_mise_fisher(z_concentration)

            mu = self.decode(z, edge_index)
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

    def guide(self, x, u, edge_index):
        pyro.module("Logit_VGAE", self)
        log_x = torch.log(x+EPS)
        z_param = self.encode(log_x, u, edge_index)

        with pyro.plate("batch", log_x.size(0)), poutine.scale(scale=self.configs.beta):
            if self.prior_dist == 'normal':
                z_mu, z_logvar = z_param
                z_std = torch.exp(z_logvar/2)
                pyro.sample(
                    "z", 
                    dist.Normal(z_mu, z_std).to_event(1)
                )
            else:
                self._sample_von_mise_fisher(z_param)

    @poutine.reparam(config={"z": ProjectedNormalReparam()})
    def _sample_von_mise_fisher(concentration):
        return pyro.sample(
            "z",
            dist.ProjectedNormal(concentration)
        )

    def get_cond_prior(self, u, device='cpu'):
        u = torch.tensor(u).to(device)
        return self.pz_u(u)

    def get_z(self, x, u, edge_index, device='cpu'):
        self.eval()
        log_x = torch.log(
            torch.tensor(x).float().to(device) + EPS
        )
        u = torch.tensor(u).float().to(device)
        ei = torch.tensor(edge_index).to(device)

        if self.prior_dist == 'normal':
            z_mu, z_logvar, attn_weights = self.encode(log_x, u, ei, return_attn=True)
            return z_mu, z_logvar, attn_weights
        else:
            z_concentration = self.encode(log_x, u, ei)
            return z_concentration
    
    def sample_z(self, x, u, edge_index, n_samples=100):
        if self.prior_dist == 'normal':
            z_mu, z_logvar = self.get_z(x, u, edge_index)
            z_samples = dist.Normal(z_mu, torch.exp(z_logvar//2)).sample((n_samples,))
        else:
            z_concentration = self.get_z(x, u, edge_index)
            z_samples = dist.ProjectedNormal(z_concentration).sample((n_samples,))
        return z_samples
    
    def get_x(self, x, edge_index, z_param, device='cpu'):
        self.eval()

        l = torch.tensor(x).float().sum(axis=-1, keepdim=True).to(device)
        ei = torch.tensor(edge_index).to(device)
        if self.prior_dist == 'normal':
            z_mu = torch.tensor(z_param[0]).float().to(device)
            z_logvar = torch.tensor(z_param[1]).float().to(device)
            z = dist.Normal(z_mu, torch.exp(z_logvar/2)).sample()
        else:
            z_conc = torch.tensor(z_param).float().to(device)
            z = dist.ProjectedNormal(z_conc).sample()
            
        mu  = self.decode(z, ei)
        px_mu = l * mu
        return px_mu
    
    def sample_x(self, x, u, edge_index, n_samples=100, device='cpu'):
        self.eval()
        x = torch.tensor(x).float().to(device)
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
        pz = np.zeros_like(qz)
        px = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qz = res.qz_params[0].detach().cpu().numpy() \
                       if isinstance(res.qz_params, tuple) \
                       else res.qz_params[0].detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()

            for pos, qz_i, pz_i, px_i in zip(data.pos, batch_qz, batch_pz, batch_px):
                idx = position_map[tuple(pos.detach().cpu().numpy().astype(np.float32))]
                qz[idx], pz[idx], px[idx] = qz_i, pz_i, px_i
        
        return ConfigDict({
            'qz':   qz,
            'pz':   pz,
            'px':   px
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
    # DEBUG the wrong factorization but force learning q(z | u):
    # q(z | x, u) ~approx q(z | x) * q(z | u)
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.prior_dist = configs.prior
        self.enc_option = configs.enc_option
        activation = configs.act
        self.num_heads = configs.num_heads

        self.gene_embedding =  nn.Parameter(torch.randn(configs.c_in, configs.c_embedding))
        self.u_embedding =  nn.Parameter(torch.randn(configs.c_aux, configs.c_embedding))
        
        c_obs = configs.c_in + configs.c_aux
        self.obs_to_hid = nn.Sequential(
            nn.Linear(c_obs, configs.c_hidden),
            activation,
            nn.Linear(configs.c_hidden, configs.c_hidden)
        )

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

        self.W_x = nn.Parameter(torch.randn(configs.c_in, configs.c_embedding//2))
        self.W_u = nn.Parameter(torch.randn(configs.c_aux, configs.c_embedding//2))

        use_bias = False
        self.to_key = nn.Linear(configs.c_embedding, configs.c_embedding, bias=use_bias)
        self.to_query = nn.Linear(configs.c_embedding, configs.c_embedding, bias=use_bias)
        self.to_value = nn.Linear(configs.c_embedding, configs.c_embedding, bias=use_bias)

        self.layer_norm_q = nn.LayerNorm(configs.c_embedding)
        self.layer_norm_k = nn.LayerNorm(configs.c_embedding)
        self.layer_norm_v = nn.LayerNorm(configs.c_embedding)

        self.register_buffer("identity_proj", torch.eye(configs.c_embedding))

        self.out_proj_weight = nn.Parameter(torch.randn(configs.c_embedding, configs.c_embedding))
        self.out_proj_bias = nn.Parameter(torch.randn(configs.c_embedding))

        self.seq_to_hidden = nn.Linear(configs.c_aux, configs.c_hidden)

        self.hid_to_zmu = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zlogvar = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        self.hid_to_zconc = SGConv(configs.c_hidden, configs.c_latent, K=configs.k_hop)
        
    def forward(self, x, u, edge_index, return_attn=False):
        if self.embed_option == 'cat':
            obs = torch.cat([x, u], dim=-1)
            h = self.obs_to_hid(obs)

        elif self.embed_option == 'attn':

            def signal_transform(x):
                assert x.shape[-1] % 2 == 0
                
                # Split the tensor into two halves along the last dimension
                x_cos = torch.cos(x)  # Apply cosine to the even indices
                x_sin = torch.sin(x)  # Apply sine to the same indices

                transformed = torch.concat([x_cos, x_sin], dim=-1)

                # Normalize by sqrt(d)
                return transformed / np.sqrt(x.shape[-1])



            m_x = self.mixed_x_signal(x)
            m_u = self.mixed_u_signal(u)

            m_x = signal_transform(m_x)
            m_u = signal_transform(m_u)

            g_prime = x[...,None]*self.W_x
            m_prime = u[...,None]*self.W_u

            g_prime = signal_transform(g_prime)
            m_prime = signal_transform(m_prime)
            

            g = g_prime - m_x[:, None, :] + self.gene_embedding
            m = m_prime - m_u[:, None, :] + self.u_embedding

            # g = self.layer_norm_k(g)
            # m = self.layer_norm_q(m)

            Q = self.to_query(m)
            K = self.to_key(g)
            V = self.to_value(g)

            #its sequence first, not batch first
            Q = torch.transpose(Q, 0, 1)
            K = torch.transpose(K, 0, 1)
            V = torch.transpose(V, 0, 1)



            # Apply LayerNorm to query, key, and value before attention
            Q = self.layer_norm_q(Q)  # Normalized query
            K = self.layer_norm_k(K)      # Normalized key
            V = self.layer_norm_v(V)  # Normalized value

            h, attn_weights = F.multi_head_attention_forward(
                query=Q,
                key=K,
                value=V,
                embed_dim_to_check=Q.shape[-1],
                num_heads=self.num_heads,
                in_proj_weight=None,       # No input projection weight
                in_proj_bias=None,         # No input projection bias
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0.0,
                out_proj_weight=self.out_proj_weight,  # Set output projection weights
                out_proj_bias=self.out_proj_bias,      # Set output projection bias
                training=self.training,
                key_padding_mask=None,
                need_weights=return_attn, #set true to return
                attn_mask=None,
                use_separate_proj_weight=True,    # Use separate projection weights
                q_proj_weight=self.identity_proj,               # No projection for query
                k_proj_weight=self.identity_proj,               # No projection for key
                v_proj_weight=self.identity_proj,               # No projection for value
                static_k=None,
                static_v=None,
                average_attn_weights=True,
                is_causal=False
            )

            h = torch.transpose(h, 0, 1)

            # theta = torch.pow(100, -2*(torch.arange(0, Q.shape[-1]/2)-1)/Q.shape[-1])

            # rotation_query = Q*torch.cos(theta)
            # rotation_key = K*


            # attn_weights = torch.softmax(torch.bmm(Q, torch.transpose(K, 1, 2)), -1)
            # h = F.scaled_dot_product_attention(Q, K, V)
            h = torch.mean(h, dim=-1)

            h = F.relu(h)
            h = self.seq_to_hidden(h)
            h = F.relu(h)

        else:
            raise NotImplementedError(
                'Integration option {} not implemented in Encoder'.format(self.integrate_option)
            )

        if self.prior_dist == 'normal':
            z_mu = self.hid_to_zmu(h, edge_index)
            z_logvar = self.hid_to_zlogvar(h, edge_index)
            return z_mu, z_logvar, attn_weights
        else:
            z_conc = self.hid_to_zconc(h, edge_index)
            return z_conc


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

    def forward(self, z, edge_index):
        h = self.z_to_hid(z, edge_index)
        mu = self.hid_to_xmu(h) + EPS
        return mu
