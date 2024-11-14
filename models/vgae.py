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
from torch_geometric.nn import SGConv, Sequential
from torch_geometric.loader import DataLoader
from pyro.poutine import trace

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dataset import XeniumGraphDataset

EPS = 1e-8


class VGAE(nn.Module):
    """
    Conditional VGAE to learn Latent Manifold 
    """
    def __init__(self, configs, device='cuda'):
        super(VGAE, self).__init__()
        self.configs = configs
        self.device = device

        # Shared layers for q(z | u) & p(z | u)
        self.pz_u = ConditionalPrior(configs)
        configs.pz_u = self.pz_u

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
            z_mu, z_logvar = self.pz_u(u)
            z_std = torch.exp(z_logvar/2)
            # z_std = torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device)
            
            z = pyro.sample(
                "z",
                dist.Normal(z_mu, z_std).to_event(1)
            )

            mu = self.decode(z, s, edge_index)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - (self.theta).log()

            nb_dist = dist.NegativeBinomial(total_count=self.theta, logits=logits)
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

    def get_z(self, x, u, s, edge_index):
        if self.configs.embed_option == 'attn':
            l = x.sum(axis=-1, keepdim=True)
            x = x / l * l.median()
        x = torch.log(x+EPS)
        z_mu, z_logvar, attn_weights = self.encode(x, u, s, edge_index)
        return z_mu, z_logvar, attn_weights
    
    def sample_z(self, x, u, s, edge_index, n_samples=100):
        z_mu, z_logvar, _ = self.get_z(x, u, s, edge_index)
        z_samples = dist.Normal(z_mu, torch.exp(z_logvar//2)).sample((n_samples,))
        return z_samples
    
    def get_x(self, x, s, edge_index, z_param):
        self.eval()
        l = x.sum(axis=-1, keepdim=True)

        z_mu = z_param[0]
        z_logvar = z_param[1]
        z = dist.Normal(z_mu, torch.exp(z_logvar/2)).sample()
            
        mu  = self.decode(z, s, edge_index)
        px_mu = l * mu
        return px_mu
    
    def sample_x(self, x, u, edge_index, n_samples=100):
        self.eval()
        x = torch.tensor(x).float()
        x = torch.log(x + EPS)
        u = torch.tensor(u).float()
        ei = torch.tensor(edge_index)

        predictive = pyro.infer.Predictive(self, self.guide, n_samples)
        pxs = predictive(x, u, ei)
        return pxs["x"]
    
    def predict(self, data, device):
        """
        Predict latent representation & reconstructions 
        on full data
        """
        self.eval()
        x = data.x.to(device).float()
        u = data.u.to(device).float()
        s = data.s.to(device).float()
        edge_index = data.edge_index.to(device)

        pz, _ = self.pz_u(u)
        qz_params = self.get_z(x, u, s, edge_index)
        x_mu = self.get_x(x, s, edge_index, qz_params)

        return ConfigDict({
            'qz_params':    qz_params,
            'pz':           pz,
            'px':           x_mu
        })
    
    def evaluate(self, adata, k=30, n_subgraphs=8, device=torch.device('cuda')):
        """
        Predict latent representation & reconstructions 
        on mini-batched subgraphs
        """
        self.eval()
        self.device = device
        self.to(device)
        self._move_attr_to(device)

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
        # attn_weights = []
        pz = np.zeros_like(qz)
        px = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)
        for data in dataloader:
            res = self.predict(data, device=device)
            batch_qz = res.qz_params[0].detach().cpu().numpy()
            batch_pz = res.pz.detach().cpu().numpy()
            batch_px = res.px.detach().cpu().numpy()

            # if self.configs.embed_option == 'attn':
            #     attn_weights.append(
            #         res.qz_params[-1].detach().cpu().numpy()
            #     )

            for pos, qz_i, pz_i, px_i in zip(data.pos, batch_qz, batch_pz, batch_px):
                idx = position_map[tuple(pos.detach().cpu().numpy().astype(np.float32))]
                qz[idx], pz[idx], px[idx] = qz_i, pz_i, px_i
        
        # if self.configs.embed_option == 'attn':
        #     attn_weights = np.vstack(attn_weights).mean(0)

        return ConfigDict({
            'qz':           qz,
            'pz':           pz,
            'px':           px
            # 'attn_weights': attn_weights,
        })
        
    def _move_attr_to(self, device):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))


class SparseVGAE(VGAE):
    """
    VGAE with Spike-and-Slab LASSO (SSL) on 
    both conditional prior p(z | u) & likelihood p(x | z)
    """
    # TODO: set sparsity constraints on z -> x
    def __init__(self, configs, device='cuda'):
        super(SparseVGAE, self).__init__(configs, device)
        self.device = device
        self.to(device)
        
        # self.decode = Decoder(configs)
        self.decode = SSLDecoder(configs, device=device)

        self.a = torch.tensor(configs.a, device=device)
        self.b = torch.tensor(configs.b, dtype=torch.float, device=device)

        # parameters for SSL prior
        self.W = pyro.param(
            "W", 
            torch.rand(self.configs.c_in, self.configs.c_latent, dtype=torch.float, device=device)
        )

        # placeholder 
        self.w_norm = None
        self.theta = None
        self.eta_map = None
        self.gamma_map = None
        self.psi1 = None
        self.psi0 = None

    def model(self, x, u, s, edge_index):
        pyro.module("SparseVGAE", self)
        l = x.sum(axis=-1, keepdim=True)

        self.theta = pyro.param(
            "theta",
            torch.ones(self.configs.c_in, dtype=torch.float, device=self.device),
            constraint=dist.constraints.positive
        ).to(self.device)

        with pyro.plate("aux", self.configs.c_latent), poutine.scale(scale=self.configs.beta):
            eta = pyro.sample("eta", dist.Beta(self.a, self.b)).to(self.device) 

            with pyro.plate("latent", self.configs.c_in), poutine.scale(scale=self.configs.beta):
                gamma = pyro.sample("gamma", dist.Bernoulli(eta)).to(self.device)
                self.psi1 = pyro.sample(
                    "psi1", 
                    dist.Laplace(loc=self.W, scale=self.configs.lambda1)
                ).to(self.device)

                self.psi0 = pyro.sample(
                    "psi0",
                    dist.Laplace(loc=self.W, scale=self.configs.lambda0)
                ).to(self.device)
        
        w = self._normalize_w(gamma*self.psi1 + (1-gamma)*self.psi0).to(self.device)

        with pyro.plate("batch", x.size(0)), poutine.scale(scale=self.configs.beta):
            z_mu, z_logvar = self.pz_u(u)
            z_std = torch.exp(z_logvar/2)
            z = pyro.sample(
                "z",
                dist.Normal(z_mu, z_std).to_event(1)
            )

            mu = self.decode(z, s, w, edge_index)
            # (z.device, s.device, w.device, mu.device, l.device)
            x_mu = l * mu
            logits = (x_mu+EPS).log() - (self.theta).log()

            nb_dist = dist.NegativeBinomial(total_count=self.theta, logits=logits)
            pyro.sample(
                "x",
                nb_dist.to_event(1),
                obs=x
            )
            
    def guide(self, x, u, s, edge_index):
        pyro.module("SparseVGAE", self)

        # MAP inference
        self.gamma_map = pyro.param(
            "gamma_map",
            0.5 * torch.ones(self.configs.c_latent, dtype=torch.float, device=self.device),
            constraint=dist.constraints.unit_interval
        )

        self.eta_map = pyro.param(
            "eta_map",
            torch.tensor(0.5, device=self.device),
            constraint=dist.constraints.unit_interval
        )

        with pyro.plate("aux", self.configs.c_latent), poutine.scale(scale=self.configs.beta):
            pyro.sample("eta", dist.Delta(self.eta_map))

            with pyro.plate("latent", self.configs.c_in), poutine.scale(scale=self.configs.beta):
                pyro.sample("gamma", dist.Bernoulli(self.gamma_map))
                pyro.sample(
                    "psi1", 
                    dist.Laplace(loc=self.W, scale=self.configs.lambda1)
                ).to(self.device)
                
                pyro.sample(
                    "psi0",
                    dist.Laplace(loc=self.W, scale=self.configs.lambda0)
                ).to(self.device)
                
        # VI inference
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

    def get_x(self, x, u, s, edge_index, z_param):
        self.eval()
        l = x.sum(axis=-1, keepdim=True)

        # Retrieve SSL parameters
        traced_model = trace(self.model).get_trace(x, u, s, edge_index)
        gamma = traced_model.nodes["gamma"]["value"]

        is_stored_param = (
            'psi1' in pyro.get_param_store().keys() and \
            'psi0' in pyro.get_param_store().keys()
        )
        if is_stored_param:
            psi1 = pyro.get_param_store().get_param('psi1').to(self.device)
            psi0 = pyro.get_param_store().get_param('psi0').to(self.device)        
        else:
            psi1 = dist.Laplace(loc=self.W, scale=self.configs.lambda1).rsample().to(self.device)
            psi0 = dist.Laplace(loc=self.W, scale=self.configs.lambda0).rsample().to(self.device)
                
        w = self._normalize_w(gamma*psi1 + (1-gamma)*psi0).to(self.device)
        self.w_norm = w      

        # Decode
        z_mu = z_param[0]
        z_logvar = z_param[1]
        z = dist.Normal(z_mu, torch.exp(z_logvar/2)).sample()
            
        mu  = self.decode(z, s, w, edge_index)
        px_mu = l * mu
        return px_mu
    
    def predict(self, data, device):
        """
        Predict latent representation & reconstructions 
        on full data
        """
        self.eval()
        x = data.x.to(device).float()
        u = data.u.to(device).float()
        s = data.s.to(device).float()
        edge_index = data.edge_index.to(device)

        pz, _ = self.pz_u(u)
        qz_params = self.get_z(x, u, s, edge_index)
        x_mu = self.get_x(x, u, s, edge_index, qz_params)

        return ConfigDict({
            'qz_params':    qz_params,
            'pz':           pz,
            'px':           x_mu
        })

    def _normalize_w(self, w):
        return F.normalize(w.abs(), p=1, dim=-1)
    
    def save(self, param_store, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'pyro_params': param_store.get_state()
        }, path)

    def load(self, path):
        assert os.path.isfile(path), "Model path {} doesn't exist".format(path)
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])

        pyro.get_param_store().set_state(checkpoint['pyro_params']) 
        param_store = pyro.get_param_store()
        self.theta = param_store.get_param('theta').to(self.device)
        self.eta_map = param_store.get_param('eta_map').to(self.device)
        self.gamma_map = param_store.get_param('gamma_map').to(self.device)
        return None


class ConditionalPrior(nn.Module):
    def __init__(self, configs):
        super(ConditionalPrior, self).__init__()
        activation = configs.act

        self.u_to_hid = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            activation
        )
        
        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)
        
    def forward(self, u):
        h = self.u_to_hid(u)
        z_mu, z_logvar = self.hid_to_zmu(h), self.hid_to_zlogvar(h)
        return z_mu, z_logvar
    

class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.embed_option = configs.embed_option
        activation = configs.act
        
        self.activation = activation
        self.num_heads = configs.num_heads
        self.c_embedding = configs.c_embedding
        self.dropout_p = configs.dropout

        # TODO: try bilinear pooling
        self.x_to_hid = Sequential('x, edge_index', [
            (SGConv(configs.c_in, configs.c_hidden, K=configs.k_hop), 'x, edge_index -> h'),
            activation, 
        ])

        self.u_to_hid = configs.pz_u.u_to_hid

        # self.fuse_layer = nn.Bilinear(configs.c_hidden, configs.c_hidden, configs.c_hidden)
        # self.hid_to_zmu = SGConv(configs.c_hidden, configs.c_latent, K=1)
        # self.hid_to_zlogvar = SGConv(configs.c_hidden, configs.c_latent, K=1)
        
        self.hid_to_zmu = nn.Bilinear(configs.c_hidden, configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Bilinear(configs.c_hidden, configs.c_hidden, configs.c_latent)

        # Cross-attention layers
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

        self.layer_norm_q = nn.LayerNorm(configs.c_embedding)
        self.layer_norm_k = nn.LayerNorm(configs.c_embedding)
        self.layer_norm_v = nn.LayerNorm(configs.c_embedding)

        self.attn_to_hid = Sequential('x, edge_index', [
            (SGConv(configs.c_in, configs.c_hidden, K=1), 'x, edge_index -> h'),
            activation,
        ])

        self.q_proj_weight = nn.Parameter(torch.empty(configs.c_embedding, configs.c_embedding))
        self.k_proj_weight = nn.Parameter(torch.empty(configs.c_embedding, configs.c_embedding))
        self.v_proj_weight = nn.Parameter(torch.empty(configs.c_embedding, configs.c_embedding))
        self.out_proj_weight = nn.Parameter(torch.empty(configs.c_embedding, configs.c_embedding))
        self.out_proj_bias = nn.Parameter(torch.randn(configs.c_embedding))

        nn.init.xavier_normal_(self.W_x)
        nn.init.xavier_normal_(self.W_u)
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)

    def forward(self, x, u, s, edge_index):
        attn_weights = None
        # need_weights = not self.training

        if self.embed_option == 'cat':
            hx = self.x_to_hid(x, edge_index)
            hu = self.u_to_hid(u)

            # z_mu = self.hid_to_zmu(h, edge_index)
            # z_logvar = self.hid_to_zlogvar(h, edge_index)

            z_mu = self.hid_to_zmu(hx, hu)
            z_logvar = F.relu(self.hid_to_zlogvar(hx, hu))

        elif self.embed_option == 'attn':            
            gene_embedding = self._signal_transform(
                torch.einsum('NG, GE -> NGE', x, self.W_x)
            )
            metabolite_embedding = self._signal_transform(
                torch.einsum('NG, GE -> NGE', u, self.W_u)
            )

            # batch first -> sequence first; dim: [G, N, E]
            Q = gene_embedding.transpose(0, 1)
            K = metabolite_embedding.transpose(0, 1)
            V = metabolite_embedding.transpose(0, 1)

            attn_output, attn_weights = F.multi_head_attention_forward(
                query=Q, key=K, value=V,
                embed_dim_to_check=Q.shape[-1],
                num_heads=self.num_heads,
                in_proj_weight=None, in_proj_bias=None,        
                bias_k=None, bias_v=None,
                add_zero_attn=False,
                dropout_p=self.dropout_p,
                out_proj_weight=self.out_proj_weight, out_proj_bias=self.out_proj_bias,      
                training=self.training, need_weights=False,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                # average_attn_weights=True
            )       
            attn_output = attn_output.transpose(0, 1)  # dim: [N, G, E]
            attn_output = F.avg_pool1d(attn_output, kernel_size=self.c_embedding).squeeze()

            h = self.attn_to_hid(attn_output, edge_index)
            z_mu = self.hid_to_zmu(h, edge_index)
            z_logvar = self.hid_to_zlogvar(h, edge_index)

        else:
            raise NotImplementedError(
                'Integration option {} not implemented in Encoder'.format(self.integrate_option)
            )
        
        return z_mu, z_logvar, attn_weights
    
    @staticmethod
    def _signal_transform(x):
        assert x.shape[-1] % 2 == 0
        x_cos = torch.cos(x)  
        x_sin = torch.sin(x)

        transformed = torch.concat([x_cos, x_sin], dim=-1)
        return transformed / np.sqrt(x.shape[-1])


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()        
        activation = configs.act
        c_hid = configs.c_hidden + configs.c_covariate  # dim. for f(z, s)

        self.z_to_hid = Sequential('z, edge_index', [
            (SGConv(configs.c_latent, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
            activation,
            nn.Dropout(p=configs.dropout)
        ])

        self.hid_to_xmu = nn.Sequential(
            nn.Linear(c_hid, configs.c_in),
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
    

class SSLDecoder(nn.Module):
    """
    Sparse Slab-and-LASSO decoder with disentangled z -> x connetions
    """
    def __init__(self, configs, device=torch.device('cuda')):
        super(SSLDecoder, self).__init__()
        self.configs = configs
        self.device = device
        activation = configs.act
        c_latent = configs.c_latent + configs.c_covariate  # dim. for f(z, s)

        self.z_to_xs = nn.ModuleList([
            Sequential('z, edge_index', [
                (SGConv(c_latent, configs.c_hidden, K=configs.k_hop), 'z, edge_index -> h'),
                activation,
                nn.Dropout(p=configs.dropout),
                nn.Linear(configs.c_hidden, 1)
            ])
            for _ in range(configs.c_in)
        ])
        
    def forward(self, z, s, w, edge_index):
        xs = torch.zeros(z.shape[0], self.configs.c_in, device=self.device)
        z_covariate = torch.cat([z, s], dim=-1)
        
        for g in range(self.configs.c_in):
            masked_z = torch.mul(z_covariate, w[g, :])
            xs[:, g] = self.z_to_xs[g](masked_z, edge_index).squeeze()

        return F.softmax(xs, dim=-1)
       
