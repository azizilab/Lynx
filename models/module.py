import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist

from torch_sparse import SparseTensor
from torch_geometric.nn import Sequential
from torch_geometric.nn import GCNConv, GATConv, SGConv


EPS = 1e-8


# ---------------------
#  VAE Prior Modules
# ---------------------

class Prior(nn.Module):
    r"""Low-dim conditional prior"""
    def __init__(self, configs, device=torch.device('cuda')):
        super().__init__()

        self.u_to_hid = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            configs.act
        )

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

        if configs.w_init is not None:
            weight = torch.tensor(configs.w_init).to(device).float()
            self.u_to_hid[0].weight = nn.Parameter(weight)

    def forward(self, u):
        h = self.u_to_hid(u)        
        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)

        return z_mu, z_logvar


# --------------------------
#  VAE Encoder / Decoders
# --------------------------
    
class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        activation = configs.act
        self.dropout_p = configs.dropout
        
        self.x_to_hid = Sequential('x, edge_index', [
            (SGConv(configs.c_in, configs.c_hidden//2, K=configs.k_hop), 'x, edge_index -> h'),
            activation, 
        ])
        
        self.u_to_hid = Sequential('u, edge_index', [
            (SGConv(configs.c_aux, configs.c_hidden//2, K=configs.k_hop), 'u, edge_index -> h'),
            activation
        ])

        self.hid_to_zmu = GCNConv(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = GCNConv(configs.c_hidden, configs.c_latent)

    def forward(self, x, u, edge_index):
        hx = self.x_to_hid(x, edge_index)
        hu = self.u_to_hid(u, edge_index)
        h = torch.cat([hx, hu], dim=-1)

        z_mu = self.hid_to_zmu(h, edge_index)
        z_logvar = self.hid_to_zlogvar(h, edge_index)        
        return z_mu, z_logvar


class PhenotypeEncoder(nn.Module):
    r"""Encoding cell-type aware hidden representation v ~ q(v | x; c)"""
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act
        self.r2r = (configs.ref, 'to', configs.ref)
        
        self.gat_conv = GATConv(
            (configs.c_in, configs.c_latent), configs.c_in, edge_dim=1,
            heads=configs.num_heads, concat=False,
        )

        self.hid_to_vmu = nn.Linear(configs.c_in, configs.c_hidden)
        self.hid_to_vlogvar = nn.Linear(configs.c_in, configs.c_hidden)

    def forward(self, x, c, edge_index_dict):
        h, attn_scores = self.gat_conv(
            (x, c),
            edge_index=edge_index_dict[self.r2r],
            return_attention_weights=True
        )

        v_mu = self.hid_to_vmu(h)
        v_logvar = self.hid_to_vlogvar(h)

        return v_mu, v_logvar, attn_scores


class GATEncoder(nn.Module):
    r"""Encoder with paired modality aggregation by
    attending `ref` (x) to `query` (u) w/ GAT -> latent: z ~ q(z | x, u)
    """
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act

        self.g_encoder = nn.Sequential(
            nn.Linear(configs.c_hidden, configs.c_hidden),
            configs.act
        )      

        self.m_encoder = nn.Sequential(
            nn.Linear(configs.c_hidden, configs.c_hidden),
            configs.act
        )
        
        # Message passing: projecting `ref` -> `query`
        self.r2q = (configs.ref, 'to', configs.query)
        self.gat_conv = GATConv(
            (configs.c_hidden, configs.c_hidden),
            configs.c_hidden,
            heads=1,
            concat=False,
            add_self_loops=False
        )

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, x, u, edge_index_dict, edge_index_attr):
        hx = self.g_encoder(x)
        hu = self.m_encoder(u)

        # q(z | x, u)
        h, attn_scores = self.gat_conv(
            (hx, hu), 
            edge_index=edge_index_dict[self.r2q], 
            return_attention_weights=True
        )   
        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)  

        return z_mu, z_logvar, attn_scores 
        

class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()        
        activation = configs.act

        self.z_to_hid = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            activation,
            nn.Dropout(p=configs.dropout)
        )

        self.hid_to_xmu = nn.Linear(configs.c_hidden, configs.c_in)

    def forward(self, z):
        h = self.z_to_hid(z)
        out = self.hid_to_xmu(h)
        return torch.softmax(out, dim=-1)
    
    
class GATDecoder(nn.Module):
    r"""Decoder with paired-modality via GATConv(u -> z -> x)"""
    def __init__(self, configs):
        super().__init__()

        # Message passing: projecting `query` -> `ref`
        self.q2r = (configs.query, 'to', configs.ref)
        self.r2r = (configs.ref, 'to', configs.ref)
        self.gat_conv = GATConv(
            (configs.c_latent, configs.c_hidden), configs.c_latent, edge_dim=1,
            heads=configs.num_heads, concat=False, add_self_loops=False, residual=True
        ) 

        self.summary = GCNConv(configs.c_latent, configs.c_latent)

        self.hid_to_xmu = nn.Linear(configs.c_latent, configs.c_latent)

    def forward(self, z, c, edge_index_dict, edge_attr_dict):
        # c_summary = self.summary(c, edge_index_dict[self.r2r])

        h = self.gat_conv(
            (z, c), 
            edge_index=edge_index_dict[self.q2r],
            # edge_attr=edge_attr_dict[self.q2r]
        )
        return h
        out = self.hid_to_xmu(h)
        return out