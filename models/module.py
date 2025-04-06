import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist

import torch_scatter
from torch_geometric.nn import Sequential
from torch_geometric.nn import GCNConv, GATConv, SGConv


EPS = 1e-8


# ---------------------
#  VAE Prior Modules
# ---------------------

class Prior(nn.Module):
    r"""Low-dim conditional prior"""
    def __init__(self, configs):
        super().__init__()

        self.u_to_hid = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, configs.c_hidden),
            configs.act
        )

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

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


class XtoZEncoder(nn.Module):
    r"""Encode paired modality via GAT by attending 
    `ref` (x) to `query` (u): z ~ q(z | x, u)
    """
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act
        self.x_to_hid = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            configs.act,
        )      
        self.u_to_hid = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            configs.act,
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
        # q(z | x, u)
        x = self.x_to_hid(x)
        u = self.u_to_hid(u)

        h, attn_scores = self.gat_conv(
            (x, u), 
            edge_index=edge_index_dict[self.r2q], 
            return_attention_weights=True
        )   
        h = self.act(h)

        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)  

        return z_mu, z_logvar, attn_scores 
    

class XtoVEncoder(nn.Module):
    r"""Encode phenotype latent v ~ q(v | x)"""
    def __init__(self, configs):
        super().__init__()        
        self.x_to_hid = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            configs.act,
        )      
        self.hid_to_vmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_vlogvar =  nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, x):
        x = self.x_to_hid(x)
        v_mu = self.hid_to_vmu(x)
        v_logvar = self.hid_to_vlogvar(x)
        
        return v_mu, v_logvar
    

class XtoOmegaEncoder(nn.Module):
    r"""Encode `ref` (x) level attention weights (omega) via edge embedding"""
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act
        self.x_to_hid = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            configs.act,
        )      

        self.r2r = (configs.ref, 'to', configs.ref)
        self.edge_to_omega = nn.Sequential(
            nn.Linear(configs.c_hidden*2+1, configs.c_hidden),  # concat(src_embedding, dst_embedding)
            configs.act,
            nn.Linear(configs.c_hidden, configs.c_latent),
            configs.act,
            nn.Linear(configs.c_latent, 2),
            nn.Softplus()
        )

    def forward(self, x, edge_index_dict, edge_attr_dict):
        x = self.x_to_hid(x)
        edge_index = edge_index_dict[self.r2r]
        edge_dist = edge_attr_dict[self.r2r].unsqueeze(-1) 
        src, dst = edge_index  # source & target edge indices

        x_src, x_dst = x[src], x[dst]
        edge_feats = torch.cat([x_src, x_dst, edge_dist], dim=-1) 

        # Weibull scale (lambda) & shape (k)
        omegas = self.edge_to_omega(edge_feats)
        lambda_ = omegas[:, 0] + EPS
        k = omegas[:, 1] + EPS

        return lambda_, k


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
    

class ZtoOmegaDecoder(nn.Module):
    r"""Decode ref-level (x) attention weights (omega) by attending 
    query-level (u) to ref-level cell types (c): omega ~ p(omega | z, c)
    """
    def __init__(self, configs):
        super().__init__()
        self.q2r = (configs.query, 'to', configs.ref)
        self.r2r = (configs.ref, 'to', configs.ref)

        self.z_to_s = GATConv(
            (configs.c_latent, configs.c_latent), configs.c_latent,
            heads=1, concat=False, add_self_loops=False, residual=False
        )

        self.edge_to_omega = nn.Sequential(
            nn.Linear(configs.c_latent*2+1, configs.c_latent),  # concat(src_embedding, dst_embedding)
            configs.act,
            nn.Dropout(p=configs.dropout),
            nn.Linear(configs.c_latent, 2),
            nn.Softplus()
        )

        self.celltype_aware = configs.celltype_aware

    def forward(self, z, c, edge_index_dict, edge_attr_dict):
        # Ablation: unpool z conditional on c vs. avg. unpool
        if self.celltype_aware:
            s = self.z_to_s((z, c), edge_index_dict[self.q2r])  # unpooled `z` from query-level -> ref-level
        else:
            q2r_src, q2r_dst = edge_index_dict[self.q2r]  # source & target edge indices (query-target graph)
            s = torch_scatter.scatter_mean(z[q2r_src], q2r_dst, dim=0, dim_size=c.size(0))

        r2r_src, r2r_dst = edge_index_dict[self.r2r]  # source & target edge indices (ref-ref graph)
        r2r_ew = edge_attr_dict[self.r2r].unsqueeze(-1) 

        # Concat the cell-type embeddings from src & dst nodes
        edge_feats = torch.cat([s[r2r_src], s[r2r_dst], r2r_ew], dim=-1) # shape [|E|, 4*c_latent]

        # Gamma shape (alpha) & rate (beta)
        omegas = self.edge_to_omega(edge_feats)
        alpha = omegas[:, 0] + EPS # * edge_dist  # scale by distance
        beta = omegas[:, 1] + EPS

        return s, alpha, beta
    

class ZtoVDecoder(nn.Module):
    r"""Decode ref-level (x) phenotype embedding via 
    sampled attention: v ~ p(v | z, c, norm(omega))
    """
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act
        self.r2r = (configs.ref, 'to', configs.ref)
        self.hid_to_vmu = nn.Linear(configs.c_latent, configs.c_latent)
        self.hid_to_vlogvar = nn.Linear(configs.c_latent, configs.c_latent)

    def forward(self, s, W_ij, edge_index_dict):
        src, dst = edge_index_dict[self.r2r]  # source & target edge indices
        feats_src = s[src]
        
        weighted_edges = W_ij.unsqueeze(-1) * feats_src # shape [|E|, c_latent]
        v_hid = self.act(
            torch_scatter.scatter_add(weighted_edges, dst, dim=0, dim_size=s.size(0))
        )  # "Attended" values

        v_mu = self.hid_to_vmu(v_hid)
        v_logvar = self.hid_to_vlogvar(v_hid) 
        return v_mu, v_logvar

