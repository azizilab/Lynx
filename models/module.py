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

        # if configs.w_init is not None:
        #     weight = torch.tensor(configs.w_init).to(device).float()
        #     self.u_to_hid[0].weight = nn.Parameter(weight)

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

        # TODO: ablade # layers
        self.x_encoder = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            configs.act
        )      

        self.u_encoder = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
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
        hx = self.x_encoder(x)
        hu = self.u_encoder(u)

        # q(z | x, u)
        h, attn_scores = self.gat_conv(
            (hx, hu), 
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
            configs.act
            # nn.Linear(configs.c_hidden, configs.c_hidden),
            # self.act,
        )  
        
        self.hid_to_vmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_vlogvar =  nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, x):
        h = self.x_to_hid(x)
        v_mu = self.hid_to_vmu(h)
        v_logvar = self.hid_to_vlogvar(h)
        
        return v_mu, v_logvar
    

class XtoOmegaEncoder(nn.Module):
    r"""Encode `ref` (x) level attention weights (omega) via edge embedding"""
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act
        self.r2r = (configs.ref, 'to', configs.ref)

        self.x_to_hid = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            configs.act
        )  

        # TODO: ablade # layers
        self.edge_to_omega = nn.Sequential(
            nn.Linear(configs.c_hidden*2, configs.c_hidden),  # concat(src_embedding, dst_embedding)
            configs.act,
            nn.Linear(configs.c_hidden, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, 2)
        )

    def forward(self, x, edge_index_dict, edge_attr_dict):
        # TODO: inconsistent usage of edge_distance in generative & inference paths, 
        # try both scale by dist? and is it dist or weights?

        edge_index = edge_index_dict[self.r2r]
        edge_dist = edge_attr_dict[self.r2r]
        src, dst = edge_index  # source & target edge indices

        h = self.x_to_hid(x)
        h_src, h_dst = h[src], h[dst]

        edge_feats = torch.cat([h_src, h_dst], dim=-1) 
        omegas = self.edge_to_omega(edge_feats)
        loc = omegas[:, 0] * edge_dist
        scale = omegas[:, 1].exp() + EPS

        return loc, scale


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
            nn.Linear(configs.c_latent*2, configs.c_latent),  # concat(src_embedding, dst_embedding)
            configs.act,
            nn.Linear(configs.c_latent, 2)
        )

    def forward(self, z, c, edge_index_dict, edge_attr_dict):
        s = self.z_to_s((z, c), edge_index_dict[self.q2r])  # unpooled `z` from query-level -> ref-level
        edge_index = edge_index_dict[self.r2r]
        edge_dist = edge_attr_dict[self.r2r]
        src, dst = edge_index  # source & target edge indices

        # Concat the cell-type embeddings from src & dst nodes
        s_src, s_dst = s[src], s[dst]
        edge_feats = torch.cat([s_src, s_dst], dim=-1) # shape [|E|, 4*c_latent]
        omegas = self.edge_to_omega(edge_feats)
        loc = omegas[:, 0] * edge_dist  # scale by distance; TODO: unify generative & inference
        scale = omegas[:, 1].exp() + EPS

        return s, loc, scale
    

class ZtoVDecoder(nn.Module):
    r"""Decode ref-level (x) phenotype embedding via 
    sampled attention: v ~ p(v | z, c, norm(omega))
    """
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act
        self.r2r = (configs.ref, 'to', configs.ref)

        self.c_to_proj = nn.Linear(configs.c_latent, configs.c_latent)
        self.hid_to_vmu = nn.Linear(configs.c_latent, configs.c_latent)
        self.hid_to_vlogvar = nn.Linear(configs.c_latent, configs.c_latent)

    def forward(self, s, W_ij, edge_index_dict):
        src, dst = edge_index_dict[self.r2r]  # source & target edge indices
        feats = self.c_to_proj(s)  # Projection on unpooled `z`
        feats_src = feats[src]
        
        weighted_edges = W_ij.unsqueeze(-1) * feats_src # shape [|E|, c_latent]
        v_hid = torch_scatter.scatter_add(weighted_edges, dst, dim=0, dim_size=s.size(0))  # "Attended" values
        v_hid = self.act(v_hid)

        v_mu = self.hid_to_vmu(v_hid)
        v_logvar = self.hid_to_vlogvar(v_logvar)

        return v_mu, v_logvar
        

        




class GATDecoder(nn.Module):
    r"""Decoder with paired-modality via GATConv(u -> z -> x)"""
    def __init__(self, configs):
        super().__init__()

        # Message passing: projecting `query` -> `ref`
        self.q2r = (configs.query, 'to', configs.ref)
        self.r2r = (configs.ref, 'to', configs.ref)
        self.gat_conv = GATConv(
            (configs.c_latent, configs.c_latent), configs.c_latent, edge_dim=1,
            heads=configs.num_heads, concat=False, add_self_loops=False, residual=False
        ) 

        self.hid_to_xmu = nn.Linear(configs.c_hidden, configs.c_in)

    def forward(self, z, v, edge_index_dict, edge_attr_dict):
        h = self.gat_conv(
            (z, v), 
            edge_index=edge_index_dict[self.q2r],
            # edge_attr=edge_attr_dict[self.q2r]
        )
        return h
        out = self.hid_to_xmu(h)
        return out