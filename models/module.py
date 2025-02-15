import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist

from torch_sparse import SparseTensor
from torch_geometric.nn import Sequential
from torch_geometric.nn import GCNConv, GATConv, SGConv


EPS = 1e-8


# --------------------
#  Layer components
# --------------------


class GCAT(nn.Module):
    def __init__(
        self, g_dim,  m_dim, embed_dim, 
        activation, num_windows, use_pos=False
    ):
        r"""
        Graph-based multi-modal Cross Attention
        
        Parameters
        ----------
        g_dim : int
            Feature dimension of modality X (key & value)
        m_dim : int
            Feature dimension of modality Y (query)
        embed_dim : int
            Output embedding dimension
        num_windows : int
            Embedding size for coordinates
        """
        super().__init__()

        self.g_dim = g_dim
        self.m_dim = m_dim
        self.embed_dim = embed_dim
        self.use_pos = use_pos

        self.g_encoder = nn.Sequential(
            nn.Linear(g_dim, embed_dim),
            activation
        )
        self.m_encoder = nn.Sequential(
            nn.Linear(m_dim, embed_dim),
            activation
        )

        if use_pos:
            self.window_embedding = nn.Embedding(num_windows, embed_dim)

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, y, neighbors, x_windows, y_windows):
        r"""Forward pass for graph cross-attention:
        >>> GCAT(X, Y, mask) = knn_mask * CrossAttention(Q, K, V)
        >>> Q = xW_q, K=xW_k, V=yW_v

        Parameters
        ----------
        x : torch.Tensor
            Matrix of fine-resolution (`ref`) modality X (dim [N, G]) 
        y : torch.Tensor
            Matrix of coarse-resolution (`query`) modality Y (dim [L, M]) 
        neighbors : torch.Tensor
            Ref neighbors to each query index  (dim: [L, k])
        x_windows : torch.Tensor
            Patched positional index for `ref` modality (dim: [N])
        y_windows : torch.Tensor
            Patched positional index for `query` modality (dim: [L])

        Returns
        -------
        H : torch.tensor
            Output matrix of dim [L, E].
        """
        hx = self.g_encoder(x)
        hy = self.m_encoder(y)
                
        if self.use_pos:
            hx = hx + self.window_embedding(x_windows)
            hy = hy + self.window_embedding(y_windows)
        
        hx_neighbors = hx[neighbors]
        H, attn_scores = self.get_attention_score(hx_neighbors, hy) # dim: [L, E]
        return H, attn_scores
    
    def get_attention_score(self, x, y):
        # Project X and Y into query, key, and value spaces
        Q = self.query_proj(y).unsqueeze(1)  # dim: [L, 1, E]
        K = self.key_proj(x)   # dim: [k, E]
        V = self.value_proj(x)  # dim: [k, E]

        # Attention scores for each pair of surjective cell-pixel map
        scale = self.embed_dim ** 0.5
        raw_scores = torch.bmm(Q, torch.transpose(K, 1, 2)) / scale  # dim: [L, 1, k]
        scores = F.softmax(raw_scores, dim=2) # dim: [L, 1, k]
        H = torch.bmm(scores, V) # 1,K @ K,E -> E dim: [L, 1, E]

        return H.squeeze(1), scores.squeeze(1) # remove sequence dimension since length 1
    
    
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


class PhenotypePrior(nn.Module):
    r"""High-dim cell-type aware prior p(v | c)"""
    def __init__(self, configs):
        super().__init__()
        
        self.c_to_hid = nn.Sequential(
            nn.Linear(configs.num_latent, configs.c_latent),
            configs.act
        )

        self.hid_to_vmu = nn.Linear(configs.c_latent, configs.c_hidden)
        self.hid_to_vlogvar = nn.Linear(configs.c_latent, configs.c_hidden)

    def forward(self, c):
        h = self.c_to_hid(c)
        v_mu = self.hid_to_vmu(h)
        v_logvar = self.hid_to_vlogvar(h)

        return v_mu, v_logvar


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
    attending `ref` (x) to `query` (u) w/ GAT -> latent (z)
    """
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act

        # self.g_encoder = nn.Sequential(
        #     nn.Linear(configs.c_in, configs.c_hidden),
        #     configs.act
        # )      

        self.m_encoder = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden),
            configs.act
        )
        
        # Message passing: projecting `ref` -> `query`
        self.r2q = (configs.ref, 'to', configs.query)
        self.gat_conv = GATConv(
            # (configs.c_hidden, configs.c_hidden), configs.c_hidden, edge_dim=1,
            (configs.c_hidden, configs.c_hidden), configs.c_hidden, edge_dim=1,
            heads=configs.num_heads, concat=False, add_self_loops=False,
        )

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, v, u, edge_index_dict):
        # hx = self.g_encoder(x)
        hu = self.m_encoder(u)

        # q(z | x, u)
        h, attn_scores = self.gat_conv(
            (v, hu), 
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
        self.gat_conv = GATConv(
            (configs.c_latent, configs.c_latent), configs.c_hidden,
            heads=configs.num_heads, concat=False, add_self_loops=False, residual=True
        ) 

        self.hid_to_xmu = nn.Linear(configs.c_hidden, configs.c_in)

    def forward(self, z, v, edge_index_dict):
        h = self.gat_conv((z, v), edge_index=edge_index_dict[self.q2r])
        out = self.hid_to_xmu(h)
        return torch.softmax(out, dim=-1)