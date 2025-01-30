import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist

from torch_sparse import SparseTensor
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch_geometric.nn import GCNConv, SGConv, Linear, Sequential
from pyro.nn import PyroModule, PyroSample


EPS = 1e-8


# --------------------
#  Layer components
# --------------------

class GPCALayer(nn.Module):
    r"""Graph-regularized PCA w/ nonliear activation
    Code reference from:
    - https://arxiv.org/pdf/2006.12294
    - https://github.com/LingxiaoShawn/GPCANet
    """
    def __init__(
        self, c_in, c_out, 
        alpha=1.0, niter=50, act=None, center=True,
        init_weight=True, ortho_weight=False
    ):
        super(GPCALayer, self).__init__()
        self.c_out = c_out
        self.alpha = alpha
        self.niter = niter
        self.center = center
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(1, c_out))
        self.init_weight = init_weight
        self.ortho_weight = ortho_weight
        
        if isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index):
        n = x.shape[0]
        A = self._get_sparse_adj(edge_index, n)
        if self.center:
            x = x - x.mean(dim=0)

        # Compute F = inv(\psi) * x
        invphi_x = self._approx_f(A, x)

        # Compute orthonormal W
        if self.init_weight and self.ortho_weight:
            _, eig_vec = torch.linalg.eigh(x.t().mm(invphi_x))
            eig_vec = torch.real(eig_vec)
            self.weight.data = eig_vec[:, -self.c_out:]
            self.init_weight = False

        # Non-linear activation
        out = self.act(invphi_x.matmul(self.weight) + self.bias)
        return out

    def freeze(self):
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def _get_sparse_adj(self, edge_index, n):
        """Get sym. normalized adj (sparse format)"""
        row, col = edge_index
        A = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
        A = A.set_diag()
        D = A.sum(dim=1).to(torch.float)
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        return D_inv_sqrt.view(-1, 1) * D_inv_sqrt.view(-1, 1) * A
        
    def _approx_f(self, A, x):
        r"""Iterative approx. of F ~ inv(I + \alpha*L) * x"""
        invphi_x = x
        for _ in range(self.niter):
            AF = A.matmul(invphi_x)
            invphi_x = self.alpha/(1+self.alpha)*AF + 1/(1+self.alpha)*x
        return invphi_x
    

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
        super(GCAT, self).__init__()

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
            self.window_embedding = nn.Embedding(num_windows, embed_dim, max_norm=0.5)

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
            hx += self.window_embedding(x_windows)
            hy += self.window_embedding(y_windows)

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
    def __init__(self, configs, device=torch.device('cuda')):
        super(Prior, self).__init__()

        self.u_to_hid = nn.Sequential(
            nn.Linear(configs.c_aux, configs.c_hidden, bias=False),
            configs.act
        )

        if configs.w_init is not None:
            weight = torch.tensor(configs.w_init).to(device).float()
            self.u_to_hid[0].weight = nn.Parameter(weight)

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, u):
        h = self.u_to_hid(u)        
        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)

        return z_mu, z_logvar
    

class AggregatePrior(nn.Module):
    def __init__(self, configs, device=torch.device('cuda')):
        super(AggregatePrior, self).__init__()

        self.x_to_hid = GCNConv(configs.c_aux, configs.c_latent, bias=False)
        self.hid_to_zmu = nn.Linear(configs.c_latent, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_latent, configs.c_latent)

        if configs.w_init is not None:
            weight = torch.tensor(configs.w_init).to(device).float()
            self.x_to_hid.lin.weight = nn.Parameter(weight)
            self.hid_to_zmu.weight = nn.Parameter(torch.eye(configs.c_latent))
        
    def forward(self, x, edge_index, neighbors):
        h = self.x_to_hid(x, edge_index)
        h_pooled = torch.mean(h[neighbors], dim=1)
        
        z_mu = self.hid_to_zmu(h_pooled)
        z_logvar = self.hid_to_zlogvar(h_pooled)

        return z_mu, z_logvar
        

# --------------------------
#  VAE Encoder / Decoders
# --------------------------
    
class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
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
    
    
class AggregateEncoder(nn.Module):
    r"""Encoder with paired modality aggregation by 
    attending `reference (x) to `query` (y) modality
    """
    def __init__(self, configs):
        super(AggregateEncoder,  self).__init__()
        self.attention = GCAT(
            g_dim=configs.c_aux,  # feature dimension for modality `x`
            m_dim=configs.c_in,   # feature dimension for modality `y`
            embed_dim=configs.c_hidden,
            activation=configs.act,
            num_windows=configs.num_windows,
            use_pos=configs.use_pos
        )
        self.act = configs.act

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, x, y, neighbors, x_windows, y_windows):
        h, attn_scores = self.attention(x, y, neighbors, x_windows, y_windows)
        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)
        return z_mu, z_logvar, attn_scores
        

class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()        
        activation = configs.act

        self.z_to_hid = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            activation,
            nn.Dropout(p=configs.dropout)
        )

        self.hid_to_xmu = nn.Linear(configs.c_hidden, configs.c_in)

    def forward(self, z):
        h = self.z_to_hid(z)
        mu = torch.softmax(self.hid_to_xmu(h), dim=-1) + EPS
        return mu
    
    
class AggregateDecoder(nn.Module):
    r"""Decoder with paired-modality aggregations
    via `reference` avg_pooling (z) & gaussian likelihood (y)
    """
    def __init__(self, configs):
        super(AggregateDecoder, self).__init__()
        activation = configs.act
        self.z_to_hid = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            activation,
            nn.Dropout(p=configs.dropout)
        )

        self.hid_to_y_mu = nn.Linear(configs.c_hidden, configs.c_in)
        self.hid_to_y_logvar = nn.Linear(configs.c_hidden, configs.c_in)

    def forward(self, z):
        h = self.z_to_hid(z)
        y_mu = self.hid_to_y_mu(h)
        y_logvar = self.hid_to_y_logvar(h)
        return y_mu, y_logvar
