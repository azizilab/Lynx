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
    def __init__(self, g_dim,  m_dim, embed_dim):
        r"""Graph-based multi-modal Cross Attention"""
        super(GCAT, self).__init__()

        self.g_dim = g_dim
        self.m_dim = m_dim
        self.embed_dim = embed_dim

        self.g_encoder = nn.Sequential(
            nn.Linear(g_dim, embed_dim),
            nn.ReLU()
        )
        self.m_encoder = nn.Sequential(
            nn.Linear(m_dim, embed_dim),
            nn.ReLU()
        )

        # self.out_proj = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU()
        # )

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, X, Y, neighbors):
        r"""Forward pass for surjective attention:
        >>> SurjectiveAttention(X, Y, mask) = mask * CrossAttention(Q, K, V)
        >>> Q = YW_q, K=XW_k, V=XW_v

        Parameters
        ----------
        X : torch.Tensor
            Matrix of fine-resolution modality X (dim [N, G]) 
        Y : torch.Tensor
            Matrix of coarse-resolution modality Y (dim [L, M]) 
        neighbors : torch.Tensor
            Neighbor index of fine-resolution graph (dim: [L, K])
            
        Returns
        -------
        H : torch.tensor
            Output matrix of dim [L, E].
        """

        H, attn_scores = self.get_attention_score(X, Y, neighbors) # dim: [L, E]
        
        return H, attn_scores
    
    def get_attention_score(self, X, Y, neighbors):

        X_h = self.g_encoder(X)
        Y_h = self.m_encoder(Y)      

        X_neighbors = X_h[neighbors]

        # Project X and Y into query, key, and value spaces
        Q = self.query_proj(Y_h).unsqueeze(1)  # dim: [L, 1, E]
        K = self.key_proj(X_neighbors)   # dim: [L, K, E]
        V = self.value_proj(X_neighbors)  # dim: [L, K, E]

        # Attention scores for each pair of surjective cell-pixel map
        raw_scores = torch.bmm(Q, torch.transpose(K, 1, 2))  # dim: [L, 1, K]

        scores = F.softmax(raw_scores, dim=2) # dim: [L, 1, K]

        H = torch.bmm(scores, V) # 1,K @ K,E -> E dim: [L, 1, E]

        # H_out = self.out_proj(H)

        return H.squeeze(1), scores.squeeze(1) #remove sequence dimension since length 1
    

# --------------------------------
#  VAE Prior / Posterior Modules
# --------------------------------

class ConditionalPrior(nn.Module):
    def __init__(self, configs, device=torch.device('cuda')):
        super(ConditionalPrior, self).__init__()

        self.x_to_hid = GPCALayer(
            configs.c_aux, configs.c_hidden,
            act=configs.act, ortho_weight=True
        )

        # self.x_to_hid = Sequential('x, edge_index', [
        #     (GCNConv(configs.c_in, configs.c_hidden, bias=False), 'x,edge_index -> h'),
        #     configs.act
        # ])

        # if configs.w_init is not None:
        #     weight = torch.tensor(configs.w_init).to(device).float()
        #     self.x_to_hid[0].lin.weight = nn.Parameter(weight)

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, x, edge_index, neighbors):
        h = self.x_to_hid(x, edge_index)
        h_pooled = torch.mean(h[neighbors], dim=1)
        
        z_mu = self.hid_to_zmu(h_pooled)
        z_logvar = self.hid_to_zlogvar(h_pooled)

        return z_mu, z_logvar
        
    
class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder,  self).__init__()
        self.embed_option = configs.embed_option
        activation = configs.act
        
        self.num_heads = configs.num_heads
        self.c_embedding = configs.c_embedding
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

    def forward(self, x, u, s, edge_index):
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
            g_dim=configs.c_aux,  # X
            m_dim=configs.c_in,   # Y
            embed_dim=configs.c_hidden
        )
        self.act = configs.act

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, x, y, neighbors):
        h, attn_scores = self.attention(
            x, y, neighbors
        )
        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)
        return z_mu, z_logvar, attn_scores
        

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

        self.hid_to_xmu = nn.Linear(c_hid, configs.c_in)

    def forward(self, z, s, edge_index):
        h = self.z_to_hid(z, edge_index)
        hs = torch.cat([h, s], dim=-1)
        mu = torch.softmax(self.hid_to_xmu(hs), dim=-1) + EPS
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


class SpikeSlabLassoDecoder(PyroModule):
    r"""Decoder with a spike-and-slab prior on the weights from z -> hidden. 
    That encourages *sparse* usage of latent dimensions.
    """
    def __init__(self, configs, device):
        super().__init__()
        self.configs = configs
        activation = configs.act
        self.device = device
        

        self.z_to_hid = PyroModule[nn.Linear](configs.c_latent, configs.c_hidden)
        
        # --- Place a spike-and-slab prior on the weight ---
        # Creates pyro sample "z_to_hid.weight"
        self.z_to_hid.weight = PyroSample(
            prior=self.spike_slab_prior(
                shape=(configs.c_hidden, configs.c_latent), 
                spike_std=configs.spike_std, 
                slab_std=configs.slab_std, 
                mixture_prob=configs.mixture_prob,
            )
        )
        
        # Simple Normal prior on the bias
        # Creates pyro sample "z_to_hid.bias"
        self.z_to_hid.bias = PyroSample(
            dist.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device)).expand([configs.c_hidden]).to_event(1)
        )
        
        self.activation = activation
        
        self.hid_to_y_mu = nn.Linear(configs.c_hidden, configs.c_in)
        self.hid_to_y_logvar = nn.Linear(configs.c_hidden, configs.c_in)

    def forward(self, z):
        """
        The spike-and-slab applies only to z->hid's weights.
        The subsequent layers have no special prior.
        """
        h = self.activation(self.z_to_hid(z))
        y_mu = F.relu(self.hid_to_y_mu(h))
        y_logvar = self.hid_to_y_logvar(h)
        return y_mu, y_logvar
    

    def spike_slab_prior(
        self,
        shape,
        spike_std=1e-2,
        slab_std=1.0,
        mixture_prob=0.5,
    ):
        """
        Returns a MixtureSameFamily distribution that places a 'spike' (low-variance)
        and a 'slab' (higher-variance) component over parameters.
        
        shape: the shape of the parameter we are putting a prior over.
        spike_std: std-dev of the 'spike' component near zero
        slab_std: std-dev of the 'slab' component
        mixture_prob: probability of the spike vs. slab component
        """
        loc = torch.zeros((2,) + shape, device=self.device)
        scale = torch.zeros((2,) + shape, device=self.device)

        # fill first row with spike_std, second row with slab_std
        scale[0] = spike_std
        scale[1] = slab_std

        # A single Normal distribution with batch_shape=[2, *shape].
        component_dist = dist.Normal(loc, scale)

        # We want the entire 'shape' to be considered as the event dimensions,
        # leaving the first dimension (2) as the mixture dimension. So:
        component_dist = dist.Independent(component_dist, reinterpreted_batch_ndims=len(shape))
        # Now component_dist has batch_shape=[2], event_shape=shape.

        # Mixture probabilities
        mixture = dist.Categorical(torch.tensor([mixture_prob, 1.0 - mixture_prob], device=self.device))

        # Finally wrap as a MixtureSameFamily with 2 components (spike, slab)
        # The mixture dimension is the leftover batch dim of size 2.
        spike_slab_dist = dist.MixtureSameFamily(mixture, component_dist)

        return spike_slab_dist

