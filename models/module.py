import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch_geometric.nn import Linear, Sequential
from torch_geometric.nn import GCNConv, GATConv, HeteroConv, SGConv

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
    

class HeteroGNN(nn.Module):
    def __init__(self, configs):
        super(HeteroGNN, self).__init__()
        self.edge_label = configs.edge_label
        self.query = configs.edge_label[-1]

        self.conv = HeteroConv({
            self.edge_label: GATConv((-1, -1), configs.c_hidden, add_self_loops=False),
        }, aggr='sum')
        self.lin = Linear(-1, configs.c_hidden)
        self.act = configs.act

    def forward(self, x_dict, edge_index_dict):
        assert self.query in x_dict.keys()
        x_adj = self.conv(x_dict, edge_index_dict)[self.query]
        x_self = self.lin(x_dict[self.query])
        out = self.act(x_self + x_adj)
        return out

# --------------------------------
#  VAE Prior / Posterior Modules
# --------------------------------

class ConditionalPrior(nn.Module):
    def __init__(self, configs, device=torch.device('cuda')):
        super(ConditionalPrior, self).__init__()
        self.c_latent = configs.c_latent
        self.u_to_zmu = nn.Linear(configs.c_aux, configs.c_latent, bias=False)
        self.u_to_zlogvar = nn.Linear(configs.c_aux, configs.c_latent)
        
        if configs.w_init is not None:
            weight = torch.tensor(configs.w_init).to(device).float()
            self.u_to_zmu.weight = nn.Parameter(weight)

    def forward(self, u):
        z_mu = self.u_to_zmu(u)
        z_logvar = self.u_to_zlogvar(u)
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
            hu = self.u_to_hid(u, edge_index)
            h = torch.cat([hx, hu], dim=-1)

            z_mu = self.hid_to_zmu(h, edge_index)
            z_logvar = self.hid_to_zlogvar(h, edge_index)

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

    
class AggregateEncoder(nn.Module):
    r"""Encoder with paired modality aggregation by 
    attending `reference (x) to `query` (y) modality
    """
    def __init__(self, configs):
        super(AggregateEncoder,  self).__init__()
        self.attention = HeteroGNN(configs)
        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, x_dict, edge_index_dict):
        h = self.attention(x_dict, edge_index_dict)
        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)
        return z_mu, z_logvar
        

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
        c_hid = configs.c_hidden + configs.c_covariate
        activation = configs.act
        self.z_to_hid = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            activation,
            nn.Dropout(p=configs.dropout)
        )

        self.hid_to_ymu = nn.Linear(c_hid, configs.c_in)
        self.hid_to_ylogvar = nn.Linear(c_hid, configs.c_in)

    def forward(self, z):
        hid = self.z_to_hid(z)
        y_mu = self.hid_to_ymu(hid)
        y_logvar = self.hid_to_ylogvar(hid)
        return y_mu, y_logvar