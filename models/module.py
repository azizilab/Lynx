import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist
import pyro

import torch_scatter
from torch_geometric.nn import Sequential
from torch_geometric.nn import GCNConv, GATConv, SGConv
from torch_geometric.utils import to_dense_adj


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
            configs.act
        )

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, u, edge_index_dict):
        h = self.u_to_hid(u)        
        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)

        return z_mu, z_logvar
    

class StructuralPrior(nn.Module):
    r"""Low-dim conditional prior"""
    def __init__(self, configs):
        super().__init__()
        self.q2q = (configs.query, 'to', configs.query)
        self.u_to_zs = nn.ModuleList([
            Sequential('u, edge_index', [
                (GCNConv(configs.c_aux, configs.c_hidden), 'u, edge_index -> u'),
                 configs.act,
                (GCNConv(configs.c_hidden, 2), 'u, edge_index -> u')
            ])
            for _ in range(configs.c_latent)
        ])
        
        # iid zero-mean Gaussian initialization on weights
        for layer in self.u_to_zs:
            nn.init.normal_(layer[0].lin.weight, mean=0., std=1./configs.c_latent)
            nn.init.normal_(layer[0].bias, mean=0., std=0.1)
            nn.init.normal_(layer[-1].lin.weight, mean=0., std=1./configs.c_latent)
            nn.init.normal_(layer[-1].bias, mean=0., std=0.1)

    def forward(self, u, edge_index_dict):
        z_mus = []
        z_logvars = []
        for layer in self.u_to_zs:
            z_mu_d, z_logvar_d = layer(u, edge_index_dict[self.q2q]).T
            z_mus.append(z_mu_d)
            z_logvars.append(z_logvar_d)
        
        z_mu = torch.stack(z_mus, dim=-1)
        z_logvar = torch.stack(z_logvars, dim=-1)
        return z_mu, z_logvar

    
class ConvPrior(nn.Module):
    r"""Convolutional prior p(z | u) for histology image patches"""
    
    def __init__(self, configs):
        super().__init__()
        
        self.patch_size = configs.patch_size if hasattr(configs, 'patch_size') else 64
        
        # Simple CNN encoder for image patches
        self.conv_encoder = nn.Sequential(
            # First conv block: (3, P, P) -> (32, P/2, P/2)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            configs.act,
            
            # Second conv block: (32, P/2, P/2) -> (64, P/4, P/4)  
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            configs.act,
                        
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Project to hidden dimension
        self.u_to_hid = nn.Sequential(
            nn.Linear(64, configs.c_hidden),
            configs.act,
        )
        
        # Output layers for z distribution
        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)
    
    def forward(self, u, edge_index_dict):
        """
        Parameters:
        -----------
        u : torch.Tensor, shape (N, 3, P, P)
            Image patches (already reshaped from flattened format)
        edge_index_dict : dict
            Edge indices (not used in this implementation)
        """
        
        # Encode image patches
        h_conv = self.conv_encoder(u)  # (N, 128)
        h = self.u_to_hid(h_conv)      # (N, c_hidden)
        
        # Get latent distribution parameters
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
        
        self.x_to_hid = Sequential('x, edge_index', [
            (GCNConv(configs.c_in, configs.c_hidden), 'x, edge_index -> h'),
            activation, 
        ])
        
        self.u_to_hid = Sequential('u, edge_index', [
            (GCNConv(configs.c_aux, configs.c_hidden), 'u, edge_index -> h'),
            activation
        ])

        self.hid_to_zmu = nn.Linear(configs.c_hidden*2, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden*2, configs.c_latent)

    def forward(self, x, u, edge_index):
        hx = self.x_to_hid(x, edge_index)
        hu = self.u_to_hid(u, edge_index)
        h = torch.cat([hx, hu], dim=-1)

        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)        
        return z_mu, z_logvar


class XtoZEncoder(nn.Module):
    r"""Encode paired modality via GAT by attending 
    `ref` (x) to `query` (u): z ~ q(z | x, u)
    """
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act
        self.r2r = (configs.ref, 'to', configs.ref)
        self.q2q = (configs.query, 'to', configs.query)
        self.r2q = (configs.ref, 'to', configs.query)

        self.x_to_hid = Sequential('x, edge_index', [
            (GCNConv(configs.c_in, configs.c_hidden), 'x, edge_index -> x'),
            configs.act
        ])   
        self.u_to_hid = Sequential('u, edge_index', [
            (GCNConv(configs.c_aux, configs.c_hidden), 'u, edge_index -> u'),
            configs.act
        ])  
        
        # Message passing: projecting `ref` -> `query`
        self.gat_conv = GATConv(
            (configs.c_hidden, configs.c_hidden),
            configs.c_hidden,
            heads=1,
            concat=False,
            add_self_loops=False
        )

        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)

    def forward(self, x, u, edge_index_dict):
        # q(z | x, u)
        x = self.x_to_hid(x, edge_index_dict[self.r2r])
        u = self.u_to_hid(u, edge_index_dict[self.q2q])
        
        h, attn_scores = self.gat_conv(
            (x, u), 
            edge_index=edge_index_dict[self.r2q], 
            return_attention_weights=True
        )   
        h = self.act(h)

        z_mu = self.hid_to_zmu(h)
        z_logvar = self.hid_to_zlogvar(h)  

        return z_mu, z_logvar, attn_scores 

    
class ConvXtoZEncoder(nn.Module):
    r"""Convolutional encoder for q(z | x, u) using image patches and genomic data"""
    
    def __init__(self, configs):
        super().__init__()
        
        self.act = configs.act
        self.r2r = (configs.ref, 'to', configs.ref)
        self.r2q = (configs.ref, 'to', configs.query)
        
        self.patch_size = configs.patch_size if hasattr(configs, 'patch_size') else 64
        
        # GCN encoder for genomic data (x)
        from torch_geometric.nn import Sequential
        self.x_to_hid = Sequential('x, edge_index', [
            (GCNConv(configs.c_in, configs.c_hidden), 'x, edge_index -> x'),
            configs.act
        ])
        
        # CNN encoder for image patches (u) - same as in ConvPrior
        self.conv_encoder = nn.Sequential(
            # First conv block: (3, P, P) -> (32, P/2, P/2)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            configs.act,
            
            # Second conv block: (32, P/2, P/2) -> (64, P/4, P/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            configs.act,
                        
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Project CNN features to hidden dimension
        self.u_to_hid = nn.Sequential(
            nn.Linear(64, configs.c_hidden),
            configs.act,
        )
        
        # Cross-modal attention: genomic -> histology
        self.gat_conv = GATConv(
            (configs.c_hidden, configs.c_hidden),
            configs.c_hidden,
            heads=1,
            concat=False,
            add_self_loops=False
        )
        
        # Output layers for z distribution
        self.hid_to_zmu = nn.Linear(configs.c_hidden, configs.c_latent)
        self.hid_to_zlogvar = nn.Linear(configs.c_hidden, configs.c_latent)
    
    def forward(self, x, u, edge_index_dict):
        """
        Parameters:
        -----------
        x : torch.Tensor, shape (N_ref, genes)
            Genomic data
        u : torch.Tensor, shape (N_query, 3, P, P)  
            Image patches (already reshaped from flattened format)
        edge_index_dict : dict
            Edge connectivity
        edge_index_attr : dict
            Edge attributes
        
        Returns:
        --------
        z_mu : torch.Tensor, shape (N_query, c_latent)
        z_logvar : torch.Tensor, shape (N_query, c_latent) 
        attn_scores : tuple
            Attention weights from GAT
        """
        
        # Encode genomic data with GCN
        x_hidden = self.x_to_hid(x, edge_index_dict[self.r2r])  # (N_ref, c_hidden)
        
        # Encode image patches with CNN
        u_conv = self.conv_encoder(u)       # (N_query, 128)
        u_hidden = self.u_to_hid(u_conv)    # (N_query, c_hidden)
        
        # Cross-modal attention: project genomic -> histology space
        h, attn_scores = self.gat_conv(
            (x_hidden, u_hidden),
            edge_index=edge_index_dict[self.r2q],
            return_attention_weights=True
        )   # (N_query, c_hidden)
        
        h = self.act(h)
        
        # Get latent distribution parameters
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

        self.x_to_hid = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            configs.act,
        )      
        self.r2r = (configs.ref, 'to', configs.ref)
        self.edge_to_omega = nn.Sequential(
            nn.Linear(configs.c_hidden*2, configs.c_hidden),  
            configs.act,
            nn.Linear(configs.c_hidden, 2)
        )

    def forward(self, x, edge_index_dict, edge_attr_dict):
        x = self.x_to_hid(x)
        edge_index = edge_index_dict[self.r2r]
        
        src, dst = edge_index  # source & target edge indices
        x_src, x_dst = x[src], x[dst]
        edge_feats = torch.cat([x_src, x_dst], dim=-1)

        omegas = self.edge_to_omega(edge_feats)
        loc = omegas[:, 0]
        scale = F.softplus(omegas[:, 1]) + EPS

        return loc, scale
    
class XtoOmegaCluEncoder(nn.Module):
    r"""Encode `ref` (x) level attention weights (omega) via edge embedding"""
    def __init__(self, configs):
        super().__init__()
        self.r2r = (configs.ref, 'to', configs.ref)

        # Projection layers for edge feature (logits) embeddings
        self.src_to_hid = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            configs.act
        )
        self.dst_to_hid = nn.Sequential(
            nn.Linear(configs.c_in, configs.c_hidden),
            configs.act
        )
        self.hid_to_logits = nn.Sequential(
            nn.Linear(configs.c_hidden*2, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, 1)
        )

    def forward(self, x, edge_index_dict, edge_attr_dict):        
        edge_index = edge_index_dict[self.r2r]
        src, dst = edge_index  # source & target edge indices

        # Concat neighbor edge weights & cluster-specific weights
        src_emb = self.src_to_hid(x[src])  # (E, c_hidden)
        dst_emb = self.dst_to_hid(x[dst])  # (E, c_hidden)
        edge_emb = torch.cat([dst_emb, src_emb], dim=-1)

        # Final projection to get edge weights
        logits = self.hid_to_logits(edge_emb).squeeze(-1)  # (E,)
        return logits

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
    

class ZtoSDecoder(nn.Module):
    r"""Convolve "pixel"(query)-level latent z to "cell"(ref)-level latent s
    """
    def __init__(self, configs):
        super().__init__()
        self.q2r = (configs.query, 'to', configs.ref)
        self.r2r = (configs.ref, 'to', configs.ref)
        
        self.z_to_s = GATConv(
            (configs.c_latent, configs.c_latent), configs.c_latent,
            heads=1, concat=False, add_self_loops=False, residual=False
        )

    def forward(self, z, c, edge_index_dict, celltype_aware=False):
        if celltype_aware:
            # "Unpooling" conditioned on cluster embedding
            s = self.z_to_s((z, c), edge_index_dict[self.q2r])
        else:
            # "Unpooling" w/o conditioning on cluster embedding
            q2r_src, q2r_dst = edge_index_dict[self.q2r]
            s = torch_scatter.scatter_mean(z[q2r_src], q2r_dst, dim=0, dim_size=c.size(0))
        return s


    
class ZtoVDecoder(nn.Module):
    r"""Decode ref-level (x) phenotype embedding via 
    sampled attention: v ~ p(v | z, c, \omega)
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


class ZtoXDecoder(nn.Module):
    r"""
    Decoder ref-level (x) expressions directly via sampled attention (v) 
    & unpooled latent representation (s): x ~ p(x | s, c, \omega)
    """
    def __init__(self, configs):
        super().__init__()
        self.act = configs.act
        self.r2r = (configs.ref, 'to', configs.ref)

        self.v_to_x = nn.Sequential(
            nn.Linear(configs.c_latent, configs.c_hidden),
            self.act,
            nn.Dropout(p=configs.dropout),
            nn.Linear(configs.c_hidden, configs.c_in)
        )
        
    def forward(self, s, W_ij, edge_index_dict):
        src, dst = edge_index_dict[self.r2r]  # source & target edge indices
        feats_src = s[src]
        weighted_edges = W_ij.unsqueeze(-1) * feats_src  # shape: [|E|, c_latent]
        v = self.act(torch_scatter.scatter_add(weighted_edges, dst, dim=0, dim_size=s.size(0)))  # Attended values
        x = self.v_to_x(v)    

        return torch.softmax(x, dim=-1)
