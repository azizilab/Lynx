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

        # self.x_to_hid = nn.Sequential(
        #     nn.Linear(configs.c_in, configs.c_hidden),
        #     configs.act,
        # )      
        # self.u_to_hid = nn.Sequential(
        #     nn.Linear(configs.c_aux, configs.c_hidden),
        #     configs.act,
        # )   

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

    def forward(self, x, u, edge_index_dict, edge_index_attr):
        # q(z | x, u)
        # x = self.x_to_hid(x)
        # u = self.u_to_hid(u)
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
        self.q2q = (configs.query, 'to', configs.query) 
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
    
    def forward(self, x, u, edge_index_dict, edge_index_attr):
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
        self.n = configs.n
        self.num_clusters = configs.num_clusters

        def __make_edge_feat(c_in, c_hidden, act):
            return nn.Sequential(
                nn.Linear(c_in, c_hidden),
                act,
                nn.Linear(c_hidden, c_hidden),
                # nn.LayerNorm(configs.c_hidden),
            )
        
        self.source_mlp = __make_edge_feat(configs.c_in, configs.c_hidden, configs.act)
        self.target_mlp = __make_edge_feat(configs.c_in, configs.c_hidden, configs.act)
        # self.target_bulk_mlp = __make_edge_feat(configs.c_in, configs.c_hidden, configs.act)
        # self.bulk_mlp = __make_edge_feat(configs.c_in, configs.c_hidden, configs.act)

        self.pi_mlp = nn.Sequential(
            nn.Linear(configs.c_hidden + configs.c_hidden + 1, configs.c_hidden),
            # nn.LayerNorm(configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, configs.c_hidden),
            configs.act,
            nn.Linear(configs.c_hidden, 1),
        )

        # self.pi_bulk_mlp = nn.Sequential(
        #     nn.Linear(configs.c_in, configs.c_hidden),
        #     nn.LayerNorm(configs.c_hidden),
        #     configs.act,
        #     nn.Linear(configs.c_hidden, 1),
        # )

    def forward(self, x, clusters, edge_index_dict, edge_attr_dict):
        edge_index = edge_index_dict[self.r2r]
        src, dst = edge_index  # source & target edge indices
        edge_attr = edge_attr_dict[self.r2r]

        #x to x feat
        edge_feat = torch.cat([self.target_mlp(x[dst]), self.source_mlp(x[src]), edge_attr.unsqueeze(-1)], dim=-1)  # (E, c_in + c_in + 1)
        logits = self.pi_mlp(edge_feat).flatten() # (E,)
        #bulk to x feat
        # bulk_dst = torch.arange(x.size(0), device=device)
        # dst_all = torch.cat([dst, bulk_dst])

        # logits_bulk_all = pyro.param(
        #         "all_clu_weight",
        #             torch.zeros(self.n, dtype=torch.float),
        #             ).to(device)
        # logits_bulk = logits_bulk_all[idx]
        # print(logits_bulk.mean(), logits_bulk.var(), idx.max())

        # assert torch.all(torch.isfinite(logits_bulk)), \
        #     f"NaN in logits_bulk: {logits_bulk}"
        
        assert torch.all(torch.isfinite(logits)), \
            f"NaN in logits_ext: {logits}"

        #confine bulk and edge feats to one simplex
        # logits_ext = torch.cat([logits, logits_bulk], dim=0).flatten() # (E,)
        # assert torch.all(torch.isfinite(logits_ext)), \
        #     f"NaN in logits_ext: {logits_ext}"

        # probs = torch_scatter.scatter_softmax(logits, dst)

        # q_omega = probs
        # q_clu_weight = probs[edge_index.size(1):]


        # ent = -(probs * (probs.clamp(min=1e-12).log()))   # (E,)
        # ent_per_dst = torch_scatter.scatter(ent, dst, dim=0, reduce="sum")  # (N_nodes,)
        # entropy = ent_per_dst.mean()  # scalar
        # per-edge contribution and per-dst (node) entropy
        # entropy = self.entropy_over_src_clusters_per_dst(probs, src, dst, clusters)
        # print("Entropy:", entropy.item())

        return logits
    

    def entropy_over_src_clusters_per_dst(
        self,
        probs: torch.Tensor,        # (E,) per-edge probs (sum=1 per dst)
        src: torch.Tensor,          # (E,) long
        dst: torch.Tensor,          # (E,) long
        node_clusters: torch.Tensor,# (N,) long, not guaranteed dense
        exclude_isolated: bool = True,
        eps: float = 1e-12,
    ):
        device = probs.device
        N = int(node_clusters.shape[0])
        C = int(self.num_clusters)   # force fixed C

        # ---- sanity checks ----
        assert probs.shape[0] == src.shape[0] == dst.shape[0], "E mismatch"
        assert src.max() < N and dst.max() < N, "src/dst out of bounds"
        assert node_clusters.max() < C, \
            f"Found cluster id {node_clusters.max().item()} >= num_clusters={C}"

        # cluster for each src node
        src_clu = node_clusters[src]    # (E,)

        # linear index into contrib_flat
        lin_idx = dst * C + src_clu     # (E,)

        # scatter probs into contrib matrix
        contrib_flat = torch_scatter.scatter_add(
            probs, lin_idx,
            dim=0, dim_size=N * C
        )                               # (N*C,)
        contrib = contrib_flat.view(N, C)

        # normalize row-wise
        mass = contrib.sum(dim=1, keepdim=True)     # (N,1)
        contrib_norm = contrib / mass.clamp_min(eps)

        if exclude_isolated:
            contrib_norm[mass.squeeze(1) <= 0] = 0.0

        # entropy per dst
        p = contrib_norm.clamp_min(eps)
        ent_per_dst = -(p * p.log()).sum(dim=1)

        # average
        if exclude_isolated:
            mask = mass.squeeze(1) > 0
            mean_entropy = ent_per_dst[mask].mean() if mask.any() \
                else torch.tensor(0.0, device=device, dtype=probs.dtype)
        else:
            mean_entropy = ent_per_dst.mean()

        return mean_entropy





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
    r"""Decode ref-level (x) attention weights (\omega) by attending 
    query-level (u) to ref-level cell types (c): \omega ~ p(\omega | z, c)
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
            nn.Linear(configs.c_latent, 2),
        )

        self.celltype_aware = configs.celltype_aware

    def forward(self, z, c, edge_index_dict, edge_attr_dict, only_omega=False):
        # Ablation: unpool z conditional on c vs. avg. unpool
        if self.celltype_aware:
            s = self.z_to_s((z, c), edge_index_dict[self.q2r])  # unpooled `z` from query-level -> ref-level
        else:
            q2r_src, q2r_dst = edge_index_dict[self.q2r]  # source & target edge indices (query-target graph)
            s = torch_scatter.scatter_mean(z[q2r_src], q2r_dst, dim=0, dim_size=c.size(0))

        if only_omega:
            return s
        
        # Concat embeddings from src & dst nodes -> edge embedding
        r2r_src, r2r_dst = edge_index_dict[self.r2r]  
        edge_feats = torch.cat([s[r2r_src], s[r2r_dst]], dim=-1)

        # Gamma shape (alpha) & rate (beta)
        omegas = self.edge_to_omega(edge_feats)
        loc = omegas[:, 0]
        scale = F.softplus(omegas[:, 1]) + EPS

        return s, loc, scale

    

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
