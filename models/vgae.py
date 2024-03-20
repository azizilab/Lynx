import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder, MLP
from torch_geometric.utils import to_dense_adj

class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        self.layer1 = GCNConv(configs.c_in, configs.c_hidden)
        self.qz_mu = GCNConv(configs.c_hidden, configs.c_latent)
        self.qz_logstd = GCNConv(configs.c_hidden, configs.c_latent)

    def forward(self, x, edge_index, edge_weight):
        x = self.layer1(
            x, 
            edge_index=edge_index,
            edge_weight=edge_weight
        ).relu()
        
        z_mu = self.qz_mu(
            x, 
            edge_index=edge_index,
            edge_weight=edge_weight
        )

        z_mu = torch.tanh(z_mu)
        z_logstd = self.qz_logstd(
            x,
            edge_index=edge_index,
            edge_weight=edge_weight
        )

        return z_mu, z_logstd
    

class SparseVGAE(VGAE):
    def __init__(self, configs):
        super(SparseVGAE, self).__init__(
            encoder=GCNEncoder(configs),
            decoder=InnerProductDecoder()
        )
        self.beta = configs.beta
        self.bs = configs.batch_size_l1
        
    def loss(self, z, edge_index, edge_weight):
        n_nodes = edge_index.shape[1]
        A_hat = self.decoder.forward_all(z).float()

        recon_loss = self.get_recon_loss(z, edge_index, edge_weight)

        l1_loss = self.get_smoothness_loss(z, edge_index, edge_weight)
        kl_loss = self.kl_loss()
        loss = recon_loss + self.beta*l1_loss + self.beta*kl_loss
        
        return loss, recon_loss, l1_loss, kl_loss

    def get_recon_loss(self, z, edge_index, edge_weight):
        # Compute BCE as the surrogate loss function for NLL
        if edge_weight is None:
            # loss = self.recon_loss(z, edge_index)
            A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze(0)
            A_hat = self.decoder.forward_all(z, sigmoid=True)
            loss = torch.norm(A-A_hat, p=2)
        else:
            src_nodes, dst_nodes = edge_index
            recon_probs = torch.sigmoid((z[src_nodes] * z[dst_nodes]).sum(1))
            loss = F.binary_cross_entropy(recon_probs, edge_weight, reduction='mean')
        return loss
    
    def get_smoothness_loss(self, z, edge_index, edge_weight):
        A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze(0)
        A_prime = A + torch.diag(torch.ones(A.shape[0]))
        D = torch.diag(torch.sum(A_prime, dim=-1))
        D_prime = torch.sqrt(torch.inverse(D))

        L = D - A
        L_prime = D_prime.t() @ L @ D_prime
        lap_loss = torch.trace(z.t() @ L_prime @ z)

        ones = torch.ones(A.shape[0])
        sparse_loss = ones.t() @ torch.log(A_prime @ ones)

        return lap_loss + sparse_loss