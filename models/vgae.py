import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import VGAE, GCNConv, InnerProductDecoder


class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        self.layer1 = GCNConv(configs.c_in, configs.c_hidden)
        self.qz_mu = GCNConv(configs.c_hidden, configs.c_latent)
        self.qz_logstd = GCNConv(configs.c_hidden, configs.c_latent)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index).relu()
        z_mu = self.qz_mu(x, edge_index)
        z_mu = torch.tanh(z_mu)
        z_logstd = self.qz_logstd(x, edge_index)
        return z_mu, z_logstd
    

class SparseVGAE(VGAE):
    def __init__(self, configs):
        super(SparseVGAE, self).__init__(
            encoder=GCNEncoder(configs),
            decoder=InnerProductDecoder()
        )
        self.beta = configs.beta
        self.bs = configs.batch_size_l1
        
    def loss(self, z, edge_index):
        n_nodes = edge_index.shape[1]
        A_hat = self.recon_adj_matrix(z)

        recon_loss = self.recon_loss(z, edge_index)
        l1_loss = torch.norm(A_hat, p=1)
        kl_loss = self.kl_loss()

        loss = recon_loss + self.beta*l1_loss + (1/n_nodes)*kl_loss
        return loss, recon_loss, l1_loss, kl_loss
            
    def recon_adj_matrix(self, z):
        A_hat = self.decoder.forward_all(z, sigmoid=True) > 0.5
        return A_hat.float()