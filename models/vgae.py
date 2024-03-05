import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from torch.distributions import Normal, kl_divergence
from torch_geometric.nn import VGAE, GCNConv
from ml_collections import ConfigDict


class GCNEncoder(nn.Module):
    def __init__(self, configs):
        super(GCNEncoder, self).__init__()
        self.layer1 = GCNConv(configs.c_in, configs.c_hidden)
        self.qz_mu = GCNConv(configs.c_hidden, configs.c_latent)
        self.qz_logstd = GCNConv(configs.c_hidden, configs.c_latent)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index).relu()
        z_mu = self.qz_mu(x, edge_index)
        z_logstd = self.qz_logstd(x, edge_index)
        return z_mu, z_logstd
    