import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from tqdm import trange
from utils import nx_to_edge_index
from torch_geometric.nn import VGAE


def run_one_epoch(model, optimizer, x, edge_index):
    model.train()
    optimizer.zero_grad()

    z = model.encode(x, edge_index)
    loss, recon_loss, l1_loss, kl_loss = model.loss(z, edge_index)
    loss.backward()
    optimizer.step()

    return float(loss), float(recon_loss), float(l1_loss), float(kl_loss)


def train(
    model,
    graph,
    feature_mat,
    train_configs,
    device = torch.device('cpu')
):
    torch.manual_seed(0)
    np.random.seed(0)
    assert isinstance(model, VGAE), "Requires model as a VGAE object"

    losses = []
    nlls = []
    sls = []
    kls = []

    model = model.to(device)
    x = torch.tensor(feature_mat)
    x = x.float().to(device)
    edge_index = nx_to_edge_index(graph).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_configs.lr)
    pbar = trange(train_configs.n_epochs, desc='Training', leave=True)
    
    for _ in pbar:
        loss, nll, sl, kl = run_one_epoch(model, optimizer, x, edge_index)
        losses.append(loss)
        nlls.append(nll)
        sls.append(sl)
        kls.append(kl)

        pbar.set_postfix({'Training loss': '{:.3f}'.format(loss),
                          'NLL': '{:.3f}'.format(nll),
                          'Sparsity loss': '{:.3f}'.format(sl),
                          'KL': '{:.3f}'.format(kl)})
    pbar.close()
    return losses, nlls, sls, kls

