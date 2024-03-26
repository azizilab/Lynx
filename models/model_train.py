import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from tqdm import trange
from utils import nx_to_edge_attrs
from torch_geometric.nn import VGAE


def run_one_epoch(model, optimizer, x, edge_index, edge_weight):
    model.train()
    optimizer.zero_grad()

    latent = model.encoder(x, edge_index, edge_weight)
    loss, recon_loss, l1_loss, kl_loss = model.loss(latent, edge_index, edge_weight)
    loss.backward()
    optimizer.step()

    return float(loss), float(recon_loss), float(l1_loss), float(kl_loss)


def train(
    model,
    dataloader,
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
    optimizer = optim.Adam(model.parameters(), lr=train_configs.lr)
    pbar = trange(train_configs.n_epochs, desc='Training', leave=True)
    
    # TODO: add batched subgraph training
    for _ in enumerate(pbar):
        for graph_data in dataloader:
            x = graph_data.x.float().to(device)
            edge_index = graph_data.edge_index.to(device)
            edge_weight = graph_data.edge_weight.to(device) if 'edge_weight' in graph_data.keys() else None

            loss, nll, sl, kl = run_one_epoch(model, optimizer, x, edge_index, edge_weight)
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

