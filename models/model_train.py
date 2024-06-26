import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from torch_geometric.nn import VGAE

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def run_one_epoch(model, optimizer, x, 
                  edge_index, edge_weight, 
                  u_prior):
    model.train()
    optimizer.zero_grad()
    
    latent = model.encoder(x, edge_index, edge_weight)
    recon = model.decoder(latent, edge_index)
    loss, recon_loss, l1_loss, ortho_loss, kl_loss, orient_loss = model.loss(latent, 
                                                                             recon, 
                                                                             u_prior,
                                                                             x, 
                                                                             edge_index)
    loss.backward()
    optimizer.step()

    return (float(loss), float(recon_loss), float(l1_loss), 
            float(ortho_loss), float(kl_loss), float(orient_loss))


def train(
    model,
    dataloader,
    train_configs,
    device = torch.device('cpu')
):
    torch.manual_seed(42)
    np.random.seed(42)
    
    assert isinstance(model, VGAE), "Requires model as a VGAE object"

    losses = []
    nlls = []
    l1s = []
    sls = []
    kls = []
    orients = []

    model = model.to(device)
    model.encoder.training = True
    optimizer = optim.Adam(model.parameters(), lr=train_configs.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_configs.gamma)
    pbar = trange(train_configs.n_epochs, desc='Training', leave=True)
    
    for _ in enumerate(pbar):
        batch_losses = []
        batch_nlls = []
        batch_l1s = []
        batch_sls = []
        batch_kls = []
        batch_orients = []

        # graph_data = next(iter(dataloader))
        for graph_data in dataloader:
            x = graph_data.x.float().to(device)
            edge_index = graph_data.edge_index.to(device)
            edge_weight = graph_data.edge_weight.to(device) if 'edge_weight' in graph_data.keys() else None
            u_prior = graph_data.u_prior.float().to(device)
            u_prior = torch.unsqueeze(u_prior, dim=-1)

            loss, nll, l1, sl, kl, orient = run_one_epoch(model, optimizer, x, 
                                                          edge_index, edge_weight, u_prior)
            batch_losses.append(loss)
            batch_nlls.append(nll)
            batch_l1s.append(l1)
            batch_sls.append(sl)
            batch_kls.append(kl)
            batch_orients.append(orient)

        losses.append(np.mean(batch_losses))
        nlls.append(np.mean(batch_nlls))
        l1s.append(np.mean(batch_l1s))
        sls.append(np.mean(batch_sls))
        kls.append(np.mean(batch_kls))
        orients.append(np.mean(batch_orients))

        scheduler.step()

        pbar.set_postfix({'Total': '{:.3f}\n'.format(losses[-1]),
                          'Recon': '{:.3f}'.format(nlls[-1]),
                          'L1': '{:.3f}'.format(l1s[-1]), 
                          'Ortho loss': '{:.3f}'.format(sls[-1]),
                          'KL': '{:.3f}'.format(kls[-1]),
                          'Orient': '{:.3f}'.format(orients[-1])})
            
    pbar.close()
    return losses, nlls, l1s, sls, kls, orients
