import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange, tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

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


def sigmoid_annealing(epoch, start=0.01, end=1, midpoint=100, slope=0.1):
    return start + (end - start) / (1 + np.exp(-slope * (epoch - midpoint)))


def train_vgae(
    model,
    dataloader,
    train_configs,
    val_dataloader=None,
):
    # TODO: add paired validation loss
    device = train_configs.device
    beta = model.configs.beta
    optimizer = ClippedAdam({
        'lr': train_configs.lr,
        'lrd': train_configs.gamma,
        'weight_decay': 1e-3,
        'betas': (0.95, 0.999)
    })
    elbo = Trace_ELBO()
    
    model = model.to(device)
    svi = SVI(model.model, model.guide, optimizer, elbo)

    # Training loop
    pbar = tqdm(range(train_configs.n_epochs))
    losses = []
    val_losses = []

    for epoch in pbar:
        epoch_loss = 0.
        n_obs = 0.
        model.train()

        if train_configs.annealing:
            model.configs.beta = sigmoid_annealing(epoch, end=beta, midpoint=50)

        for data in dataloader:
            x = data.x.to(device).float()
            u = data.u.to(device).float()
            s = data.s.to(device).float()
            edge_index = data.edge_index.to(device)

            loss = svi.step(x, u, s, edge_index)
            epoch_loss += loss
            n_obs += x.size(0)
        losses.append(epoch_loss/n_obs)

        if val_dataloader is not None:
            epoch_val_loss = 0.
            n_val_obs = 0.
            model.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    x = data.x.to(device).float()
                    u = data.u.to(device).float()
                    s = data.s.to(device).float()
                    edge_index = data.edge_index.to(device)

                    val_loss = svi.evaluate_loss(x, u, s, edge_index)
                    epoch_val_loss += val_loss
                    n_val_obs += x.size(0)
                val_losses.append(epoch_val_loss/n_val_obs)

        if val_dataloader is None:
            pbar.set_description(
                "Epoch {0} train ELBO: {1}".format(
                    epoch, np.round(epoch_loss/n_obs, 3)
                )
            )        
        else:
            pbar.set_description(
                "Epoch {0} train ELBO: {1}; val Elbo: {2}".format(
                    epoch, np.round(epoch_loss/n_obs, 3), np.round(epoch_val_loss/n_val_obs, 3)
                )
            )  
                
    return model, losses
