import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from ml_collections import ConfigDict

from .. import CyIFDataset


# --------------------------
# Training & Model configs
# --------------------------

def set_model_configs(verbose=False, **kwargs):
    model_configs = ConfigDict()

    model_configs.c_in = 3
    model_configs.c_out = 3
    model_configs.c_base = 4
    model_configs.layer_mults = [1, 2]
    model_configs.drop_rate = 0.1

    model_configs.device = torch.device('cpu')
    model_configs.batch_size = 1
    model_configs.beta = 0.1

    model_configs.ydim = 128
    model_configs.xdim = 128
    model_configs.latent_dim = 4096

    model_configs.pz_std = 0.01
    model_configs.device = torch.device('cpu')

    if verbose:
        for k, v in model_configs.items():
            print('Model config {0} = {1}'.format(k, v))

    for k, v in kwargs.items():
        model_configs[k] = v
        if k in model_configs.keys() and verbose:
            print('Updating model config {0} as {1}'.format(k, v))

    return model_configs


def set_train_configs(data_path, prior_path, verbose=False, **kwargs):
    train_configs = ConfigDict()

    train_configs.data_path = data_path
    train_configs.prior_path = prior_path
    train_configs.lr = 1e-5
    train_configs.n_epochs = 200

    if verbose:
        for k, v in train_configs.items():
            print('Model config {0} = {1}'.format(k, v))

    for k, v in kwargs.items():
        train_configs[k] = v
        if k in train_configs.keys() and verbose:
            print('Updating model config {0} as {1}'.format(k, v))


# ---------------
# Util functions
# ---------------

def run_one_epoch(model, dataloader, optimizer, device):
    model = model.to(device)
    model.train()

    loss_sum = 0.0
    nll_sum = 0.0
    kl_sum = 0.0    

    cnt = 0
    for x, pz_mu in dataloader:
        
        cnt += 1
        x = x.float().to(device)
        pz_mu = pz_mu.float().to(device)
        
        inference_terms = model.inference(x)
        x_pred = model.generative(inference_terms.x_encs, 
                                  inference_terms.qz)
                                  
        if any(torch.isnan(p).any() for p in model.parameters()):
            print('NaNs detected in model parameters, Skipping current epoch...')
            continue    

        loss_configs = model.get_loss(x, x_pred, pz_mu, inference_terms)
        loss, loss_nll, loss_kl = loss_configs.tot, loss_configs.nll, loss_configs.kl

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        loss_sum += loss
        nll_sum += loss_nll
        kl_sum = loss_kl

    avg_loss = loss_sum / cnt
    avg_nll = nll_sum / cnt
    avg_kl = kl_sum / cnt

    return avg_loss, avg_nll, avg_kl


def train(
    vae_model,
    train_configs, 
    model_configs
):
    torch.manual_seed(0)
    np.random.seed(0)

    losses = []
    losses_nll = []
    losses_kl = []

    dataset = CyIFDataset(data_path=train_configs.data_path, prior_path=train_configs.prior_path)
    dataloader = DataLoader(dataset, batch_size=model_configs.batch_size, shuffle=True, drop_last=True)

    device = model_configs.device
    model = vae_model(model_configs, prior_std=model_configs.pz_std)
    optimizer = optim.Adam(model.parameters(), lr=train_configs.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    n_epochs = train_configs.n_epochs
    for epoch in range(n_epochs):
        avg_loss, avg_nll, avg_kl = run_one_epoch(model, dataloader, optimizer, device=device)
        losses.append(avg_loss)
        losses_nll.append(avg_nll)
        losses_kl.append(avg_kl)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print("Epoch[{}/{}], total_loss: {:.4f}, reconst: {:.4f}, kl: {:.4f}".format(
                epoch + 1, n_epochs, avg_loss, avg_nll, avg_kl)
            )

        torch.cuda.empty_cache()

    losses_dict = {
        'total': losses,
        'NLL': losses_nll,
        'KL': losses_kl
    }

    return model, losses_dict
