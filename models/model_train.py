import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange


# ---------------
# Util functions
# ---------------

def run_one_epoch(model, dataloader, optimizer, device):
    model = model.to(device)
    model.train()

    losses = []
    nll_losses = []
    kl_losses = []

    cnt = 0
    for x in dataloader:
        
        cnt += 1
        x = x.float().to(device)
        
        inference_terms = model.inference(x)
        x_pred = model.generative(inference_terms.qz)
                                  
        if any(torch.isnan(p).any() for p in model.parameters()):
            print('NaNs detected in model parameters, Skipping current epoch...')
            continue    

        loss_configs = model.get_loss(x, x_pred, inference_terms)
        loss, loss_nll, loss_kl = loss_configs.tot, loss_configs.nll, loss_configs.kl

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        losses.append(loss.detach().item())
        nll_losses.append(loss_nll.detach().item())
        kl_losses.append(loss_kl.detach().item())

    return np.mean(losses), np.mean(nll_losses), np.mean(kl_losses)


def train(
    model,
    dataloader, 
    train_configs, 
    model_configs
):
    torch.manual_seed(0)
    np.random.seed(0)

    losses = []
    losses_nll = []
    losses_kl = []

    device = model_configs.device
    optimizer = optim.Adam(model.parameters(), lr=train_configs.lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    n_epochs = train_configs.n_epochs    
    pbar = trange(n_epochs, desc='Training', leave=True)
    for epoch in pbar:
        avg_loss, avg_nll, avg_kl = run_one_epoch(model, dataloader, optimizer, device=device)
        losses.append(avg_loss)
        losses_nll.append(avg_nll)
        losses_kl.append(avg_kl)

        # scheduler.step()
        pbar.set_postfix({'Training loss': '{:.3f}'.format(avg_loss),
                          'reconst': '{:.3f}'.format(avg_nll),
                          'kl': '{:.3f}'.format(avg_kl)},
                          refresh=True)

        torch.cuda.empty_cache()
    
    pbar.close()

    losses_dict = {
        'total': losses,
        'NLL': losses_nll,
        'KL': losses_kl
    }

    return model, losses_dict
