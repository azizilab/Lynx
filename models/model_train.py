import os
import gc
import sys
import numpy as np
import torch
import torch.nn as nn
import pyro

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import comb
from sklearn.metrics import r2_score
from ml_collections import ConfigDict
from torch_geometric.loader import DataLoader
from tqdm import trange, tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.optim import PyroOptim
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import vgae

def train_vgae(
    model: nn.Module,
    train_configs: ConfigDict,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    DEBUG: bool = False
):  
    device = train_configs.device
    # optimizer = ClippedAdam({
    #     'lr': train_configs.lr,
    #     'lrd': train_configs.gamma,
    #     'weight_decay': 1e-3,
    #     'betas': (0.95, 0.999)
    # })

    
    elbo = Trace_ELBO()
    model = model.to(device)
    # pyro.module("model", model)

    optimizer = PyroOptim(optim.AdamW, {
    'lr': train_configs.lr,
    'weight_decay': 1e-3,
    'betas': (0.95, 0.999)
    })

    svi = SVI(model.model, model.guide, optimizer, elbo)

    # print("Registered Pyro Parameters:")
    # for name, param in pyro.get_param_store().items():
    #     print(name, param)

    # Training loop
    pbar = tqdm(range(train_configs.n_epochs))
    losses = []
    val_losses = []
    patience = train_configs.patience
    min_val_loss = np.inf

    # Monitor disentanglement 
    pz_corr_scores = []
    qz_corr_scores = []

    corr = 0
    r2 = 0

    for epoch in pbar:
        epoch_loss = 0.
        n_obs = 0.

        model.train()
        model = model.to(device)
        model.device = device

        for data in dataloader:
            edge_index = data.edge_index.contiguous().to(device)

            if True:
                # VGAE with multi-scale graphs (X -> Z -> Y)
                x = data.x.to(device).float()
                y = data.y.to(device).float()
                neighbors = data.neighbors.to(device)

                # DEBUG NaNs:
                with torch.autograd.detect_anomaly():
                    loss = svi.step(x, y, edge_index, neighbors)
                n_obs += y.size(0)

                


                # for name, param in pyro.get_param_store().items():
                #     if param.grad is not None:
                #         print(f"{name}: grad norm = {param.grad.norm()}")
                #     else:
                #         print(f"{name}: no gradient")

            else:
                # VGAE with interpolated (same-dim) graphs (U -> Z -> X)
                u = data.u.to(device).float()  
                x = data.x.to(device).float()
                loss = svi.step(x, u, s, edge_index)
                n_obs += x.size(0)

            epoch_loss += loss

        # if epoch % 10 == 0:
        #     for name, param in pyro.get_param_store().items():
        #             param.register_hook(lambda grad: print(f"Gradient for {name}: {grad}"))
        #     for name, param in pyro.get_param_store().items():
        #                 if param.grad is not None:
        #                     print(f"{name}: grad norm = {param.grad.norm()}")
        #                 else:
        #                     print(f"{name}: no gradient")

        losses.append(epoch_loss/n_obs)

        if val_dataloader is not None:
            model.eval()
            epoch_val_loss = 0.
            n_val_obs = 0.

            with torch.no_grad():
                for data in val_dataloader:
                    edge_index = data.edge_index.contiguous().to(device)
                    if True:
                        x = data.x.to(device).float()
                        y = data.y.to(device).float()
                        neighbors = data.neighbors.to(device)
                        val_loss = svi.evaluate_loss(x, y, edge_index, neighbors)
                        n_val_obs += y.size(0)

                    else:
                        u = data.u.to(device).float()  
                        x = data.x.to(device).float()
                        val_loss = svi.evaluate_loss(x, u, s, edge_index)
                        n_val_obs += x.size(0)

                    epoch_val_loss += val_loss

                val_losses.append(epoch_val_loss/n_val_obs)

            pbar.set_description(
                "Epoch {0} train ELBO: {1}; val ELBO: {2}; val R2: {3}; val corr: {4}".format(
                    epoch, np.round(epoch_loss/n_obs, 3), np.round(epoch_val_loss/n_val_obs, 3), np.round(r2, 3), np.round(corr, 3)
                )
            )  

            if val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                patience = train_configs.patience
            else:
                patience -= 1

            if patience == 0:
                break

            if DEBUG and epoch % 10 == 0:
                # Monitor factor disentanglement
                # model = model.to('cpu')
                # model.device = torch.device('cpu')

                data = next(iter(val_dataloader))
                res = model.predict(data, device=device)
                pz = res.pz.detach().cpu().numpy()
                qz = res.qz_params[0].detach().cpu().numpy()
                py = res.py.detach().cpu().numpy()

                pz_corr = np.corrcoef(pz.T)
                qz_corr = np.corrcoef(qz.T)

                corr = np.abs(np.tril(qz_corr, k=-1)).sum() / comb(qz_corr.shape[0], 2)
                r2 = r2_score(data.y.flatten(), py.flatten())

                # Compute avg. pariwise factor correlations
                pz_corr_scores.append(
                    np.abs(np.tril(pz_corr, k=-1)).sum() / comb(pz_corr.shape[0], 2)
                )

                qz_corr_scores.append(
                    np.abs(np.tril(qz_corr, k=-1)).sum() / comb(qz_corr.shape[0], 2)
                )
                gc.collect()

        else:
            pbar.set_description(
                "Epoch {0} train ELBO: {1}".format(
                    epoch, np.round(epoch_loss/n_obs, 3)
                )
            )        

    if DEBUG:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(
            np.arange(len(pz_corr_scores)), pz_corr_scores, '.--', label='Prior'
        )
        ax.plot(
            np.arange(len(qz_corr_scores)), qz_corr_scores, '.--', label='Posterior'
        )
        ax.set_xlabel('Epoch checkpoint')
        ax.set_ylabel('Avg. factor correlations')
        ax.legend()

        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.show()
                
    return (model, losses) if val_dataloader is None else (model, losses, val_losses)
