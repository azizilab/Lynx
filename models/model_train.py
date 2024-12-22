import os
import gc
import sys
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import comb
from ml_collections import ConfigDict
from torch_geometric.loader import DataLoader
from tqdm import trange, tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import vgae

def train_vgae(
    model: vgae.VGAE,
    train_configs: ConfigDict,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    DEBUG: bool = False
):  
    device = train_configs.device
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
    patience = train_configs.patience
    min_val_loss = np.inf

    # Monitor disentanglement 
    pz_corr_scores = []
    qz_corr_scores = []

    for epoch in pbar:
        epoch_loss = 0.
        n_obs = 0.

        model.train()
        model = model.to(device)
        model.device = device

        for data in dataloader:
            s = data.s.to(device).float()
            edge_index = data.edge_index.to(device)

            if isinstance(model, vgae.MultiscaleVGAE):
                # VGAE with multi-scale graphs (X -> Z -> Y)
                x = data.x.to(device).float()
                y = data.y.to(device).float()
                pooling_cluster = data.cluster.to(device)
                loss = svi.step(x, y, s, edge_index, pooling_cluster)
                n_obs += y.size(0)
            else:
                # VGAE with interpolated (same-dim) graphs (U -> Z -> X)
                u = data.u.to(device).float()  
                x = data.x.to(device).float()
                loss = svi.step(x, u, s, edge_index)
                n_obs += x.size(0)

            epoch_loss += loss

        losses.append(epoch_loss/n_obs)

        if val_dataloader is not None:
            model.eval()
            epoch_val_loss = 0.
            n_val_obs = 0.

            with torch.no_grad():
                for data in val_dataloader:
                    s = data.s.to(device).float()
                    edge_index = data.edge_index.to(device)
                    if isinstance(model, vgae.MultiscaleVGAE):
                        x = data.x.to(device).float()
                        y = data.y.to(device).float()
                        pooling_cluster = data.cluster.to(device)
                        val_loss = svi.evaluate_loss(x, y, s, edge_index, pooling_cluster)
                        n_val_obs += y.size(0)
                    else:
                        u = data.u.to(device).float()  
                        x = data.x.to(device).float()
                        val_loss = svi.evaluate_loss(x, u, s, edge_index)
                        n_val_obs += x.size(0)

                    epoch_val_loss += val_loss

                val_losses.append(epoch_val_loss/n_val_obs)

            pbar.set_description(
                "Epoch {0} train ELBO: {1}; val ELBO: {2}".format(
                    epoch, np.round(epoch_loss/n_obs, 3), np.round(epoch_val_loss/n_val_obs, 3)
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
                model = model.to('cpu')
                model.device = torch.device('cpu')

                data = next(iter(val_dataloader))
                res = model.predict(data, device=torch.device('cpu'))
                pz = res.pz.detach().cpu().numpy()
                qz = res.qz_params[0].detach().cpu().numpy()

                pz_corr = np.corrcoef(pz.T)
                qz_corr = np.corrcoef(qz.T)

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
