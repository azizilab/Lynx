import os
import sys
import numpy as np
import torch
import torch.nn as nn

from ml_collections import ConfigDict
from torch_geometric.loader import DataLoader
from tqdm import trange, tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def sigmoid_annealing(epoch, start=0.01, end=1, midpoint=100, slope=0.1):
    return start + (end - start) / (1 + np.exp(-slope * (epoch - midpoint)))


def train_vgae(
    model: nn.Module,
    train_configs: ConfigDict,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    DEBUG: bool = False
):
    import gc
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

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
    patience = train_configs.patience
    min_val_loss = np.inf

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

            # DEBUG: plotting disentanglement
            if DEBUG and epoch % 10 == 0:
                data = next(iter(val_dataloader))
                res = model.predict(data, device=device)
                pz = res.pz.detach().cpu().numpy()
                qz = res.qz_params[0].detach().cpu().numpy()

                # axes[0].clear()
                # axes[1].clear()
                axes[0].set_title('p(z)')
                axes[1].set_title('q(z|x, u)')

                sns.heatmap(np.corrcoef(pz.T), cmap='RdBu_r', square=True, ax=axes[0])
                sns.heatmap(np.corrcoef(qz.T), cmap='RdBu_r', square=True, ax=axes[1])

                fig.canvas.draw()
                plt.show()

                del data, res, pz, qz
                gc.collect()

        else:
            pbar.set_description(
                "Epoch {0} train ELBO: {1}".format(
                    epoch, np.round(epoch_loss/n_obs, 3)
                )
            )        
        
        plt.ioff()
                
    return (model, losses) if val_dataloader is None else (model, losses, val_losses)
