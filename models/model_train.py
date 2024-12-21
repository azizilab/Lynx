import os
import sys
import numpy as np
import torch
import torch.nn as nn

from scipy.special import comb
from ml_collections import ConfigDict
from torch_geometric.loader import DataLoader
from tqdm import trange, tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from module import GPCALayer


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
            x = data.x.to(device).float()
            u = data.u.to(device).float()
            s = data.s.to(device).float()
            edge_index = data.edge_index.to(device)

            loss = svi.step(x, u, s, edge_index)
            epoch_loss += loss
            n_obs += x.size(0)
            
        losses.append(epoch_loss/n_obs)

        if val_dataloader is not None:
            model.eval()
            epoch_val_loss = 0.
            n_val_obs = 0.

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

            if DEBUG and epoch % 10 == 0:
                # Monitor factor disentanglement
                # TODO: add total correlation computation (in actual model?)
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

                # if epoch % 100 == 0:
                #     fig, axes = plt.subplots(1, 2, figsize=(7, 3))
                #     axes[0].set_title('p(z)')
                #     axes[1].set_title('q(z|x, u)')
                #     sns.heatmap(np.corrcoef(pz.T), cmap='RdBu_r', square=True, ax=axes[0])
                #     sns.heatmap(np.corrcoef(qz.T), cmap='RdBu_r', square=True, ax=axes[1])
                #     plt.show()

                # del data, res, pz, qz
                gc.collect()

        else:
            pbar.set_description(
                "Epoch {0} train ELBO: {1}".format(
                    epoch, np.round(epoch_loss/n_obs, 3)
                )
            )        
        
        plt.ioff()

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
