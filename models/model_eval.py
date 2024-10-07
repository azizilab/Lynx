import os
import sys
import numpy as np
import torch

from scipy import sparse
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from util.utils import nx_to_edge_attrs

def eval_sbvae(model, graph, feature_mat,
         device = torch.device('cpu')):
    
    model = model.to(device)
    x = torch.tensor(feature_mat)
    x = x.float().to(device)
    edge_index, edge_weight = nx_to_edge_attrs(graph)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    model.eval()
    with torch.no_grad():
        latent = model.encoder(x, edge_index, edge_weight)
        recon = model.decoder(latent, edge_index)
    return latent, recon


def evaluate_elbo(model, dataloader, device=torch.device('cpu')):
    optimizer = Adam({"lr": 1.0e-3})  # dummy optimizer
    
    model = model.to(device)
    elbo = Trace_ELBO()
    svi = SVI(model.model, model.guide, optimizer, elbo)
    elbos = []
    
    for data in dataloader:
        x = data.x.to(device).float()
        u = data.u.to(device).float()
        edge_index = data.edge_index.to(device)
        elbo = svi.evaluate_loss(x, u, edge_index)

        elbos.append(elbo / x.shape[0])

    return elbos


def evaluate_kl(model, dataloader, device=torch.device('cpu')):
    from pyro.optim import Adam
    from pyro.infer import TraceMeanField_ELBO
    from pyro import poutine
    
    optimizer = Adam({"lr": 1.0e-3})  # dummy optimizer
    
    model = model.to(device)
    elbo = TraceMeanField_ELBO()
    svi = SVI(model.model, model.guide, optimizer, elbo)
    
    kl_divs = []
    for data in dataloader:
        x = data.x.to(device).float()
        u = data.u.to(device).float()
        edge_index = data.edge_index.to(device)

        with poutine.scale(scale=1e-7):
            kl_div = svi.evaluate_loss(x, u, edge_index)
            kl_divs.append(kl_div)

    return kl_divs
