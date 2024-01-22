import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import norm_by_channel
from utils import norm_transform, inv_norm_transform
from constants import CYIF_MEAN, CYIF_STD


def get_latent(model, img, device=torch.device('cpu')):    
    """
    Get latent dim (`qz`)
    """
    # Normalize image
    normalize = norm_transform(CYIF_MEAN, CYIF_STD)    
    x = torch.unsqueeze(normalize(img.transpose(1,2,0)), 0)
    x = x.to(device).float()

    model = model.to(device)
    model.eval()

    # Linear interpolation in low-dim
    with torch.no_grad():
        z = model.inference(x).qz

    return z.detach().cpu().squeeze().numpy()


def predict(model, dataloader, device, batch_size=1):
    """
    Get reconstructed image (X')
    """
    model.eval()
    model.batch_size = batch_size
    
    x_list = []
    x_pred_list = []
    qz_list = []

    with torch.no_grad():
        for x in dataloader:
            x = x.float().to(device)
            
            # Inverse normalization setup
            inv_normalize = inv_norm_transform(CYIF_MEAN, CYIF_STD)

            # X & X_pred
            inference_terms = model.inference(x)
            x_pred = model.generative(inference_terms.qz)

            x = inv_normalize(x.detach().squeeze()).cpu().numpy()
            x_list.append(x)
            
            x_pred = norm_by_channel(
                inv_normalize(x_pred.detach().squeeze()).cpu().numpy()
            )
            x_pred_list.append(x_pred)
            
            qz = inference_terms.qz.detach().cpu().numpy().squeeze()
            out_dim = np.int8(np.sqrt(len(qz)))
            temps_pred = qz.reshape(out_dim, -1)
            qz_list.append(temps_pred)

    return x_list, x_pred_list, qz_list


def interp(model, img0, img1, alpha=0.5, device=torch.device('cpu')):
    """
    Get interpolated image (X_{interp}) from `q_{z1}` & `q_{z2}`
    """
    assert img0.ndim == img1.ndim == 3 and img0.shape == img1.shape, \
        "Images 1 & 2 should have the same dim (C, Y, X)"
    
    # Normalization setup
    normalize = norm_transform(CYIF_MEAN, CYIF_STD)
    inv_normalize = inv_norm_transform(CYIF_MEAN, CYIF_STD)

    x0 = torch.unsqueeze(normalize(img0.transpose(1,2,0)), 0)
    x1 = torch.unsqueeze(normalize(img1.transpose(1,2,0)), 0)
    x0 = x0.to(device).float()
    x1 = x1.to(device).float()

    model = model.to(device)
    model.eval()

    # Linear interpolation in low-dim
    with torch.no_grad():
        z0 = model.inference(x0).qz
        z1 = model.inference(x1).qz
        z_interp = (1-alpha)*z0 + alpha*z1
        x_interp = model.generative(z_interp)

    x_interp = norm_by_channel(
        inv_normalize(x_interp.detach().squeeze()).cpu().numpy()
    )

    return x_interp
