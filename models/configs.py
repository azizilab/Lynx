import torch
from ml_collections import ConfigDict
import logging

LOGGER = logging.getLogger()

# ----------------
# Model configs
# ----------------

def set_model_configs(verbose=False, **kwargs):
    model_configs = ConfigDict()

    model_configs.c_in = 247
    model_configs.c_hidden = 8
    model_configs.c_latent = 1 
    model_configs.dropout = 0.1
    model_configs.c0 = 5.  # Beta prior parameter (`c0`)

    model_configs.device = torch.device('cpu')
    model_configs.batch_size = 1
    model_configs.alpha = 0.5  # weight: GPCA Laplacian regularization 
    model_configs.beta = 0.5  # weight: KL div. (beta-VAE)
    
    # model priors
    model_configs.px_scale = 1.

    for k, v in kwargs.items():
        model_configs[k] = v
        if k in model_configs.keys():
            LOGGER.info('Updating model config\t{0}: {1}'.format(k, v))

    if verbose:
        for k, v in model_configs.items():
            LOGGER.info('Model config\t{0}: {1}'.format(k, v))

    return model_configs


def set_train_configs(verbose=False, **kwargs):
    train_configs = ConfigDict()
    train_configs.lr = 0.01
    train_configs.n_epochs = 200
    train_configs.gamma = 0.95   # LR decay rate

    for k, v in kwargs.items():
        train_configs[k] = v
        if k in train_configs.keys():
            LOGGER.info('Updating training config\t{0}: {1}'.format(k, v))

    if verbose:
        for k, v in train_configs.items():
            LOGGER.info('Model config\t{0}: {1}'.format(k, v))

    return train_configs
