import torch
from ml_collections import ConfigDict
import logging

LOGGER = logging.getLogger()

# ----------------
# Model configs
# ----------------

def set_model_configs(c_in, c_aux=-1, verbose=False, **kwargs):
    configs = ConfigDict()

    configs.c_in = c_in
    configs.c_aux = c_in if c_aux == -1 else c_aux    # Reduced auxiliary dim.
    configs.c_hidden = 64
    configs.c_latent = 6

    configs.batch_size = 1
    configs.dropout = 0.1
    configs.k_hop = 3
    configs.beta = 1.0  # KL div. weight (beta-VAE)
    configs.w_init = None

    for k, v in kwargs.items():
        configs[k] = v
        if k in configs.keys():
            LOGGER.info('Updating model config\t{0}: {1}'.format(k, v))
    print('\n')

    if verbose:
        for k, v in configs.items():
            LOGGER.info('Model config\t{0}: {1}'.format(k, v))
    
    return configs


def set_train_configs(verbose=False, **kwargs):
    configs = ConfigDict()
    configs.lr = 1e-3
    configs.n_epochs = 500
    configs.weight_decay = 1e-3
    configs.betas = (.95, .999) 
    configs.patience = 20  # early-stopping counter

    for k, v in kwargs.items():
        configs[k] = v
        if k in configs.keys():
            LOGGER.info('Updating training config\t{0}: {1}'.format(k, v))
    print('\n')

    if verbose:
        for k, v in configs.items():
            LOGGER.info('Model config\t{0}: {1}'.format(k, v))

    return configs
