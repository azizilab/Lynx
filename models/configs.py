import torch
from ml_collections import ConfigDict
import logging

LOGGER = logging.getLogger()

# ----------------
# Model configs
# ----------------

def set_model_configs(c_in, c_aux=-1, verbose=False, **kwargs):
    model_configs = ConfigDict()

    model_configs.c_in = c_in
    model_configs.c_aux = c_in if c_aux == -1 else c_aux    # Reduced auxiliary dim.
    model_configs.c_hidden = 8
    model_configs.c_latent = 1 
    model_configs.dropout = 0.1
    model_configs.k_hop = 3

    model_configs.device = torch.device('cpu')
    model_configs.batch_size = 1
    model_configs.beta = 0.5  # weight: KL div. (beta-VAE)
    model_configs.prior = 'normal'
    model_configs.enc_option = 'cat'

    for k, v in kwargs.items():
        model_configs[k] = v
        if k in model_configs.keys():
            LOGGER.info('Updating model config\t{0}: {1}'.format(k, v))
    print('\n')

    if verbose:
        for k, v in model_configs.items():
            LOGGER.info('Model config\t{0}: {1}'.format(k, v))

    return model_configs


def set_train_configs(verbose=False, **kwargs):
    train_configs = ConfigDict()
    train_configs.lr = 0.01
    train_configs.n_epochs = 200
    train_configs.gamma = 0.1   # LR decay rate
    train_configs.annealing = False

    for k, v in kwargs.items():
        train_configs[k] = v
        if k in train_configs.keys():
            LOGGER.info('Updating training config\t{0}: {1}'.format(k, v))
    print('\n')

    if verbose:
        for k, v in train_configs.items():
            LOGGER.info('Model config\t{0}: {1}'.format(k, v))

    return train_configs
