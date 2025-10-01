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
    configs.c_aux = c_in if c_aux == -1 else c_aux
    configs.c_hidden = 64
    configs.c_latent = 6

    configs.batch_size = 1
    configs.dropout = 0.
    configs.beta = 1.0  # KL div. weight (beta-VAE)
    configs.k_hop = 1
    configs.num_heads = 1
    configs.celltype_aware = False  # whether to use cell-type-aware projection (z -> s)
    configs.seed = 42  # manual seed

    # Hyperparameter for cluster & edge priors
    configs.base_sparsity = 1.
    configs.abundance_penalization = 5.
    configs.clu_weight = 0.1   # cluster weight initialization

    for k, v in kwargs.items():
        configs[k] = v

    if verbose:
        for k, v in configs.items():
            LOGGER.info('Model config\t{0}: {1}'.format(k, v))
        print('\n')
    print('\n\n')

    return configs


def set_train_configs(verbose=False, **kwargs):
    configs = ConfigDict()
    configs.lr = 1e-2
    configs.n_epochs = 500
    configs.weight_decay = 1e-3
    configs.betas = (.95, .999) 
    configs.anneal = False
    configs.warmup_epochs = 100
    configs.gamma = 0.999
    configs.patience = 20  # early-stopping counter

    for k, v in kwargs.items():
        configs[k] = v

    if verbose:
        for k, v in configs.items():
            LOGGER.info('Training config\t{0}: {1}'.format(k, v))
        print('\n')
    print('\n\n')

    return configs
