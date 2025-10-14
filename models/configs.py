import torch
from ml_collections import ConfigDict
import logging

LOGGER = logging.getLogger()

# ----------------
# Model configs
# ----------------

def set_model_configs(graph_data, verbose=False, **kwargs):
    configs = ConfigDict()

    configs.ref = graph_data.ref
    configs.query = graph_data.query
    configs.c_in = graph_data[0][configs.ref].x.shape[1]  # ref-dim (observation)
    configs.c_aux = graph_data[0][configs.query].x.shape[1]  # query-dim (auxiliary)

    configs.c_hidden = 64
    configs.c_latent = 6

    configs.batch_size = 1
    configs.dropout = 0.
    configs.beta = 1.0  # KL div. weight (beta-VAE)
    configs.num_heads = 1
    configs.seed = 42  # manual seed

    # Hyperparameter for cluster & edge priors
    configs.alpha = 10.0   # Distance-spread dispersion
    configs.num_clusters = graph_data.num_clusters
    configs.gamma_shift = graph_data.gamma_shift  # Cluster-specific edge strength shift

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
