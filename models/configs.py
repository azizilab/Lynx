import torch
from ml_collections import ConfigDict


# ----------------
# Model configs
# ----------------

def set_model_configs(verbose=False, **kwargs):
    model_configs = ConfigDict()

    model_configs.c_in = 247
    model_configs.c_hidden = 8
    model_configs.c_latent = 1 
    model_configs.drop_rate = 0.1
    model_configs.alpha = 2.

    model_configs.device = torch.device('cpu')
    model_configs.batch_size = 1
    model_configs.beta = 0.01  # regularization weights
    
    # model priors
    model_configs.pz_scale = 1.
    model_configs.pu_scale = 1.
    model_configs.px_scale = 1.

    for k, v in kwargs.items():
        model_configs[k] = v
        if k in model_configs.keys():
            print('Updating model config {0} as {1}'.format(k, v))

    if verbose:
        for k, v in model_configs.items():
            print('Model config {0} = {1}'.format(k, v))

    return model_configs


def set_train_configs(verbose=False, **kwargs):
    train_configs = ConfigDict()

    train_configs.weight = 0.1
    train_configs.lr = 0.01
    train_configs.n_epochs = 200

    for k, v in kwargs.items():
        train_configs[k] = v
        if k in train_configs.keys():
            print('Updating training config {0} as {1}'.format(k, v))

    if verbose:
        for k, v in train_configs.items():
            print('Model config {0} = {1}'.format(k, v))

    return train_configs
