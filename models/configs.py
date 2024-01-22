import torch
from ml_collections import ConfigDict


# ----------------
# Model configs
# ----------------

def set_model_configs(verbose=False, **kwargs):
    model_configs = ConfigDict()

    model_configs.c_in = 3
    model_configs.c_out = 3
    model_configs.c_base = 4
    model_configs.layer_mults = [1, 2, 4]
    model_configs.drop_rate = 0.1

    model_configs.device = torch.device('cpu')
    model_configs.batch_size = 1
    model_configs.beta = 0.1

    model_configs.ydim = 128
    model_configs.xdim = 128
    model_configs.latent_dim = 256

    model_configs.pz_std = 0.01

    for k, v in kwargs.items():
        model_configs[k] = v
        if k in model_configs.keys():
            print('Updating model config {0} as {1}'.format(k, v))

    if verbose:
        for k, v in model_configs.items():
            print('Model config {0} = {1}'.format(k, v))

    return model_configs


def set_train_configs(data_path, verbose=False, **kwargs):
    train_configs = ConfigDict()

    train_configs.data_path = data_path
    train_configs.lr = 1e-5
    train_configs.n_epochs = 200

    for k, v in kwargs.items():
        train_configs[k] = v
        if k in train_configs.keys():
            print('Updating training config {0} as {1}'.format(k, v))

    if verbose:
        for k, v in train_configs.items():
            print('Model config {0} = {1}'.format(k, v))

    return train_configs
