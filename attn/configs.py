from ml_collections import ConfigDict

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
    configs.dropout = 0.5
    configs.beta = 1.0  # KL div. weight (beta-VAE)
    configs.k_hop = 2
    configs.w_init = None   # Intialize weight parameters of conditional prior layer(s)

    for k, v in kwargs.items():
        configs[k] = v

    if verbose:
        for k, v in configs.items():
            print('Model config\t{0}: {1}'.format(k, v))
        print('\n')
    
    return configs


def set_train_configs(verbose=False, **kwargs):
    configs = ConfigDict()
    configs.lr = 1e-3
    configs.n_epochs = 500
    configs.weight_decay = 1e-3
    configs.betas = (.95, .999) 
    configs.anneal = False
    configs.warmup_epochs = 50
    configs.patience = 20  # early-stopping counter

    for k, v in kwargs.items():
        configs[k] = v

    if verbose:
        for k, v in configs.items():
            print('Training config\t{0}: {1}'.format(k, v))
        print('\n')

    return configs
