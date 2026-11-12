import torch
from ml_collections import ConfigDict
import logging

LOGGER = logging.getLogger()

# ----------------
# Model configs
# ----------------

ORIGINAL = dict(  # the published model (version='original')
    alpha=1.0, omega_prior='exponential', kappa_prior_scale='inverse', kappa_guide='amortized',
    niche_path=False, hsic_weight=1e-3,
)


def set_model_configs(graph_data, verbose=False, version='revised', **kwargs):
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

    # Hyperparameter for cell-cell interaction module
    configs.infer_cell_interaction = True
    configs.n_cluster = graph_data.num_clusters
    alpha = getattr(graph_data, 'alpha', None)
    configs.alpha = 1.0 if alpha is None else alpha   # Distance-spread dispersion (the dataset's)
    configs.temperature = 0.3  # Softmax temperature
    configs.omega_prior = 'lognormal'  # Edge weights ∝ (1+d)^-alpha exp(g), g ~ N(0, omega_sigma^2); 'exponential'
    configs.omega_sigma = 1.0
    configs.kappa_prior_scale = 'balanced'  # kappa prior weight per cell: mean n_c / n_c; 'inverse': 1 / n_c
    configs.kappa_guide = 'posterior_solve'  # q(kappa) at its posterior mean given the encoded state; 'amortized'
    configs.kappa_solve_lambda = 0.3  # Noise variance of the encoded state relative to the kappa prior
    configs.kappa_solve_tol = 0.01  # Certified error of the conjugate-gradient solve, relative to ||kappa* - mu||
    configs.kappa_solve_iters = 0  # Step cap (0: the tissue bound)
    configs.niche_path = True  # Direct term from the cell's niche latent z to its state
    configs.hsic_weight = 0.  # HSIC(msg, kappa) penalty (original: 1e-3)
    configs.volume = getattr(graph_data, 'volume', None)  # Registered 3D stack of the auxiliary modality: 3D prior
    assert version in ('revised', 'original'), version
    configs.version = version
    if version == 'original':
        configs.update(ORIGINAL)

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
