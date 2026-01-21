# %%
import os
import gc
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.append('../../')
sys.path.append('../../util/')
import IO

%load_ext autoreload
%autoreload 2

import gaston
from glmpca import glmpca
from gaston import neural_net,cluster_plotting, dp_related, segmented_fit, model_selection
from gaston import binning_and_plotting, isodepth_scaling, run_slurm_scripts, parse_adata
from gaston import spatial_gene_classification, plot_cell_types, filter_genes, process_NN_output
from kneed import DataGenerator, KneeLocator

# %%
# Dataset specs
data_path = '../data/thymus/'
outdir = '../figures/'
sample_id = 'Mouse_Thymus1'

adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))

# %%
# Train GASTON model
def load_data(adata):
    counts_mat = adata.X.toarray() if type(adata.X) != np.ndarray else adata.X
    coords_mat = adata_rna.obsm['spatial']
    gene_labels = np.array(adata.var.index)
    return counts_mat, coords_mat, gene_labels

def run_glmpca(
    save_dir,
    counts_mat, 
    num_dims=20, 
    penalty=1, 
    num_iters=30, 
    num_genes=10000,
    eps=1e-4, 
):
    counts_mat_glmpca = counts_mat[:,np.argsort(np.sum(counts_mat, axis=0))[-num_genes:]]
    glmpca_res = glmpca.glmpca(counts_mat_glmpca.T,
                               num_dims, 
                               fam="poi",
                               penalty=penalty, 
                               verbose=True,
                               ctl = {"maxIter":num_iters, "eps":eps, "optimizeTheta":True})
    A = glmpca_res['factors']
    np.save(os.path.join(save_dir, 'glmpca.npy'), A)
    return A

def train_model(
    save_dir,
    coords_mat, 
    num_epochs=100000, 
    checkpoint=500, 
    optimizer='adam', 
    num_restarts=5, 
    use_gpu=True, 
):
    out_dir = os.path.join(save_dir, 'models/')
    
    A = np.load(os.path.join(save_dir, 'glmpca.npy'))
    S = coords_mat
    
    # z-score normalize S and A
    S_torch, A_torch = neural_net.load_rescale_input_data(S,A)
    if use_gpu:
        S_torch = S_torch.to('cuda')
        A_torch = A_torch.to('cuda')

    # NN parameters
    # architectures are encoded as list, eg [20,20] means two hidden layers of size 20 hidden neurons
    isodepth_arch = [20,20]        # architecture for isodepth neural network d(x,y) : R^2 -> R 
    expression_fn_arch = [20,20]   # architecture for 1-D expression function h(w) : R -> R^G
    
    seed_list = range(num_restarts)
    for seed in seed_list:
        print(f'training neural network for seed {seed}')
        out_dir_seed = f"{out_dir}/rep{seed}"
        os.makedirs(out_dir_seed, exist_ok=True)

        if use_gpu:
            model = neural_net.GASTON(A_torch.shape[1], isodepth_arch, expression_fn_arch) 
            model.to('cuda')
        else:
            model = None
            
        mod, loss_list = neural_net.train(S_torch, A_torch, gaston_model=model,
                                          S_hidden_list=isodepth_arch, A_hidden_list=expression_fn_arch, 
                                          epochs=num_epochs, checkpoint=checkpoint, 
                                          save_dir=out_dir_seed, optim=optimizer, seed=seed, save_final=True)

def calc_isodepth(
    save_dir, 
    num_layers=None, 
    start_from=1,
    kmax=10, 
    normalize=True, 
    use_gpu=True
):
    gaston_model, A, S = process_NN_output.process_files(os.path.join(save_dir, 'models/'))
    if use_gpu:
        gaston_model.to('cpu')

    if num_layers:
        num_layers = num_layers
    else:
        ll_list = model_selection.get_ll_list(gaston_model, A, S, num_buckets=150, kmax=kmax)
        kneedle = KneeLocator(np.arange(start_from,len(ll_list)+1), ll_list[start_from-1:], curve="convex", direction="decreasing")
        kneedle_opt = kneedle.knee
        print(f'Kneedle number of domains: {kneedle_opt}')
        num_layers = kneedle_opt
        
    gaston_isodepth, gaston_labels = dp_related.get_isodepth_labels(gaston_model, A, S, num_layers)

    if normalize:
        min_val = gaston_isodepth.min()
        max_val = gaston_isodepth.max()
        gaston_isodepth = (gaston_isodepth - min_val)/(max_val - min_val)
        
    return gaston_isodepth, gaston_labels

def run_gaston(adata, save_dir, num_dims=20, num_layers=3, return_new_adata=True, use_gpu=True):
    os.makedirs(save_dir, exist_ok=True)
    
    # counts_mat, coords_mat, _ = load_data(adata)

    # # GLM-PCA
    # run_glmpca(save_dir, counts_mat, num_dims)

    # # Train model
    # train_model(save_dir, coords_mat)

    # Calculate isodepth
    isodepth, labels = calc_isodepth(save_dir, num_layers)

    # Return
    if return_new_adata:
        adata.obs['gaston_isodepth'] = isodepth
        adata.obs['gaston_labels'] = labels
        return adata, isodepth, labels
    return isodepth, labels

# %%
ndims = 10
isodepth, seg = run_gaston(adata_rna, save_dir='../../results/thymus/gaston/', num_dims=ndims, num_layers=4, return_new_adata=False, use_gpu=True)

# %%
# Save GASTON isodepth
np.save('../../results/thymus/GASTON_thymus_isodepth.npy', isodepth)
np.save('../../results/thymus/GASTON_thymus_seg.npy', seg)


# %%