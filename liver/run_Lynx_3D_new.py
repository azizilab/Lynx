# %% [markdown]
# #### Run LYNX on the revised 3D alignment
#
# Same pipeline as `run_Lynx_3D.ipynb`, with inputs from `*_alignment_revision` (tables exported by
# `revision_nov_2026/pipeline/15_export_tables.py`).

# %%
import os
import gc
import sys

import numpy as np
import scanpy as sc
import squidpy as sq

import torch
import torch.nn as nn

sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')
import IO, plot, trajectory
import vgae, configs, dataset

# %%
# Hyperparameters
n_subgraphs = 16
n_hidden = 32
n_latent = 6
n_epochs = 500
radius = 50
lr = 1e-2
patience = 20

xenium_path = '../data/xenium_3d/NIH_F5_alignment_revision/'
desi_path = '../data/desi_3d/NIH_F5_40um_alignment_revision/'
sample_id = 'NIH_F5'

sections = [
   'section_01',
   'section_02',
   'section_04',
   'section_05',
   'section_06',
   'section_07',
   'section_08',
]

cluster_key = 'cell_type'

# %% [markdown]
# Iterative run on 3D samples:

# %%
# Joint training
adatas_desi = []
adatas_xenium = []

for section in sections:
    print(f'Loading section: {section}')
    adata_xenium = sc.read_h5ad(os.path.join(xenium_path, 'xenium_'+section+'.h5ad'))
    adata_desi = sc.read_h5ad(os.path.join(desi_path, 'desi_'+section+'.h5ad'))
    adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')
    adatas_desi.append(adata_desi)
    adatas_xenium.append(adata_xenium)

# %%
# Configure dataset and model
graph_data = dataset.HeteroDataset(
    adatas_ref=adatas_xenium,
    adatas_query=adatas_desi,
    n_subgraphs=n_subgraphs,
    r=radius, is_weighted=True, alpha=1.0,
    cluster_key=cluster_key
)

train_configs = configs.set_train_configs(
    n_epochs=n_epochs, lr=lr, patience=patience,
    device=torch.device('cuda')
)

model_configs = configs.set_model_configs(
    graph_data=graph_data,
    c_hidden=n_hidden,
    c_latent=n_latent,
    act=nn.SiLU(),
    infer_cell_interaction=True,
)

model = vgae.HeteroAttnVGAE(model_configs, device=torch.device('cuda'))
model.fit(graph_data, train_configs, DEBUG=True)
gc.collect()

# %% [markdown]
# Save predictions:

# %%
# Per-sample inference
for i, section in enumerate(sections):
    print(f'Running LYNX inference on section: {section}...')
    res = model.evaluate(
        adatas_xenium[i], adatas_desi[i],
        graph_data=graph_data,
        device=torch.device('cpu')
    )

    # Downstream analysis: infer gradient pseudotime (t)
    np.random.seed(42)  # reproducible principal curves
    curve = trajectory.get_curve(adatas_desi[i], epg_lambda=0.01, trim_radius_ratio=0.25)
    trajectory.compute_pseudotime(adatas_desi[i], curve, root_marker=adatas_desi[i].var_names[0])
    sq.pl.spatial_scatter(adatas_desi[i], color='t', cmap='RdBu_r', size=1, img=False)
    plot.disp_trajectory(adatas_desi[i], cmap='RdBu_r')

    curve = trajectory.get_curve(adatas_xenium[i], epg_lambda=0.01, trim_radius_ratio=0.25)
    trajectory.compute_pseudotime(adatas_xenium[i], curve, root_marker='DPT')
    sq.pl.spatial_scatter(adatas_xenium[i], color='t', cmap='RdBu_r', size=20, img=False)
    plot.disp_trajectory(adatas_xenium[i], cmap='RdBu_r')

gc.collect()

# %%
# Save adatas w/ inferred pseudotime
outdir = '../results/liver_3d_alignment_revision/new_seed42/'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

for i, section in enumerate(sections):
    adatas_desi[i].write_h5ad(os.path.join(outdir, f'LYNX_{sample_id}_{section}_desi.h5ad'))
    adatas_xenium[i].write_h5ad(os.path.join(outdir, f'LYNX_{sample_id}_{section}_xenium.h5ad'))
