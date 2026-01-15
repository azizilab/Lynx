## Spatial cluster & trajectory inference via CORAL

# %%
import os
import gc
import sys
import time

import numpy as np
import scanpy as sc
import squidpy as sq
import torch

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from IPython.display import display

sns.set_context('paper')
rcParams.update({'font.family': 'Liberation Sans'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# %%
sys.path.append('../')
sys.path.append('../util/')
import IO, plot

# %%
# Coral related modules
sys.path.pop(sys.path.index('../'))
sys.path.pop(sys.path.index('../util/'))
sys.path.append('../external/CORAL_MOD/')
sys.path.append('../external/CORAL_MOD/coral')

from coral import coral_main, VisCoxDataset
from coral import utils as coral_utils
from coral import utils_simu as coral_utils_simu

# %%
from importlib import reload
reload(coral_main)


# %%
# Simulation
# data_path = '../data/simulation'
# adata_xenium = sc.read_h5ad(os.path.join(data_path, 'xenium_feature_matrix.h5'))
# adata_desi = sc.read_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))

# Try real data
xenium_path = '../data/xenium/'
desi_path = '../data/desi/'
sample_id = 'NIH_F5'
adata_xenium = IO.load_xenium(os.path.join(xenium_path, sample_id), load_img=True)
adata_desi = sc.read_h5ad(os.path.join(desi_path, sample_id+'.h5'))
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')

# %%
# Reformat alignment & preprocess according to CORAL tutorial
# `reference`: high-resolution modality; `query`: low-resolution modality
adata_xenium.X = adata_xenium.X.A.copy()
adata_desi.obsm['spatial'] = adata_desi.obsm['xenium_map'].copy()
combined_expr, ref_coords, one_hot_cell_types, query_indices, query_expr = coral_utils.preprocess_data(
    adata_desi, adata_xenium
)

n_query_features = adata_desi.shape[1]
n_ref_features = adata_xenium.shape[1]
gc.collect()

# %%
dataloader = coral_utils.prepare_local_subgraphs(
    combined_expr,
    ref_coords,
    one_hot_cell_types,
    query_indices,
    query_expr,
    n_neighbors=100
)

# %%
torch.manual_seed(0)
coral_model, _ = coral_main.create_model(
    visium_dim=n_query_features,  # Dim of low-res `query`,
    codex_dim=n_ref_features,  # Dim of hires `ref`,
    cell_type_dim=one_hot_cell_types.shape[1],
    latent_dim=6,
    hidden_channels=32,
    v_dim=1
)

optimizer = torch.optim.Adam(coral_model.parameters(), lr=5e-4)


# %%
# reload(coral_main)
# torch.cuda.empty_cache()
# gc.collect()
# del coral_model

# %%
# Training
device = torch.device('cuda')
coral_model = coral_model.to(device)
t0 = time.perf_counter()
losses = coral_main.train_model(coral_model, optimizer, dataloader, epochs=200, device=device)
t1 = time.perf_counter()
print('Training time:', t1 - t0)


# %%
plt.figure(figsize=(10, 3))
plt.plot(np.arange(len(losses)), losses, '.-')
plt.xlabel('Epochs')
plt.ylabel('CORAL losses')
plt.show()

# %%
# Inference
res = coral_main.generate_and_validate(coral_model, dataloader, device)

# %%
px, py = res[0], res[1]
qz, qv = res[2], res[8]
adata_xenium.obsm['X_z_coral'] = qz
adata_xenium.obsm['X_v_coral'] = qv
adata_xenium.obs['X_v_coral'] = qv
adata_xenium.obs['t'] = (qv-qv.min()) / (qv.max()-qv.min())

# %%
# Check reconstruction (y)
rand_indices = np.random.choice(
    np.arange(adata_xenium.shape[0]*adata_xenium.shape[1]), 10000, replace=False
)
plot.disp_kde_scatter(
    adata_xenium.X.A.flatten()[rand_indices],
    py.flatten()[rand_indices],
    xlabel=r"Ground-truth observation",
    ylabel=r"Reconstructed observation",
    title='Xenium feature reconstruction'
)
del rand_indices
gc.collect()


# %%
px_aggr = np.zeros_like(adata_desi.X)
desi_coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(adata_desi.obsm['spatial'])}
proj_counts = np.zeros(adata_desi.shape[0])

for i in range(adata_xenium.shape[0]):
    proj_coord = adata_xenium.obsm['desi_map'][i]
    idx = desi_coord_to_idx[tuple(proj_coord)]
    px_aggr[idx] += px[i]
    proj_counts[idx] += 1
px_aggr = px_aggr / proj_counts[:, None]


# Check reconstruction (x)
# Note: CORAL produces high-resolution p(x | z), project to "spot"-level DESI by aggr('mean')
rand_indices = np.random.choice(
    np.arange(adata_desi.shape[0]*adata_desi.shape[1]), 10000, replace=False
)
plot.disp_kde_scatter(
    adata_desi.X.flatten()[rand_indices],
    px_aggr.flatten()[rand_indices],
    xlabel=r"Ground-truth observation",
    ylabel=r"Reconstructed observation",
    title='DESI feature reconstruction'
)
del rand_indices
gc.collect()

# %%
# Visualize latent (z)
plot.disp_factor_corr(qz)
plot.disp_spatial_latents(adata_xenium, qz, ncols=3)


# %%
# Evaluation:
# (1). Comparison w/ ground-truth trajectory gradients (\gamma(t))
def _convert_gradients(gradients):
    """TMP: convert ground-truth gradients to 0-1"""
    v = gradients + gradients.min()
    return (v-v.min()) / (v.max()-v.min())

gradients_true = np.load(os.path.join(data_path, 'gradients.npy'))
gradients_true = _convert_gradients(gradients_true)
zonation_true = np.load(os.path.join(data_path, 'zonation.npy'))

gamma_true = gradients_true[
    tuple([adata_xenium.obsm['desi_map'][:, 1], adata_xenium.obsm['desi_map'][:, 0]])
]  # 1D gradient

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))
ax1.imshow(gradients_true, cmap='RdBu_r')
ax1.axis('off')
ax1.set_title('Pseudotime \nGround-truth')
ax2.imshow(zonation_true, cmap='turbo')
ax2.axis('off')
ax2.set_title('Zonation \nGround-truth')

plt.tight_layout()
plt.show()


# %%
plot.disp_kde_scatter(
    gamma_true, adata_xenium.obs['t'], 
    xlabel=r"Ground-truth $\gamma(t)$",
    ylabel=r"CORAL prediction $\gamma(t)$",
    title="Spatial Trajectory"
)

# %%
# (2). Clustering measuresments
# Reference: https://github.com/zou-group/CORAL/blob/main/coral/utils_simu.py

# Currently clustering fails (`z` posterior collapses)

# sc.pp.neighbors(adata_xenium, n_neighbors=64, use_rep='X_z_coral')
# sc.tl.leiden(adata_xenium, resolution=.1)
# sq.pl.spatial_scatter(
#     adata_xenium, color='leiden', 
#     cmap='turbo', size=20, img=False,
#     title=r'Spatial clustering'+'\nCORAL'
# )

# %%
adata_xenium.obs['t'].to_csv('../results/Coral_pseudotime.csv')
np.save('../results/Coral_6.npy', adata_xenium.obsm['X_z_coral'])
np.save('../results/Coral_py.npy', py)
np.save('../results/Coral_px.npy', px)
