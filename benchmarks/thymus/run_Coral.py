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
sys.path.pop(sys.path.index('../'))
sys.path.pop(sys.path.index('../util/'))
sys.path.append('../external/CORAL/')
sys.path.append('../external/CORAL/coral')

from coral import coral_main, VisCoxDataset
from coral import utils as coral_utils
from coral import utils_simu as coral_utils_simu

# %%
%load_ext autoreload
%autoreload 2


# %%
import random

seed = 42
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %%
# # TODO (Debug CORAL): Load raw RNA & Protein dataset, perform leiden cluster
# data_path = '../data/thymus/Mouse_Thymus1/'
# # adata_rna = sc.read_h5ad(os.path.join(data_path, 'adata_RNA.h5ad'))  # Preprocessed, filtered out low-quality cells
# # adata_rna.X = adata_rna.X.toarray().copy()
# adata_rna = sc.read_h5ad(os.path.join(data_path, 'adata_rna.h5'))

# adata_protein = sc.read_h5ad(os.path.join(data_path, 'adata_ADT.h5ad'))  # Raw
# adata_protein.X = adata_protein.X.toarray().copy()

# # Append dummy spatial metadata
# IO.load_spatial_metadata(adata_rna)
# IO.load_spatial_metadata(adata_protein)
# adata_protein = adata_protein[adata_rna.obs_names]  # Filter out low-quality bins

# print(adata_rna.shape, adata_protein.shape)

# %%
# Load dataset
data_path = '../data/thymus/'
sample_ids = sorted([
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
])

sample_id = sample_ids[0]
adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))


# %%
# Leiden clustering
adata_protein_copy = coral_utils_simu.plot_pca_cluster(adata_protein, res=0.62, size=1, return_=True)
adata_rna.obs['cell_type'] = adata_protein_copy.obs['cluster'].copy()
adata_protein.obs['cell_type'] = adata_protein_copy.obs['cluster'].copy()
del adata_protein_copy

# %%
# Optional: Downsample RNA expression to 'visium' level
color_list = [
    '#8ecae6',
    '#219ebc',
    '#126782',
    '#023047',
    '#9f86c0',                                   
    '#d4e09b',
    '#ff9f1c',
    '#dc2f02',
    '#dc2f02',
    '#9d0208',
    '#f08080',
    '#f94144',
    '#f3722c',
    '#f8961e',
    '#f9c74f',
    '#90be6d',
    '#43aa8b',
    '#577590',  
    '#226CE0',
    '#534B62'
]
block_data, block_spatial = coral_utils_simu.downsample_and_plot_spatial_data_smooth(
    adata_rna,
    color_list=color_list,
    n_blocks_x=40, 
    n_blocks_y=40,
    size = 3,
    figsize=(5,5),
    invert_yaxis=False
)
block_data = block_data[:,0,:]
# Create a new AnnData object for the downsampled data
adata_rna_ds = sc.AnnData(X=block_data)
adata_rna_ds.obsm['spatial'] = block_spatial
adata_rna = adata_rna_ds.copy()

# %%
# subset HVGs
adata_rna_norm = adata_rna.copy()
sc.pp.normalize_total(adata_rna_norm, target_sum=1e4)
sc.pp.log1p(adata_rna_norm)
sc.pp.highly_variable_genes(adata_rna_norm, n_top_genes=3000)

adata_rna = adata_rna[:, adata_rna_norm.var['highly_variable']]

# %%
sc.pl.spatial(adata_protein, color='cell_type', spot_size=100)


# %%
combined_expr, protein_coords, one_hot_cell_types, spot_indices, rna_expr = coral_utils.preprocess_data(adata_rna, adata_protein)
    
dataloader = coral_utils.prepare_local_subgraphs(
    combined_expr, protein_coords, one_hot_cell_types, 
    spot_indices, rna_expr, n_neighbors=40
)    

# %%
# Define model
device = torch.device('cuda')
model, optimizer = coral_main.create_model(
    visium_dim = adata_rna.shape[1],
    codex_dim = adata_protein.shape[1],
    cell_type_dim=one_hot_cell_types.shape[1],
    latent_dim=32, 
    hidden_channels=128, 
    v_dim = 1
)
model = model.to(device)

# %%
# Training
coral_main.train_model(model, optimizer, dataloader, epochs=100, device=device)

# %%
# Inference
(generated_expr, 
 generated_protein, latent_rep, 
 locations,
 rna_true,
 protein_true, 
 attn_weights_all, 
 edges_all,
 v_values, 
 cell_types) = coral_main.generate_and_validate(model, dataloader, device) 


# %%
v_values = v_values.squeeze()
gamma_coral = (v_values-v_values.min()) / (v_values.max()-v_values.min())
adata_rna.obsm['Coral_z'] = latent_rep
adata_rna.obsm['Coral_v'] = gamma_coral



# %%
adata_rna.obs['t'] = gamma_coral.copy()
ax = sq.pl.spatial_scatter(
    adata_rna, color='t', 
    cmap='RdBu_r', size=100, img=False,
    return_ax=True
)
ax.set_title(r'Inferred spatial gradient $(t)$ - CORAL', fontdict={'fontsize': 14})

# sc.pp.neighbors(adata_rna, use_rep='Coral_z')
# sc.pp.neighbors(adata_rna)
# sc.tl.umap(adata_rna)
# sc.pl.umap(adata_rna, color='t')

# %%

# %%
gamma_true = adata_rna.obs['CMA'].values
gamma_true = (gamma_true-gamma_true.min()) / (gamma_true.max()-gamma_true.min())
adata_rna.obs['CMA']

plot.disp_kde_scatter(
    gamma_true, gamma_coral, ss_ratio=1.,
    xlabel=r"Ground-truth $\gamma(t)$",
    ylabel=r"CORAL prediction $\gamma(t)$",
    title="CMA\n CORAL vs. Ground-truth"
)

# %%
adata_rna.obs['library_size'] = adata_rna.X.sum(1)
sc.pl.spatial(
    adata_rna, color='library_size', size=100, cmap='magma'
)

# %%
plot.disp_kde_scatter(
    gamma_true, -adata_rna.obs['library_size'].values, 
    xlabel=r"Ground-truth $\gamma(t)$",
    ylabel=r"-library size",
    title="CMA"
)


# %%
# Spatial clustering
sc.pp.neighbors(adata_protein, use_rep='Coral_z')
sc.tl.leiden(adata_protein, resolution=0.91, random_state=42)
sc.pl.spatial(adata_protein, color='leiden', spot_size=100, title='Spatial Clustering\n CORAL')

# %%
np.save('../results/thymus/CORAL_z_Mouse_Thymus1.npy', adata_protein.obsm['Coral_z'])
np.save('../results/thymus/CORAL_v_Mouse_Thymus1.npy', v_values)


