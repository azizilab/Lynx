# Annotations for CMA spatial axis (continuous) & layers (discrete)
# References:
#  - Yavon et al., 2024: https://www.nature.com/articles/s41586-024-07944-6 (CMA)
#  - Liao et al., 2023: https://www.biorxiv.org/content/10.1101/2023.04.28.538364v1.full.pdf (Mouse thymus dataset)
#  - Nitta & Takayanagi, 2021: https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2020.620894/full (annotation)

# %%
import os
import sys
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

# %%
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from matplotlib import rcParams
sns.set_context('paper')
rcParams.update({'font.family': 'Liberation Sans'})
rcParams.update({'font.size': 15})
rcParams.update({'figure.dpi': 300})
rcParams.update({'savefig.dpi': 300})

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# %%
sys.path.append('..')
sys.path.append('../models/')
sys.path.append('../util')

import utils, IO

# %%
# Remove ribosomal genes & high-percentile mitochrondriol genes
def preprocess_rna(
    adata_raw, 
    min_cells=1, 
    min_count_percentile=10, 
    n_top_genes=2000, 
    markers=[],
    show=False
):
    """Remove ribosomal genes, keep only HVGs or specified markers"""
    adata = adata_raw.copy()
    adata.var['rb'] = adata.var_names.str.startswith(('RP', 'Rp', 'rp'))
    mask_gene = np.logical_not(adata.var['rb'])
    adata = adata[:, mask_gene]

    libsize = adata.X.A.sum(1)
    min_counts = np.percentile(libsize, min_count_percentile)
    if show:
        adata.obs['artifact'] = (libsize <= min_counts)
        sc.pl.spatial(adata, color='artifact', spot_size=100, title='Stereo-seq')
        adata.obs.drop('artifact', axis=1, inplace=True)

    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # HVGs
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, inplace=True)
    sc.pp.log1p(adata_norm)

    sc.pp.highly_variable_genes(adata_norm, n_top_genes=n_top_genes, inplace=True)
    mask = np.logical_or(
        adata_norm.var['highly_variable'], 
        np.isin(adata_norm.var_names, markers)
    )
    adata = adata[:, mask]

    return adata

def preprocess_protein(
    adata_raw, 
    min_cells=1, 
    min_count_percentile=10, 
    CLR=True,
    show=False
):
    r"""
    Normalize intensities via either
    centered log-ratio (CLR) or arcsinh transformation
    """
    adata = adata_raw.copy()    
    libsize = adata.X.A.sum(1)
    min_counts = np.percentile(libsize, min_count_percentile)
    if show:
        adata.obs['artifact'] = (libsize <= min_counts)
        sc.pl.spatial(adata, color='artifact', spot_size=100, title='CITE-seq')
        adata.obs.drop('artifact', axis=1, inplace=True)

    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    expr = adata.X.A.copy()
    if CLR:
        geometric_mean = np.exp(np.mean(np.log1p(expr), axis=1, keepdims=True)) 
        norm_expr = np.log1p(expr / geometric_mean)
    else:
        norm_expr = np.arcsinh(expr / 10)
    adata.X = norm_expr

    return adata
    

def rename_features(adata):
    # Unify channel labels
    channels = [s.replace("-", "_") for s in adata.var_names.copy()]
    channels_unified = []
    for chan in channels:
        if 'Mouse_Rat_Human_' in chan:
            channels_unified.append(chan.split('Mouse_Rat_Human_')[-1])
        elif 'mouse_rat_human_' in chan:
            channels_unified.append(chan.split('mouse_rat_human_')[-1])
        elif 'Mouse_Rat_' in chan:
            channels_unified.append(chan.split('Mouse_Rat_')[-1])
        elif 'mouse_rat_' in chan:
            channels_unified.append(chan.split('mouse_rat_')[-1])
        elif 'Mouse_Human_' in chan:
            channels_unified.append(chan.split('Mouse_Human_')[-1])
        elif 'mouse_human_' in chan:
            channels_unified.append(chan.split('mouse_human_')[-1])
        else:
            channels_unified.append(chan.partition('_')[-1])
    adata.var['features'] = channels_unified
    adata.var = adata.var.set_index('features')

    return None


# %%
# Load dataset
data_path = '../data/thymus/'
sample_ids = sorted([
    f for f in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, f))
])

# %%
sample_id = sample_ids[0]  # Work on first sample
adata_rna = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_RNA.h5ad'))
adata_protein = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata_ADT.h5ad'))

# Append dummy spatial metadata
IO.load_spatial_metadata(adata_rna)
IO.load_spatial_metadata(adata_protein)


# %%
tec_markers = [
    'Psmb11', 'Ly75', 'Ccl25', 'H2-Aa', 'H2-Ab1',   # Pan-cTEC 
    'Tbata', 'Tp53aip1', 'Dll4', # cTEC subtypes
    'Dlk2', 'Igfbp5', 'Igfbp6', 'Ccn2', 'Ccl2', 'Krt15', 'Itga6', 'Mki67',  # mcTEC subtypes (KTR15 -> Krt15)
    'Epcam', # Pan-mTEC
    'Ascl1', 'Ccl21a',   # mTECI (CCL21 -> Ccl21a)
    'Aire', 'Fezf2', 'Crip1',  # mTECII
    'Slpi', 'Ivl', 'Krt10', 'Cdkn2a'  # mTEC subtypes
]

marophage_markers = [
    'Cd68', 'Cd163', 'Cd11b', 'Cd11c',
    'Timd4', 'Hpgd', 'Serpinb6a', 'Slc40a1', 'Cd81',  # Cortex-enriched Timd4+ markers
    'Cx3cr1', 'Ctsz', 'Cd63', 'Pmepa1',' Zmynd15', # Medulla-enriched Cx3cr1+ markers
]

immune_markers = [
    'Cd3d', 'Cd3e', 'Cd4', 'Cd8a', 'Cd8b1',  # T cells
    'Cd19', 'Ptprc', 'Ighd',  # B cells
    'Cd5', 'Cd27', 'Cd44'   # General thymocytes
]

markers = list(tuple(tec_markers + marophage_markers + immune_markers))

# %%
# Preprocess: HVGs / normalize intensities
adata_rna = preprocess_rna(adata_rna, n_top_genes=3000, markers=markers, min_count_percentile=10, show=True)
adata_protein = preprocess_protein(adata_protein, CLR=False, min_count_percentile=10, show=True)
common_bins = np.intersect1d(adata_rna.obs_names, adata_protein.obs_names)
adata_rna, adata_protein = adata_rna[common_bins], adata_protein[common_bins]


# %%
# Check feasibility of OrganAxis / Heat diffusion

# %%
# sc.pp.normalize_total(adata_rna)
# sc.pp.log1p(adata_rna)
# sc.pp.pca(adata_rna)
# sc.pp.neighbors(adata_rna, n_neighbors=30)
# sc.tl.umap(adata_rna)

 
# %%
# Notes: 
# - Medulla (inner): mature thymocytes + stromal (low density)
# - Cortico-Medullary Junction: blood-vessel righ, immigration of T-precursor; emmigration of SP thymocytes
# - Coretex (outer): immature thymocytes + stromal (high density) 

# Layer-specific markers
# Level 1: Medulla, Cortex, Capsule

# - Medulla: 
#   - Cd5 (neg regulator of TCR signaling in inmature thymocytes)
#   - Cd44 (invariant NKT cells)
#   - Cx3cr1, Ctsz, Cd63, Pmepa1, Zmynd15 (macrophages)

# - CMJ:
#   - Cd169, F4/80, Cd163 (Macrophages)

# - Cortex:
#   - Cd4, Cd8
#   - Timd4, Hpgd, Serpinb6a, Slc40a1, Cd81 (Macrophages)

# - Capsule (cortex-edge):
#   - Cd29

# %%
# Annotate broad CMA layers (Medulla - Cortex - Capsule)
rename_features(adata_protein)
sc.pl.spatial(adata_protein, color=['CD29', 'CD45R_B220'], size=100, cmap='turbo')

# %%
# Sample-specific annotation: e.g. Try CD5
mask_medulla_prelim = (
    adata_protein[:, 'CD8a'].X.flatten() <= np.percentile(adata_protein[:, 'CD8a'].X, 15)
).flatten()

mask_capsule_prelim = np.logical_or(
    adata_protein[:, 'CD29'].X > np.percentile(adata_protein[:, 'CD29'].X, 85),
    adata_protein[:, 'CD45R_B220'].X > np.percentile(adata_protein[:, 'CD45R_B220'].X, 85)
).flatten()

mask_cortex_prelim = (
    adata_protein[:, 'CD4'].X.flatten() > np.percentile(adata_protein[:, 'CD4'].X, 50)
).flatten()

# %%
cma_layers = np.array(['Cortex'] * len(adata_protein), dtype='object')

mask_capsule = ((mask_capsule_prelim) & (~mask_cortex_prelim))
mask_medulla = ((mask_medulla_prelim) & (~mask_capsule))
cma_layers[mask_capsule] = 'Capsule'
cma_layers[mask_medulla] = 'Medulla'

adata_rna.obs['CMA_layer'] = cma_layers
adata_protein.obs['CMA_layer'] = cma_layers
sc.pl.spatial(adata_protein, color='CMA_layer', size=100)

# %%
# Run OrganAxis
from scipy.spatial import KDTree

def calculate_cma(
    adata, use_rep='CMA_layer', 
    w1=.8, w2=.2, 
    vmin=-.8, vmax=.8, k=10
):
    """
    Calculate the Cortex-Medulla Axis (CMA) for spatial observations in an AnnData object.
    Reference: https://www.nature.com/articles/s41586-024-07944-6#Sec46

    Parameters:
    - adata: AnnData object with spatial coordinates and annotations for 'Capsule', 'Cortex', and 'Medulla'.
    - k: Number of nearest neighbors to consider for distance calculations.

    Returns:
    - cma_values: Array of CMA values for each observation in adata.
    """
    # Extract spatial coordinates
    coords = adata.obsm['spatial']

    # Identify indices for each structure
    capsule_indices = np.where(adata.obs[use_rep] == 'Capsule')[0]
    cortex_indices = np.where(adata.obs[use_rep] == 'Cortex')[0]
    medulla_indices = np.where(adata.obs[use_rep] == 'Medulla')[0]

    # Build KD-Trees for each structure
    capsule_tree = KDTree(coords[capsule_indices])
    cortex_tree = KDTree(coords[cortex_indices])
    medulla_tree = KDTree(coords[medulla_indices])

    # Initialize array to hold CMA values
    cma_values = np.zeros(coords.shape[0])

    # Calculate mean distances to each structure
    for i, point in enumerate(coords):
        d_capsule, _ = capsule_tree.query(point, k=k)
        d_cortex, _ = cortex_tree.query(point, k=k)
        d_medulla, _ = medulla_tree.query(point, k=k)

        mu_capsule = np.mean(d_capsule)
        mu_cortex = np.mean(d_cortex)
        mu_medulla = np.mean(d_medulla)

        # Calculate H1(Medulla vs. Cortex) & H2 (Cortex vs. Capsule)
        H1 = (mu_cortex - mu_medulla) / (mu_cortex + mu_medulla)
        H2 = (mu_capsule - mu_cortex) / (mu_capsule + mu_cortex)

        # Combine H1 and H2 to get CMA value
        cma_values[i] = w1 * H1 + w2 * H2

    # Rescale to [vmin, vmax]
    cma_min, cma_max = np.min(cma_values), np.max(cma_values)
    cma_values = vmin + (cma_values - cma_min) * ((vmax-vmin) / (cma_max-cma_min))

    return cma_values


def cma_to_bins(adata):
    """
    Convert continuous CMA to bins
    Reference: https://www.nature.com/articles/s41586-024-07944-6#Sec46 (suppl, table 8)
    """
    assert 'CMA' in adata.obs_keys(), "Please run OrganAxis CMA first"
    cma = adata.obs['CMA'].values.copy()

    # Major bins
    is_capsule = np.logical_and(cma < -0.2, adata.obs['CMA_layer'] == 'Capsule')
    is_cortex = np.logical_and(-0.2 <= cma, cma < 0.2)
    is_cmj = np.logical_and(0.2 <= cma, cma < 0.45)
    is_medulla = (cma >= 0.45)

    major_bin = np.array(['Cortex']*len(cma), dtype='object')
    major_bin[is_capsule] = 'Capsule'
    major_bin[is_cortex] = 'Cortex'
    major_bin[is_cmj] = 'CMJ'
    major_bin[is_medulla] = 'Medulla'

    adata.obs['CML_Major'] = major_bin

    # Minor bins
    is_capsular = np.logical_and(cma < -0.2, adata.obs['CMA_layer'] == 'Capsule')

    is_cortical_1 = np.logical_and(-0.2 <= cma, cma < -0.1)
    is_cortical_2 = np.logical_and(-0.1 <= cma, cma < 0.05)
    is_cortical_3 = np.logical_and(0.05 <= cma, cma < 0.2)
    is_cortical_cmj = np.logical_and(0.2 <= cma, cma < 0.32)

    is_medullary_cmj = np.logical_and(0.32<= cma, cma < 0.45)
    is_medullary_1 = np.logical_and(0.45 <= cma, cma < 0.6)
    is_medullary_2 = np.logical_and(0.6 <= cma, cma < 0.7)
    is_medullary_3 = (cma >= 0.7)

    minor_bin = np.array(['Subcapsular']*len(cma), dtype='object')
    minor_bin[is_capsular] = 'Capsular'
    minor_bin[is_cortical_1] = 'Cortical 1'
    minor_bin[is_cortical_2] = 'Cortical 2'
    minor_bin[is_cortical_3] = 'Cortical 3'
    minor_bin[is_cortical_cmj] = 'Cortical CMJ'
    minor_bin[is_medullary_cmj] = 'Medullary CMJ'
    minor_bin[is_medullary_1] = 'Medullary 1'
    minor_bin[is_medullary_2] = 'Medullary 2'
    minor_bin[is_medullary_3] = 'Medullary 3'

    adata.obs['CML_Minor'] = minor_bin

    return None
 
# %%
cma = calculate_cma(adata_protein, w1=.5, w2=.5, k=10)
adata_protein.obs['CMA'] = cma
sc.pl.spatial(adata_protein, color='CMA', cmap='RdBu_r', size=100)

plt.figure()
plt.plot(np.arange(len(cma)), np.sort(cma))
plt.show()

# %%
cma_to_bins(adata_protein)
sc.pl.spatial(adata_protein, color='CML_Major', size=100)
sc.pl.spatial(adata_protein, color='CML_Minor', size=100)

# %%
# Save with continuous & discrete annotations
adata_rna.obs['CMA'] = adata_protein.obs['CMA'].copy()
adata_rna.obs['CML_Major'] = adata_protein.obs['CML_Major'].copy()
adata_rna.obs['CML_Minor'] = adata_protein.obs['CML_Minor'].copy()

sc.set_figure_params(scanpy=True, fontsize=15)
sc.pl.spatial(adata_rna, color=['CMA', 'CML_Major', 'CML_Minor'], cmap='RdBu_r', legend_fontsize=12, size=100)


# %%
adata_rna.X = adata_rna.X.A
adata_rna.write_h5ad(os.path.join(data_path, sample_id, 'adata_rna.h5'))
adata_protein.write_h5ad(os.path.join(data_path, sample_id, 'adata_protein.h5'))

# %%
