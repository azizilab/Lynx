# %%

# Synthetic data generation workflow:
# (1). Real Xenium ROI patch, annotate PVs & CVs
# (2). Generate ground-truth "gradients" from heat-based diffusion
# (3). Generate synthetic DESI
# (4). Generate ground-truth z with mixing function from discretized "gradients"

# %%
import os
import gc
import sys
import json

import tifffile
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import squidpy as sq

import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
from skimage.transform import rescale

sys.path.append('..')
from util import IO, utils, zonation

# %%
from importlib import reload

# %%
def load_xenium_img(dir, sample_id):
    adata = IO.load_xenium(os.path.join(dir, sample_id), load_img=True)

    with open(os.path.join(dir, sample_id, 'experiment.xenium'), 'r') as ifile:
        scalefactor = json.load(ifile)['pixel_size'] 

    img = np.squeeze(adata.uns['spatial'][sample_id]['images']['hires'])
    img = rescale(img, scale=scalefactor).astype(np.float32)
    img = (img-img.min()) / (img.max()-img.min())
    adata.uns['spatial'][sample_id]['images']['hires'] = img

    return adata


# %%
sample_id = 'NIH_F3'
xenium_path = '../data/xenium/'
ab_path = '../data/integrated/antibody/'
outdir = '../data/simulation/'

# %%
# ---------------
#   Simulation
# ---------------

# %%
# Load sample Xenium
adata = load_xenium_img(xenium_path, sample_id)
coords = adata.obs[['x_centroid', 'y_centroid']]
img = adata.uns['spatial'][sample_id]['images']['hires']
antibody_img = tifffile.imread(os.path.join(ab_path, sample_id+'.ome.tif'))[1:]

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

# Select Xenium ROI
ystart, yend = 100, 3100
xstart, xend = 3000, 6000
xspan, yspan = xend-xstart, yend-ystart

# %%
roi_img = np.zeros((antibody_img.shape[0]+1, yspan, xspan))
roi_img[0] = img[ystart:yend, xstart:xend]
for i, chan in enumerate(antibody_img):
    roi_img[i+1] = antibody_img[i, ystart:yend, xstart:xend]
del chan

# fig, axes = plt.subplots(4, 1, figsize=(20, 30))
# axes[0].imshow(ab_img[0][ystart:yend, xstart:xend], cmap='magma')
# axes[0].axis('off')
# axes[1].imshow(ab_img[1][ystart:yend, xstart:xend], cmap='magma')
# axes[1].axis('off')
# axes[2].imshow(ab_img[2][ystart:yend, xstart:xend], cmap='magma')
# axes[2].axis('off')
# axes[3].imshow(ab_img[3][ystart:yend, xstart:xend], cmap='magma')
# axes[3].axis('off')

# plt.tight_layout()
# plt.show()

# %%
IO.save_annot_tif(
    os.path.join(outdir, 'morphology.ome.tif'),
    img=roi_img,
    annots=['DAPI', 'GS', 'CYP', 'ASS1', 'Col1']
)

# Heuristic steps: QuPath - ROI fluorescence image -> Vein annotations

# %%

# Simulate ground-truth gradients on rescaled (zoom-in view) low-res image
def simulate_lowres_zonation(
    mask, cv_val=1, pv_val=2,
    scale=1/10, dilate_pixel=5
):
    """
    Simulate zonation mask for downscaled DESI view
    """
    cv_mask = morphology.dilation(
        (mask == cv_val).astype(np.int8),
        footprint=morphology.disk(radius=dilate_pixel)
    )

    pv_mask = morphology.dilation(
        (mask == pv_val).astype(np.int8),
        footprint=morphology.disk(radius=dilate_pixel)
    )

    hires_mask = np.zeros_like(mask).astype(np.int8)
    hires_mask[cv_mask == 1] = 1
    hires_mask[pv_mask == 1] = -1

    lowres_mask = rescale(
        hires_mask, 
        scale=scale, 
        preserve_range=True
    ).astype(np.int8)

    return lowres_mask

# %%
# Load manual annotations (1: CV mask, 2: PV mask)
scale = 1/20
n_zones = 7
vein_mask = tifffile.imread(os.path.join(outdir, 'vein_annotations.tif'))

lowres_vein_mask = simulate_lowres_zonation(
    vein_mask, scale=scale, dilate_pixel=50
)

plt.figure()
plt.imshow(lowres_vein_mask, cmap='RdBu_r')
plt.show()

# -------------------
#   Simulate \gamma
# -------------------

# %%
# Graph-based diffusion simulation
gradient_model = zonation.HeatDiffusion(vein_prior=lowres_vein_mask, ndim=2)
_, _ = gradient_model.get_interior_U()
gradients = gradient_model.infer_zone_dynamics()
lobule_layers = gradient_model.infer_zones(n_layers=n_zones)

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True, dpi=200)
ax1.imshow(lowres_vein_mask,  cmap='RdBu_r')
ax1.set_title("CV/PV before diffusion")
ax1.axis('off')

ax2.imshow(gradients, cmap='RdBu_r')
ax2.set_title('Zonation gradient after diffusion')
ax2.axis('off')

im = ax3.imshow(lobule_layers, cmap='turbo')
fig.colorbar(im, ax=ax3, fraction=0.02)
ax3.set_title('Lobule layers along PV-CV gradient')
ax3.axis('off')
fig.show()

del gradient_model
gc.collect()

# %%
np.save(os.path.join(outdir, 'gradients.npy'), gradients)
np.save(os.path.join(outdir, 'zonation.npy'), lobule_layers)

# %%
adata = IO.load_xenium(os.path.join(xenium_path, sample_id), raw_count=True)
coords = adata.obs[['x_centroid', 'y_centroid']]

# %%
# Save Xenium expressions mapped to the ROI
condition = np.logical_and(
    (xstart < coords['x_centroid']) & (xend > coords['x_centroid']+1/scale),
    (ystart < coords['y_centroid']) & (yend > coords['y_centroid']+1/scale),
)  

adata_roi = adata[condition]
adata_roi.obs.loc[:, 'x_centroid'] = adata_roi.obs['x_centroid'] - xstart
adata_roi.obs.loc[:, 'y_centroid'] = adata_roi.obs['y_centroid'] - ystart

# Store mappable hi-res -> low-res coords
lowres_coords = (adata_roi.obs[['x_centroid', 'y_centroid']]*scale).round().astype(np.int32)

adata_roi.obsm['spatial'] = adata_roi.obs[['x_centroid', 'y_centroid']].values
adata_roi.obsm['desi_map'] = lowres_coords.values
adata_roi.write_h5ad(os.path.join(outdir, 'xenium_feature_matrix.h5'))

del adata, adata_roi, condition
gc.collect()

# ---------------
#   Simulate U
# ---------------


# %%
# Simulate correlated high-res DESI features from
# ground-truth zonation gradients

import torch
import torch.nn as nn
from scipy.stats import bernoulli


class FeatureGenerator(nn.Module):
    def __init__(self, n_features):
        super(FeatureGenerator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, n_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    

def simulate_sinusoidal_noise(in_channels, height, width, amplitude=.1, frequency=.5):
    """
    Usage: e.g. sine_noise = simulate_sinusoidal_noise(in_channels=20, width=150)
    """
    np.random.seed(0)
    sinusoidal_noise = np.zeros((in_channels, height, width), dtype=np.float32)
    
    # Generate sinusoidal patterns for each channel
    for c in range(in_channels):
        orientation = np.random.choice(['horizontal', 'vertical', 'diag1', 'diag2'])
        x = np.linspace(0, 2 * np.pi * frequency, height)
        y = np.linspace(0, 2 * np.pi * frequency, width)
        xx, yy = np.meshgrid(x, y)
        intercept = np.random.uniform(-1, 1)
        
        if orientation == 'horizontal':
            sinusoidal_noise[c] = np.sin(xx + intercept)
        elif orientation == 'vertical':
            sinusoidal_noise[c] = np.sin(yy + intercept) + intercept
        elif orientation == 'diag1':
            sinusoidal_noise[c] = np.sin(xx + yy + intercept) 
        else:
            sinusoidal_noise[c] = np.sin(-xx + yy + intercept)

    return 1/4*amplitude*sinusoidal_noise


def simulate_desi_img(gradients, n_features, dropout_rate=.1, sine_amplitude=.1, eps=.01):
    h, w = gradients.shape
    x_in = torch.tensor(gradients).float()
    x_in = x_in.unsqueeze(0).unsqueeze(0)  # dim: [N, C, H, W]

    generator = FeatureGenerator(n_features)
    features = generator(x_in).squeeze().detach().cpu().numpy()
    features = utils.norm_by_channel(features)
    features[np.isnan(features)] = 0

    sine_noises = simulate_sinusoidal_noise(n_features, h, w, amplitude=sine_amplitude)

    # Random noise, dropout & gradients
    for i, feature in enumerate(features):
        dropout = bernoulli.rvs(p=1-dropout_rate, size=h*w)
        dropout = dropout.reshape(h, -1)
        epsilon = np.random.randn(h, w) * eps
        
        feature = feature*dropout + epsilon + sine_noises[i]
        feature[feature < 0] = 0
        features[i] = (feature-feature.min()) / (feature.max()-feature.min())
    
    features[np.isnan(features)] = 0
    return features

# %%
n_features = 100
desi_img = simulate_desi_img(
    gradients, n_features=n_features, 
    dropout_rate=.05, sine_amplitude=.5, eps=.1
)

# %%
# Check complexity
nrows = 3
ncols = 5
rand_indices = np.random.choice(n_features, nrows*ncols, replace=False)

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
idx = 0
for r in range(nrows):
    for c in range(ncols):
        axes[r, c].imshow(desi_img[rand_indices[idx]], cmap='magma')
        axes[r, c].axis('off')
        idx += 1

plt.tight_layout()
plt.suptitle('Synthetic DESI channels', fontsize=30, y=1.03)
plt.show()
del idx, r, c, nrows, ncols, rand_indices

# %%
desi_features = desi_img.reshape(n_features, -1).T
desi_corr = np.corrcoef(desi_features.T)
sns.clustermap(desi_corr, cmap='RdBu_r')

adata_desi = sc.AnnData(desi_features)
sc.pp.pca(adata_desi)
sc.pl.pca_variance_ratio(adata_desi)

# %%
# Save DESI image
IO.save_annot_tif(
    file=os.path.join(outdir, 'DESI_img.ome.tif'),
    img=desi_img,
    annots=['chan'+str(i) for i in range(n_features)]
)

# %%
# Project aligned Xenium corodinates to low-res DESI `adata.obsm`
# outdir = '../data/simulation/'
adata_xenium = sc.read_h5ad(os.path.join(outdir, 'xenium_feature_matrix.h5'))
adata_desi = IO.load_desi(
    os.path.join(outdir, 'DESI_img.ome.tif'), 
    raw_img=True, load_img=True, sigma=0., erode_pixel=0
)
adata_xenium, adata_desi = IO.filter_cells(adata_xenium, adata_desi, by='map')

coord_map = {}  # DESI coord -> list of Xenium coords
for hires_coord, lowres_coord in zip(adata_xenium.obsm['spatial'], adata_xenium.obsm['desi_map']):
    hires_coord, lowres_coord = tuple(hires_coord), tuple(lowres_coord)
    coord_map.setdefault(lowres_coord, []).append(hires_coord)

proj_coords = [
    np.array(coord_map[tuple(coord)]).mean(0)
    for coord in adata_desi.obsm['spatial']
]
adata_desi.obsm['xenium_map'] = np.array(proj_coords)

# Save updated low-res DESI feature
adata_desi.write_h5ad(os.path.join(outdir, 'desi_feature_matrix.h5'))
del coord_map, proj_coords


# %%
# -----------
#   Notice
# -----------

# The following section is for LYNX V1 (interpolation + concat encoder)

# Interpolate hi-res gradient assignment, generate hi-res DESI adata
scale = 1/20
xspan, yspan = 3000, 3000

hires_desi_img = rescale(desi_img, 1/scale, preserve_range=True, channel_axis=0)
hires_gradients = rescale(gradients, 1/scale, preserve_range=True)

# %%
# Extract mapped DESI features (attention: DESI image has [Y, X])
adata = sc.read_h5ad(os.path.join(outdir, 'xenium_feature_matrix.h5'))

xcoords = adata.obs['x_centroid'].values.round().astype(np.int32)
ycoords = adata.obs['y_centroid'].values.round().astype(np.int32)
adata_desi_hires = sc.AnnData(hires_desi_img[:, ycoords, xcoords].T)   # image: YX-index

adata_desi_hires.obs['x_centroid'] = xcoords
adata_desi_hires.obs['y_centroid'] = ycoords
adata_desi_hires.obs.index = adata.obs_names
adata_desi_hires.var.index = ['chan'+str(i) for i in range(n_features)]

adata_desi_hires.write_h5ad(os.path.join(outdir, 'desi_hires_feature_matrix.h5'))
del xcoords, ycoords


# %%
# ---------------------------
#   Simulate ground-truth z 
# ---------------------------

# Simulate `z` as mixture of zonation states (weighted by inverse distance) + noise

import torch
import torch.nn as nn
import torch.nn.functional as F

def _convert_gradients(gradients):
    """TMP: convert ground-truth gradients to 0-1"""
    v = gradients + gradients.min()
    return (v-v.min()) / (v.max()-v.min())


def _smooth_1d_gradients(gamma, n_bins):
    step = len(gamma) // n_bins if len(gamma) % n_bins == 0 else \
           len(gamma) // n_bins + 1
    smoothed_gamma = []
    for i in range(0, len(gamma), step):
        start, end = i, i+step
        if start < end < len(gamma):
            smoothed_gamma.append(np.mean(gamma[start:end]))
    return np.array(smoothed_gamma)


def gradient_to_zone(gradients, cutoffs):
    n_zones = len(cutoffs) - 1
    zones = np.zeros_like(gradients, dtype=np.uint8)
    for i in range(len(cutoffs)-1):
        mask = np.logical_and(
            gradients >= cutoffs[i],
            gradients < cutoffs[i+1]
        )
        zones[mask] = i

    zones[gradients < cutoffs[0]] = 0
    zones[gradients >= cutoffs[-1]] = n_zones - 1
    return zones


def simulate_z(
    gradients, coords, n_zones,
    std=.1, r=30, show_plot=False
):

    def _get_patch_coords(y, x, r, height, width):
        y_min, y_max = max(0, y - r), min(height - 1, y + r)
        x_min, x_max = max(0, x - r), min(width - 1, x + r)

        # Generate the grid of y and x coordinates
        yy, xx = np.meshgrid(np.arange(y_min, y_max + 1), np.arange(x_min, x_max + 1), indexing='ij')
        return (yy.ravel(), xx.ravel())    

    h, w = gradients.shape
    gamma = np.sort(gradients.flatten())
    cutoffs = utils.piecewise_linear_fit(
        _smooth_1d_gradients(gamma, 100), 
        k=n_zones, show=show_plot
    )
    zone = gradient_to_zone(gradients, cutoffs)

    # Compute neighbor-smoothed "lookup table" for each pixel
    # z_lookup[i] = [p(zone_0),..., p(zone_k)]
    n_cells = len(coords[0]) # coords dim: [2, N]
    z_lookup = np.zeros((n_cells, n_zones), dtype=np.float32)

    for i, coord in enumerate(coords.T):
        patched_coords = _get_patch_coords(coord[0], coord[1], r, h, w)
        zone_patch = zone[patched_coords]
        z_lookup[i] = [(zone_patch == label).sum() for label in range(n_zones)]
    z_lookup = z_lookup / z_lookup.sum(1, keepdims=True)
    
    z = np.zeros_like(z_lookup)
    for i, zi in enumerate(z_lookup.T):
        z[:, i] = np.random.normal(zi, scale=std)
    z -= z.mean(1, keepdims=True)

    return z, zone

# %%
gamma = np.sort(gradients.flatten())
cutoffs = utils.piecewise_linear_fit(
    _smooth_1d_gradients(gamma, 100), 
    k=n_zones, show=True
)
zone = gradient_to_zone(gradients, cutoffs)

plt.imshow(zone, cmap='turbo')
plt.title('Zones')
plt.show()


# %%
# # Low-res (pixel-level) ground-truth gradients & img
gradients = np.load(os.path.join(outdir, 'gradients.npy'))
gradients = _convert_gradients(gradients)
desi_img = tifffile.imread(os.path.join(outdir, 'DESI_img.ome.tif'))  

# high-res (cell-level)
adata = sc.read_h5ad(os.path.join(outdir, 'xenium_feature_matrix.h5'))
adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].copy().to_numpy()
IO.load_spatial_metadata(adata)
adata_desi = IO.load_desi(os.path.join(outdir, 'desi_hires_feature_matrix.h5'), raw_img=False)  

scale = 1/20
n_cells, n_features = adata.shape

# %%
n_zones = 6
coords = np.vstack((adata.obsm['desi_map'][:, 1], adata.obsm['desi_map'][:, 0])) # YX-index 
z, z_counts = simulate_z(
    gradients, coords=coords, n_zones=n_zones, std=.1, r=, show_plot=True
)
sns.clustermap(np.corrcoef(z.T), cmap='RdBu_r', vmin=-1, vmax=1)
plt.show()

# %%
z_labels = ['z'+str(i) for i in range(n_zones)]
for i, zi in enumerate(z.T):
    adata.obs['z'+str(i)] = zi

sq.pl.spatial_scatter(
    adata, color=z_labels, 
    img=False, size=20, cmap='turbo', ncols=2
)
adata.obs.drop(z_labels, axis=1, inplace=True)
del zi

# %%
# Save latent factors
adata.obsm['X_z'] = z
adata.write_h5ad(os.path.join(outdir, 'xenium_feature_matrix.h5'))


# %%
# ---------------------
#   Validation
# ---------------------

# Compare w/ real DESI
adata_desi_real = IO.load_desi('../data/desi/NIH_F3.ome.tif')
desi_img_real = adata_desi_real.uns['X_img']

nrows = 3
ncols = 5
rand_indices = np.random.choice(adata_desi_real.shape[1], nrows*ncols, replace=False)

fig, axes = plt.subplots(nrows, ncols, figsize=(25, 10))
idx = 0
for r in range(nrows):
    for c in range(ncols):
        axes[r, c].imshow(desi_img_real[rand_indices[idx]], cmap='magma')
        axes[r, c].axis('off')
        idx += 1

plt.tight_layout()
plt.suptitle('Real DESI channels', fontsize=30, y=1.03)
plt.show()
del idx, r, c, nrows, ncols, rand_indices

# %%
desi_corr_real = np.corrcoef(adata_desi_real.X.T)
sns.clustermap(desi_corr_real, cmap='RdBu_r')

sc.pp.pca(adata_desi_real)
sc.pl.pca_variance_ratio(adata_desi_real)

# %%
