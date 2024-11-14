# %%

# Synthetic simulation workflow
# (1). Real Xenium ROI patch, annotate PVs & CVs
# (2). Generate ground-truth "gradients" from heat-based diffusion
# (3). Generate synthetic DESI with GMMs

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
reload(IO)
reload(zonation)

# %%
def load_xenium_img(dir, sample_id):
    adata = IO.load_xenium(os.path.join(dir, sample_id))

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

roi_img = np.zeros((antibody_img.shape[0]+1, yend-ystart, xend-xstart))
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

# %%

# Simulate ground-truth gradients on "zoomed-in" low-res image

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
scale = 1/10
vein_mask = tifffile.imread(os.path.join(outdir, 'vein_annotations.tif'))

lowres_vein_mask = simulate_lowres_zonation(
    vein_mask, scale=scale, dilate_pixel=20
)

plt.figure()
plt.imshow(lowres_vein_mask, cmap='RdBu_r')
plt.show()

# %%
# Graph-based diffusion simulation
gradient_model = zonation.HeatDiffusion(vein_prior=lowres_vein_mask, ndim=2)
_, _ = gradient_model.get_interior_U()
gradients = gradient_model.infer_zone_dynamics()
lobule_layers = gradient_model.infer_zones(n_layers=8)


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

plt.show()


# %%
np.save(os.path.join(outdir, 'gradients.npy'), gradients)
np.save(os.path.join(outdir, 'zonation.npy'), lobule_layers)


# %%
# Save Xenium expressions mapped to the ROI
scale = 0.1

condition = np.logical_and(
    (xstart <= coords['x_centroid']) & (xend >= coords['x_centroid']),
    (ystart <= coords['y_centroid']) & (yend >= coords['y_centroid']),
)  

adata_roi = adata[condition]
adata_roi.obs.loc[:, 'x_centroid'] = adata_roi.obs['x_centroid'] - xstart
adata_roi.obs.loc[:, 'y_centroid'] = adata_roi.obs['y_centroid'] - ystart

# Store mappable hi-res -> low-res coords
lowres_coords = (adata_roi.obs[['x_centroid', 'y_centroid']]*scale).values.astype(np.int32)
adata_roi.obsm['desi_map'] = lowres_coords
adata_roi.write_h5ad(os.path.join(outdir, 'xenium_feature_matrix.h5'))

del condition

# %%
# Simulate correlated high-res DESI features from
# ground-truth zonation gradients

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import bernoulli

# %%

# (1). random NN or CNN?

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
    

def generate_desi_img(gradients, n_features, dropout_rate=0.5, eps=.01):
    h, w = gradients.shape
    x_in = torch.tensor(gradients).float()
    x_in = x_in.unsqueeze(0).unsqueeze(0)  # dim: [N, C, H, W]

    generator = FeatureGenerator(n_features)
    features = generator(x_in).squeeze().detach().cpu().numpy()
    features = utils.norm_by_channel(features)
    features[np.isnan(features)] = 0

    # Random noise & dropout
    for i, feature in enumerate(features):
        dropout = bernoulli.rvs(p=1-dropout_rate, size=h*w)
        dropout = dropout.reshape(h, -1)
        epsilon = np.random.randn(h, w) * eps
        
        feature = feature*dropout + epsilon
        feature[feature < 0] = 0
        features[i] = (feature-feature.min()) / (feature.max()-feature.min())

    return features

# %%

gradients = np.load(os.path.join(outdir, 'gradients.npy'))

n_features = 100
desi_img = generate_desi_img(
    gradients, n_features=n_features, 
    dropout_rate=.1, eps=.1
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
