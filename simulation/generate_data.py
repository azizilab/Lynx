# %%
# Synthetic data generation workflow:

# (1). Annotate GT end-points from antibody image
# (2). Generate ground-truth gradients from graph heat diffusion
# (3). Generate spatial factor probabilistic assignment (z)
# (4). Fit observation x with mixing function via NB regression from z



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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as torch_optim
import pyro
import pyro.distributions as dist
import pyro.optim as pyro_optim
import pyro.infer as infer
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer.autoguide import AutoNormal
from tqdm import tqdm

import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
from skimage.transform import rescale

sys.path.append('..')
from util import IO, utils, plot, zonation

# %%
from matplotlib import rcParams
from IPython.display import display
sns.set_context('paper')
rcParams.update({'font.family': 'Liberation Sans'})
rcParams.update({'font.size': 12})
rcParams.update({'figure.dpi': 180})
rcParams.update({'savefig.dpi': 300})

from importlib import reload

# %%
%load_ext autoreload
%autoreload 2


# %%
# ----------------------
#  Simulate \gamma & z 
# ----------------------

# %%
# Annotate & compute heat-diffused spatial gradient to fit (z) & (\gamma)

data_path = '../data/simulation/'
adata_desi = sc.read_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'xenium_feature_matrix.h5'))

# %%
# Annotate PV & CVs from corresponding DESI patch!
# Load corresponding DESI image:
xsize, ysize = adata_desi.obsm['spatial'].max(0)
xsize += 1
ysize += 1

sc.pp.pca(adata_desi)
# pc_labels = ['PC'+str(i+1) for i in range(10)]
# for i, pc in enumerate(adata_desi.obsm['X_pca'][:, :10].T):
#     adata_desi.obs['PC'+str(i+1)] = pc
# sq.pl.spatial_scatter(adata_desi, color=pc_labels, img=False, size=1, cmap='turbo')

# %%
# Annotate based approximate PCA representations 
coords = tuple(np.vstack((adata_desi.obsm['spatial'][:, 1], adata_desi.obsm['spatial'][:, 0])))
pc_cv = -adata_desi.obsm['X_pca'][:, 0]
pc_pv = adata_desi.obsm['X_pca'][:, 2]

cv_img = np.zeros((ysize, xsize))
pv_img = np.zeros((ysize, xsize))

cv_img[coords] = pc_cv
pv_img[coords] = pc_pv

# %%
pv_thld = np.percentile(pv_img.flatten(), 90)
pv_bin = (pv_img >= pv_thld).astype(np.uint8)
cv_thld = np.percentile(cv_img.flatten(), 90)
cv_bin = np.logical_and(cv_img >= cv_thld, pv_bin == 0).astype(np.uint8)

cv_bin = morphology.erosion(
    utils.remove_holes(cv_bin, min_area=3),
    footprint=morphology.disk(radius=1)
)

pv_bin = morphology.erosion(
    utils.remove_holes(pv_bin, min_area=3),
    footprint=morphology.disk(radius=1)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv_bin, cmap='RdBu_r')
ax2.imshow(pv_bin, cmap='RdBu_r')
plt.show()


# %%
# Heat diffusion on `gamma`
annot_mask = np.zeros((ysize, xsize), dtype=np.int8)
annot_mask[pv_bin == 1] = -1
annot_mask[cv_bin == 1] = 1

diff_model = zonation.HeatDiffusion(annot_mask, ndim=2)
_, _ = diff_model.get_interior_U()
gradients = diff_model.infer_zone_dynamics()
plt.figure()
plt.imshow(gradients, cmap='RdBu_r')
plt.colorbar()
plt.show()


# %%

# Refit z & \gamma (TODO: fit w/ deep learning instead)
class UtoZRegressor(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, ortho_weight=.1):
        """
        A 2-layer neural network for fitting z (D-dimension) and γ (1-dimension)
        """
        super().__init__()
        self.ortho_weight = ortho_weight

        # Define layers
        self.z_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.gamma_layer = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, u):
        z = self.z_layer(u)
        gamma = self.gamma_layer(z)
        return z, gamma

    def predict(self, u):
        self.eval()
        with torch.no_grad():
            return self.forward(u)

    def orthogonality_loss(self, z):
        """
        Compute the orthogonality loss using the Gram matrix.
        Encourages z[:, d] to be uncorrelated with each other.
        """
        G = torch.matmul(z.T, z)  # Compute Gram matrix (D x D)
        I = torch.eye(G.shape[0], device=G.device)  # Identity matrix
        return torch.norm(G - I, p="fro")  # Frobenius norm of (G - I)

    def fit(self, u_train, gamma_train, epochs=100, lr=1e-2, batch_size=512):
        self.train() 
        optimizer = torch_optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(u_train, gamma_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            epoch_loss = 0.0
            for u_batch, gamma_batch in dataloader:
                optimizer.zero_grad()
                z_pred, gamma_pred = self.forward(u_batch)
                
                
                loss_gamma = loss_fn(gamma_pred, gamma_batch)
                loss_z = self.orthogonality_loss(z_pred)
                loss = loss_gamma + self.ortho_weight*loss_z
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss)

            pbar.set_description(
                f"Epoch {epoch}, Loss: {epoch_loss:.2f}"
            )

        return losses

# %%
N, M = adata_desi.shape
D = 6
gamma_obs = torch.tensor(gradients[coords]).unsqueeze(-1).float()
u = torch.tensor(adata_desi.X).float()

# %%
torch.cuda.empty_cache()
del model

model = UtoZRegressor(input_dim=M, latent_dim=D, ortho_weight=.5)
losses = model.fit(u, gamma_obs, epochs=1000, lr=1e-3)


# %%
plt.figure(figsize=(5, 2))
plt.plot(np.arange(1000), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Two-layer regression \nfor (u -> z) simulation')
plt.show()


# %%
z_fit, gamma_fit = model.predict(u)
z_fit = z_fit.detach().cpu().numpy()
gamma_fit = gamma_fit.squeeze().detach().cpu().numpy()

adata_desi.obsm['z_refit'] = z_fit
adata_desi.obsm['t_refit'] = gamma_fit

# %%
# Visualization
z_labels = ['z'+str(i) for i in range(D)]
adata_desi.obs[z_labels] = z_fit
sq.pl.spatial_scatter(
    adata_desi, color=z_labels, size=1, img=False, ncols=3, cmap='turbo'
)
adata_desi.obs.drop(z_labels, axis=1, inplace=True)

adata_desi.obs['t_refit'] = gamma_fit
sq.pl.spatial_scatter(
    adata_desi, color=['t_true', 't_refit'], size=1.5, img=False, cmap='RdBu_r'
)
adata_desi.obs.drop('t_refit', axis=1, inplace=True)

# %%
# Save ground-truth z & gradients (refits)
adata_desi.write_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))


# %%
# ----------------------------------
#  Simulate x (NB regression)
# ----------------------------------

# from sklearn.mixture import GaussianMixture

# def simulate_z(
#     gradients, coords, n_zones, 
#     std=.1, r=30
# ):

#     def _get_patch_coords(y, x, r, height, width):
#         y_min, y_max = max(0, y - r), min(height - 1, y + r)
#         x_min, x_max = max(0, x - r), min(width - 1, x + r)

#         # Generate the grid of y and x coordinates
#         yy, xx = np.meshgrid(np.arange(y_min, y_max + 1), np.arange(x_min, x_max + 1), indexing='ij')
#         return (yy.ravel(), xx.ravel())    

#     h, w = gradients.shape
#     gmm = GaussianMixture(n_components=n_zones, random_state=42)
#     zone = gmm.fit_predict(gradients.flatten()[:, None])
#     zone = zone.reshape(gradients.shape)

#     # Compute neighbor-smoothed "lookup table" for each pixel
#     # z_lookup[i] = [p(zone_0),..., p(zone_k)]
#     n_cells = len(coords[0]) # coords dim: [2, N]
#     z_lookup = np.zeros((n_cells, n_zones), dtype=np.float32)

#     for i, coord in enumerate(coords.T):
#         patched_coords = _get_patch_coords(coord[0], coord[1], r, h, w)
#         zone_patch = zone[patched_coords]
#         z_lookup[i] = [(zone_patch == label).sum() for label in range(n_zones)]

#     z_lookup = z_lookup / z_lookup.sum(1, keepdims=True)
    
#     z = np.zeros_like(z_lookup)
#     for i, zi in enumerate(z_lookup.T):
#         z[:, i] = np.random.normal(zi, scale=std)
#     z -= z.mean(1, keepdims=True)

#     return zone, z

# %%
data_path = '../data/simulation/'
adata_desi = sc.read_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'xenium_feature_matrix.h5'))

# %%
# Project low-res (query) ground-truth latent & gradients -> hi-res (ref)
# TODO: add cell-type specific factor differences!
def get_hires_factors(z, gamma, query_coords, query_map):
    coord_to_idx = {
        tuple(coord): idx 
        for (idx, coord) in enumerate(query_coords)
    }

    z_ref = np.zeros((len(query_map), z.shape[-1]))
    gamma_ref = np.zeros(len(query_map))

    for i, query_coord in enumerate(query_map):
        j = coord_to_idx[tuple(query_coord)]
        z_ref[i] = z[j]
        gamma_ref[i] = gamma[j]

    return z_ref, gamma_ref

# %%
z_query, gamma_query = adata_desi.obsm['z_refit'], adata_desi.obsm['t_refit']
z_ref, gamma_ref = get_hires_factors(
    z_query, gamma_query, adata_desi.obsm['spatial'], adata_xenium.obsm['desi_map']
)

# %%
class ZtoXRegressor(pyro.nn.PyroModule):
    def __init__(self, in_channels, latent_channels):
        super().__init__()
        self.eps = 1e-7

        self.theta = PyroParam(torch.ones(in_channels), constraint=dist.constraints.positive)  # Dispersion (θ > 0)
        self.beta = PyroParam(torch.zeros(in_channels))  # Gene-specific intercept
        self.W_g = PyroParam(torch.randn(in_channels, latent_channels))  # Gene-factor loadings

        # Library-size correction terms
        self.l_mean = PyroParam(torch.tensor(0.)) 
        self.l_std = PyroParam(torch.tensor(1.), constraint=dist.constraints.positive)

    def forward(self, z, x):
        # Sample per-cell scaling factor
        with pyro.plate("lib_scale", z.size(0)):
            lib_dist = dist.Normal(self.l_mean, self.l_std)
            l = pyro.sample("l", lib_dist)

        # Compute NB mean & logits
        log_mu = self.beta.unsqueeze(0) + torch.matmul(z, self.W_g.T) + l.unsqueeze(1)
        mu = log_mu.exp()  # Convert to mean rates (μ)
        logits = torch.log(mu+self.eps) - torch.log(self.theta.unsqueeze(0))

        with pyro.plate("cells", z.size(0)):
            nb_dist = dist.NegativeBinomial(total_count=self.theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x)

    def predict(self, z, x=None, noise_ratio=.1):
        """Generate simulated counts `x_tilde` with learned parameters"""
        with torch.no_grad():
            l = dist.Normal(self.l_mean, self.l_std).sample(sample_shape=(z.size(0),))
            log_mu = self.beta.unsqueeze(0) + torch.matmul(z, self.W_g.T) + l.unsqueeze(1)
            mu = log_mu.exp() + torch.relu(torch.randn(log_mu.size())) * noise_ratio

            if x is None:
                x_tilde = mu
            else:
                x_tilde = mu.clone()
                for g in range(x.size(1)):
                    rank_idx = torch.argsort(x[:, g])
                    x_tilde[:, g] = torch.sort(mu[:, g])[0][rank_idx]

        return x_tilde.detach().cpu().numpy()
    
# TODO: assign ranked expressions to original locations
# %%
z_factor = torch.softmax(torch.tensor(z_ref), dim=-1).float()  # Assume additive factors per cell
x_obs = torch.tensor(adata_xenium.X.A).float()
N, G, D = x_obs.shape[0], x_obs.shape[1], z_factor.shape[1]

# %%
# Model and inference
torch.cuda.empty_cache()
pyro.clear_param_store()
del model

model = ZtoXRegressor(in_channels=G, latent_channels=D)
optimizer = pyro_optim.Adam({"lr": 1e-2})
guide = AutoNormal(model)
svi = infer.SVI(model, guide, optimizer, loss=infer.Trace_ELBO())

# %%
num_epochs = 500
losses = []
pbar = tqdm(range(num_epochs))
for epoch in pbar:
    tot_loss = svi.step(z_factor, x_obs)
    loss = tot_loss / N
    pbar.set_description(
        f"Epoch {epoch}, Loss: {loss:.2f}"
    )
    losses.append(loss)

# %%    qdAS
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(np.arange(num_epochs), losses)
ax.set_xlabel('Epochs')
ax.set_ylabel('-ELBO')

ax.legend()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.title('Negative Binomial regression \nfor (z -> x) simulation')
plt.show()


# %%
# x_tilde = model.predict(z, x_obs)
x_tilde = model.predict(z_factor, noise_ratio=1e-3)
adata_xenium_refit = adata_xenium.copy()
adata_xenium_refit.X = x_tilde
print('MSE:', ((adata_xenium.X.A.flatten() - x_tilde.flatten())**2).mean()) 


# %%
# Log-transform both observation & fitted 
sc.pp.normalize_total(adata_xenium)
sc.pp.log1p(adata_xenium)

sc.pp.normalize_total(adata_xenium_refit)
sc.pp.log1p(adata_xenium_refit)

# %%
rand_features = np.random.choice(adata_xenium.var_names, 20, replace=False)
sq.pl.spatial_scatter(
    adata_xenium, color=rand_features, img=False, size=20, cmap='Reds', ncols=4
)

sq.pl.spatial_scatter(
    adata_xenium_refit, color=rand_features, img=False, size=20, cmap='Reds', ncols=4
)
del rand_features

# %%
sq.pl.spatial_scatter(
    adata_xenium, color='DPT', img=False, size=20, cmap='Reds', ncols=4,
)
sq.pl.spatial_scatter(
    adata_xenium_refit, color='DPT', img=False, size=20, cmap='Reds', ncols=4, title='DPT (fitted)'
)

# %%
plt.figure(figsize=(5, 3))
plt.hist(adata_xenium.X.A.sum(1), bins=50, edgecolor='white', alpha=.5, label='Observation')
plt.hist(x_tilde.sum(1), bins=50, edgecolor='white', alpha=.5, label='Simulation')
plt.legend()
plt.title('log(Library)')
plt.show()


# %%
plt.figure(figsize=(5, 3))
plt.hist(adata_xenium.X.A.flatten(), bins=50, edgecolor='white', alpha=.5, label='Observation')
plt.hist(adata_xenium_refit.X.flatten(), bins=50, edgecolor='white', alpha=.5, label='Simulation')
plt.legend()
plt.title('Log-normalized')
plt.xlabel('log(x)')
plt.show()

gc.collect()

# %%
sc.pp.pca(adata_xenium)
sc.pl.pca_variance_ratio(adata_xenium)

sc.pp.pca(adata_xenium_refit)
sc.pl.pca_variance_ratio(adata_xenium_refit)

# %%
# Save refitted Xenium
adata_xenium_refit.write_h5ad(os.path.join(data_path, 'xenium_refit_feature_matrix.h5'))
