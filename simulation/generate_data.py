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
from pyro.infer.autoguide import AutoDelta, AutoNormal
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
rcParams.update({'figure.dpi': 100})
rcParams.update({'savefig.dpi': 300})

from importlib import reload

# %%
%load_ext autoreload
%autoreload 2


# %%
# ----------------------
#  Simulate \t & z 
# ----------------------

# %%
# Annotate & compute heat-diffused spatial gradient to fit (z) & (t)
data_path = '../data/simulation/'
adata_desi = sc.read_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'xenium_feature_matrix.h5'))

# %%
# Annotate PV & CVs from corresponding DESI patch!
# Load corresponding DESI image:
xsize, ysize = adata_desi.obsm['spatial'].max(0)
xsize += 1
ysize += 1

# sc.pp.pca(adata_desi)
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

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(cv_bin, cmap='RdBu_r')
# ax2.imshow(pv_bin, cmap='RdBu_r')
# plt.show()


# %%
# Heat diffusion on `t`
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
# Refit z & t
import pyro.distributions.constraints as constraints

class UtoZRegressor(pyro.nn.PyroModule):
    def __init__(self, in_dim, latent_dim, multiplier=0.1, batch_size=128, orth_lambda=1.0):
        super().__init__()
        self.W_z = PyroParam(torch.randn(in_dim, latent_dim)*multiplier, constraint=constraints.interval(-3, 3))
        self.b_z = PyroParam(torch.zeros(latent_dim), constraint=constraints.interval(-1, 1))
        self.W_t = PyroParam(torch.randn(latent_dim, 1)*multiplier, constraint=constraints.interval(-3, 3))
        self.b_t = PyroParam(torch.zeros(1), constraint=constraints.interval(-1, 1))
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.orth_lambda = orth_lambda

    def forward(self, u, t):
        with pyro.plate("batch_z", size=u.size(0), subsample_size=self.batch_size) as ind:
            z = torch.matmul(u.index_select(0, ind), self.W_z) + self.b_z
            t_mu = torch.matmul(z, self.W_t) + self.b_t
            pyro.sample("t", dist.Normal(t_mu, 0.1).to_event(1), obs=t.index_select(0, ind))
        
        # Orthogonality Regularization
        WT_W = self.W_z.T @ self.W_z  # Compute W^T W
        identity = torch.eye(WT_W.shape[0], device=WT_W.device)  # Identity matrix
        ortho_loss = torch.norm(WT_W - identity, p="fro")**2  # Frobenius norm penalty
        pyro.factor("orth_penalty", -self.orth_lambda * ortho_loss)  # Add penalty to ELBO


        # DEBUG W dim!!!!!
        # with pyro.plate("latent_dims", size=self.in_dim):
        #     tau = pyro.sample("tau", dist.HalfCauchy(1.0))  # Global sparsity
        #     lambdas = pyro.sample("lambda", dist.HalfCauchy(torch.ones(self.in_dim)))  # Local sparsity
        #     W_z = pyro.sample("W_z", dist.Normal(0, tau * lambdas))

        # z = torch.matmul(u, W_z.unsqueeze(1).expand(-1, self.latent_dim)) + self.b_z
        # t_mu = torch.matmul(z, self.W_t) + self.b_t
        # pyro.sample("t", dist.Normal(t_mu, 0.1).to_event(1), obs=t)


    def predict(self, u):
        with torch.no_grad():
            # guide_trace = pyro.poutine.trace(guide).get_trace(torch.zeros((1, self.in_dim)), torch.zeros((1, 1)))
            # W_z = guide_trace.nodes["W_z"]["value"]  # MAP estimate of W_z
            z_fit = torch.matmul(u, self.W_z) + self.b_z
            t_fit = torch.matmul(z_fit, self.W_t) + self.b_t
        
        return z_fit.detach().cpu().numpy(), t_fit.detach().cpu().numpy()
    
# %%
N, M = adata_desi.shape
batch_size = 512
D = 4
t_obs = torch.tensor(gradients[coords]).unsqueeze(-1).float()
u = torch.tensor(adata_desi.X).float()

pyro.clear_param_store()
torch.cuda.empty_cache()

del model
model = UtoZRegressor(in_dim=M, latent_dim=D, batch_size=batch_size, orth_lambda=.1)
optimizer = pyro_optim.Adam({"lr": 1e-3})
guide = AutoNormal(model)
svi = infer.SVI(model, guide, optimizer, loss=infer.Trace_ELBO())

n_epochs = 500
losses = []
pbar = tqdm(range(n_epochs))

for epoch in pbar:
    tot_loss = svi.step(u, t_obs)
    loss = tot_loss / batch_size
    pbar.set_description(
        f"Epoch {epoch}, Loss: {loss:.2f}"
    )
    losses.append(loss)


# %%
plt.figure(figsize=(5, 2))
plt.plot(np.arange(n_epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Two-layer regression \n(u -> z) simulation')
plt.show()

# %%
z_fit, t_fit = model.predict(u)
t_fit = (t_fit-t_fit.min()) / (t_fit.max()-t_fit.min())
adata_desi.obsm['z_refit'] = z_fit
adata_desi.obsm['t_refit'] = t_fit

# %%
# Visualization
fig, ax = plt.subplots()
sns.heatmap(np.corrcoef(z_fit.T), cmap='RdBu_r', ax=ax)
ax.set_title('Corr(z) - simulation')
plt.show()

z_labels = ['z'+str(i) for i in range(D)]
adata_desi.obs[z_labels] = z_fit
sq.pl.spatial_scatter(
    adata_desi, color=z_labels, size=1, img=False, ncols=3, cmap='turbo'
)
adata_desi.obs.drop(z_labels, axis=1, inplace=True)

adata_desi.obs['Gradient (graph diffusion)'] = t_obs
adata_desi.obs['Gradient (refit)'] = t_fit
sq.pl.spatial_scatter(
    adata_desi, color=['Gradient (graph diffusion)', 'Gradient (refit)'], size=1.5, img=False, cmap='RdBu_r'
)
adata_desi.obs.drop('Gradient (graph diffusion)', axis=1, inplace=True)
adata_desi.obs.drop('Gradient (refit)', axis=1, inplace=True)

# %%
# Save refits as ground-truth z & gradients (t)
adata_desi.obs['t_gt'] = t_fit
adata_desi.write_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))


# %%
# -----------------------------------------
#  Simulate Xenium (x)  w/ NB regression
# -----------------------------------------

# %%
data_path = '../data/simulation/'
adata_desi = sc.read_h5ad(os.path.join(data_path, 'desi_feature_matrix.h5'))
adata_xenium = sc.read_h5ad(os.path.join(data_path, 'xenium_feature_matrix.h5'))

# %%
# Project low-res (query) ground-truth latent & gradients -> hi-res (ref)
def get_hires_factors(z, t, query_coords, query_map):
    coord_to_idx = {
        tuple(coord): idx 
        for (idx, coord) in enumerate(query_coords)
    }

    z_ref = np.zeros((len(query_map), z.shape[-1]))
    t_ref = np.zeros(len(query_map))

    for i, query_coord in enumerate(query_map):
        j = coord_to_idx[tuple(query_coord)]
        z_ref[i] = z[j]
        t_ref[i] = t[j]

    return z_ref, t_ref

z_query, t_query = adata_desi.obsm['z_refit'], adata_desi.obsm['t_refit']
z_ref, t_ref = get_hires_factors(
    z_query, t_query, adata_desi.obsm['spatial'], adata_xenium.obsm['desi_map']
)

# %%
class ZtoXRegressor(pyro.nn.PyroModule):
    def __init__(self, in_dim, latent_dim, celltype_dim, batch_size=128):
        super().__init__()
        self.eps = 1e-8
        self.batch_size = batch_size

        self.theta = PyroParam(torch.ones(in_dim), constraint=dist.constraints.positive)  # Dispersion (θ > 0)
        self.beta = PyroParam(torch.zeros(in_dim))  # Gene-specific intercept
        self.W_g = PyroParam(torch.randn(latent_dim, in_dim))  # Gene-factor loadings
        self.W_c = PyroParam(torch.randn(celltype_dim, in_dim))  # Cell-type specific loadings

        self.epsilon_std = PyroParam(torch.tensor(0.1), constraint=dist.constraints.positive)

        # Library-size correction terms
        self.l_mean = PyroParam(torch.tensor(0.)) 
        self.l_std = PyroParam(torch.tensor(1.), constraint=dist.constraints.positive)

    def forward(self, z, c, x):
        # Sample per-cell scaling factor & noise
        with pyro.plate("lib_scale", z.size(0)):
            lib_dist = dist.Normal(self.l_mean, self.l_std)
            l = pyro.sample("l", lib_dist)
            epsilon_dist = dist.Normal(torch.tensor(0.), self.epsilon_std)
            epsilon = pyro.sample("epsilon", epsilon_dist)

        # Compute NB mean & logits
        with pyro.plate("cells", size=z.size(0), subsample_size=self.batch_size) as ind:
            log_mu = torch.matmul(z.index_select(0, ind), self.W_g) + torch.matmul(c.index_select(0, ind), self.W_c) + \
                     l.index_select(0, ind).unsqueeze(1) + self.beta.unsqueeze(0)
            mu = log_mu.exp() + torch.relu(epsilon.index_select(0, ind).unsqueeze(1))

            logits = torch.log(mu+self.eps) - torch.log(self.theta.unsqueeze(0))
            nb_dist = dist.NegativeBinomial(total_count=self.theta, logits=logits)
            pyro.sample("x", nb_dist.to_event(1), obs=x.index_select(0, ind))

    def predict(self, z, c, x):
        """Generate simulated counts `x_tilde` with learned parameters"""
        with torch.no_grad():
            l = dist.Normal(self.l_mean, self.l_std).sample((z.size(0),))
            l = self._reorder_rank(x.sum(1), l)  # Reorder sampled libsize based on observation

            epsilon = dist.Normal(torch.tensor(0.), self.epsilon_std).sample((z.size(0),))
            log_mu = self.beta.unsqueeze(0) + l.unsqueeze(1) + torch.matmul(z, self.W_g) + torch.matmul(c, self.W_c)
            mu = log_mu.exp() + torch.relu(epsilon.unsqueeze(1))
            x_tilde = torch.round(mu)

            # Reorder sampled expressions based on observation
            # logits = torch.log(mu+self.eps) - torch.log(self.theta.unsqueeze(0))
            # x_tilde = dist.NegativeBinomial(total_count=self.theta, logits=logits).sample()
            # for g in range(x.size(1)):
            #     x_tilde[:, g] = self._reorder_rank(x[:, g], x_tilde[:, g])

        return x_tilde.detach().cpu().numpy().astype(np.int32)
    
    @staticmethod
    def _reorder_rank(a, b):
        """Reorder b follwoing the rank positions of a"""
        return b[np.argsort(np.argsort(a))]


# %%
z_factor = torch.softmax(torch.tensor(z_ref), dim=-1).float()  # Assume additive factors per cell
x_obs = torch.tensor(adata_xenium.X.A).float()
celltype_categories = adata_xenium.obs['cell_type'].cat.categories.tolist()
c = pd.get_dummies(adata_xenium.obs['cell_type'], dtype=int).values
c = torch.tensor(c).float()

N, G, D, C = x_obs.shape[0], x_obs.shape[1], z_factor.shape[1], c.shape[1]

# %%
# Model and inference
torch.cuda.empty_cache()
pyro.clear_param_store()
# del model

model = ZtoXRegressor(in_dim=G, latent_dim=D, celltype_dim=C, batch_size=512)
optimizer = pyro_optim.Adam({"lr": 1e-3})
guide = AutoNormal(model)
svi = infer.SVI(model, guide, optimizer, loss=infer.Trace_ELBO())

n_epochs = 10000
losses = []
pbar = tqdm(range(n_epochs))

for epoch in pbar:
    tot_loss = svi.step(z_factor, c, x_obs)
    loss = tot_loss / N
    pbar.set_description(
        f"Epoch {epoch}, Loss: {loss:.2f}"
    )
    losses.append(loss)

# %%
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(np.arange(n_epochs), losses)
ax.set_xlabel('Epochs')
ax.set_ylabel('-ELBO')

ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.title('Negative Binomial regression \n(z -> x) simulation')
plt.show()


# %%
x_tilde = model.predict(z_factor, c, x_obs)
adata_xenium_refit = adata_xenium.copy()
adata_xenium_refit.X = x_tilde
print('MSE:', ((adata_xenium.X.A.flatten() - x_tilde.flatten())**2).mean()) 


# %%
rand_features = np.random.choice(adata_xenium.var_names, 8, replace=False)
sq.pl.spatial_scatter(
    adata_xenium, color=rand_features, img=False, size=20, cmap='Reds', ncols=4
)

sq.pl.spatial_scatter(
    adata_xenium_refit, color=rand_features, img=False, size=20, cmap='Reds', ncols=4
)
del rand_features

# %%
marker = 'CYP3A4'
sq.pl.spatial_scatter(
    adata_xenium, color=marker, img=False, size=20, cmap='Reds', ncols=4,
)
sq.pl.spatial_scatter(
    adata_xenium_refit, color=marker, img=False, size=20, cmap='Reds', ncols=4, title='{} (fitted)'.format(marker)
)
del marker

# %%
marker = 'CYP1A1'
sq.pl.spatial_scatter(
    adata_xenium, color=marker, img=False, size=20, cmap='Reds', ncols=4,
)
sq.pl.spatial_scatter(
    adata_xenium_refit, color=marker, img=False, size=20, cmap='Reds', ncols=4, title='{} (fitted)'.format(marker)
)
del marker

# %%
plt.figure(figsize=(5, 3))
plt.hist(adata_xenium.X.A.sum(1), bins=50, edgecolor='white', alpha=.5, label='Observation')
plt.hist(x_tilde.sum(1), bins=50, edgecolor='white', alpha=.5, label='Simulation')
plt.legend()
plt.title('log(Library)')
plt.show()

gc.collect()


# %%

# %%
# Save refitted Xenium
adata_xenium_refit.X = adata_xenium_refit.X.astype(np.uint32)
adata_xenium_refit.obsm['z_gt'] = z_ref
adata_xenium_refit.obs['t_gt'] = t_ref
adata_xenium_refit.write_h5ad(os.path.join(data_path, 'xenium_refit_feature_matrix.h5'))

# %% 
# Validation
adata_xenium_norm = adata_xenium.copy()
sc.pp.normalize_total(adata_xenium_norm)
sc.pp.log1p(adata_xenium_norm)

sc.pp.pca(adata_xenium_norm)
sc.pl.pca_variance_ratio(adata_xenium_norm)

# adata_xenium_refit = sc.read_h5ad('../data/simulation/xenium_refit_feature_matrix.h5')

adata_xenium_refit_norm = adata_xenium_refit.copy()
sc.pp.normalize_total(adata_xenium_refit_norm)
sc.pp.log1p(adata_xenium_refit_norm)

sc.pp.pca(adata_xenium_refit_norm)
sc.pl.pca_variance_ratio(adata_xenium_refit_norm)


# %%
adata_xenium_norm.obs['pc1'] = adata_xenium_norm.obsm['X_pca'][:, 0]
adata_xenium_norm.obs['pc2'] = adata_xenium_norm.obsm['X_pca'][:, 1]
adata_xenium_norm.obs['pc3'] = adata_xenium_norm.obsm['X_pca'][:, 2]

sq.pl.spatial_scatter(
    adata_xenium_norm, color=['pc1', 'pc2', 'pc3'], cmap='Reds', img=False, size=20
)

# %%
adata_xenium_refit_norm.obs['pc1'] = adata_xenium_refit_norm.obsm['X_pca'][:, 0]
adata_xenium_refit_norm.obs['pc2'] = adata_xenium_refit_norm.obsm['X_pca'][:, 1]
adata_xenium_refit_norm.obs['pc3'] = adata_xenium_refit_norm.obsm['X_pca'][:, 2]

sq.pl.spatial_scatter(
    adata_xenium_refit_norm, color=['pc1', 'pc2', 'pc3'], cmap='Reds', img=False, size=20
)



# %%

