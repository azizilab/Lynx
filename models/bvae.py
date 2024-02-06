import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel

from torch.distributions import Normal, MultivariateNormal, kl_divergence
from ml_collections import ConfigDict


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, p=0.1):
        super(ResidualBlock, self).__init__()
        n_groups = 4 if c_in % 4 == 0 and c_out % 4 == 0 else 1
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(n_groups, c_out),
            nn.SiLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(n_groups, c_out)
        )

        self.drop_layer = nn.Dropout2d(p)
        self.skip_layer = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_resid = self.layer1(x)
        x_resid = self.layer2(x_resid)
        x_skip = self.skip_layer(x)
        x = x_resid + x_skip
        out = self.drop_layer(x)
        return out
    

class Encoder(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super(Encoder, self).__init__()
        c_in = configs.c_in
        c_base = configs.c_base
        p = configs.drop_rate
        c_hiddens = [c_base * n 
                     for n in configs.layer_mults]
        
        self.layers = nn.ModuleList([
            ResidualBlock(c_in, c_hiddens[0], p=p)
        ])
        for i in range(len(c_hiddens)-1):
            self.layers.append(
                ResidualBlock(c_hiddens[i], c_hiddens[i+1], p=p)
            )

        self.downscale_layer = nn.MaxPool2d(2)
        
    def forward(self, x):
        for layer in self.layers:
            x_out = layer(x)
            x = self.downscale_layer(x_out)
        return x
        
        
class Decoder(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super(Decoder, self).__init__()
        c_base = configs.c_base
        p = configs.drop_rate
        c_hiddens = [c_base * n
                     for n in configs.layer_mults[::-1]]
        c_hiddens.insert(0, c_hiddens[0])

        self.layers = nn.ModuleList([
            ResidualBlock(c_hiddens[i], c_hiddens[i+1], p=p)
            for i in range(len(c_hiddens)-1)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x_out = F.interpolate(x, scale_factor=(2, 2), mode='bilinear') 
            x = layer(x_out)
        return x


class CovarianceGP(gpytorch.models.ExactGP):
    """
    GP prior for full covariance matrix
    """
    def __init__(self, train_x, train_y, kernel):
        super(CovarianceGP, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.cov_module = kernel

    def forward(self, x):
        x_mean = self.mean_module(x)
        x_cov = self.cov_module(x)
        return gpytorch.distributions.MultivariateNormal(x_mean, x_cov)
    

class BetaVAE(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super(BetaVAE, self).__init__()
        nlayers = len(configs.layer_mults)
        device = configs.device
        
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.beta = configs.beta  # weights for B-VAE
        self.pz_std = torch.tensor(configs.pz_std).to(device)
        self.drop_rate = configs.drop_rate

        self.c_out = configs.c_out
        self.c_bn = configs.c_base * configs.layer_mults[-1]

        self.ny_in, self.nx_in = configs.ydim, configs.xdim
        self.ny_bn = self.ny_in // (2**nlayers)
        self.nx_bn = self.nx_in // (2**nlayers)
    
        # Encoder
        self.in_layer = nn.Linear(
            configs.c_in * configs.ydim * configs.xdim,
            configs.c_base * configs.layer_mults[0]
        )

        enc_modules = [
            self._hidden_layer(
                configs.c_base * configs.layer_mults[i], 
                configs.c_base * configs.layer_mults[i+1])
            for i in range(len(configs.layer_mults)-1)
        ]
        self.encoder = nn.Sequential(*enc_modules)

        self.enc_z_mu = nn.Sequential(
            nn.Linear(self.c_bn, configs.latent_dim),
            nn.Tanh()
        )

        self.enc_z_logvar = nn.Sequential(
            nn.Linear(self.c_bn, configs.latent_dim),
            nn.Softplus()
        )

        # Decoder
        self.dec_z_to_hidden = nn.Linear(configs.latent_dim, self.c_bn)

        dec_modules = [
            self._hidden_layer(
                configs.c_base * configs.layer_mults[i],
                configs.c_base * configs.layer_mults[i-1]
            )
            for i in range(len(configs.layer_mults)-1, 0, -1)
        ]
        self.decoder = nn.Sequential(*dec_modules)

        self.out_layer = nn.Linear(
            configs.c_base * configs.layer_mults[0],
            configs.c_out * configs.ydim * configs.xdim
        )

    def _hidden_layer(self, c_in, c_out):
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate)
        )
    
    def inference(self, x):
        hidden = self.in_layer(x.view(self.batch_size, -1))
        hidden = self.encoder(hidden)

        qz_mu = self.enc_z_mu(hidden)
        qz_logvar = self.enc_z_logvar(hidden)
        qz = Normal(qz_mu, torch.exp(0.5*qz_logvar)).rsample()

        inference_terms = ConfigDict()
        inference_terms.qz = qz
        inference_terms.qz_mu = qz_mu
        inference_terms.qz_logvar = qz_logvar

        return inference_terms
    
    def generative(self, qz):
        hidden = self.dec_z_to_hidden(qz)
        hidden = self.decoder(hidden)
        x_hat = self.out_layer(hidden).view(self.batch_size, self.c_out, self.ny_in, self.nx_in)
        return x_hat
    
    def get_loss(self, x, x_hat, inference_terms):
        # Reconstruction loss
        mse = nn.MSELoss(reduction='none')
        loss_NLL = mse(x, x_hat).sum((1,2,3)).mean()

        # KL( q(z|x) || p(z) )
        qz_mu = inference_terms.qz_mu
        qz_logvar = inference_terms.qz_logvar
        
        pz_mu = torch.zeros_like(qz_mu).to(self.device)
        pz_std = torch.ones_like(pz_mu) * self.pz_std
        
        loss_KL = kl_divergence(
            Normal(qz_mu, torch.exp(0.5*qz_logvar)),
            Normal(pz_mu, pz_std)
        ).sum(-1).mean()

        loss_configs = ConfigDict()
        loss_configs.tot = loss_NLL + self.beta*loss_KL
        loss_configs.nll = loss_NLL
        loss_configs.kl = loss_KL

        return loss_configs         

 
class BetaVAE2D(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super(BetaVAE2D, self).__init__()
        c_out = configs.c_out
        nlayers = len(configs.layer_mults)
        device = configs.device
        
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.beta = configs.beta  # weights for B-VAE
        self.pz_std = torch.tensor(configs.pz_std).to(device)

        self.c_bn = configs.c_base * configs.layer_mults[-1]

        ny_in, nx_in = configs.ydim, configs.xdim
        self.ny_bn = ny_in // (2**nlayers)
        self.nx_bn = nx_in // (2**nlayers)

        # Full Covariance prior terms
        latent_xx, latent_yy = torch.meshgrid(torch.linspace(-1, 1, self.nx_bn), torch.linspace(-1, 1, self.ny_bn))
        self.latent_coords = torch.stack([latent_xx.flatten(), latent_yy.flatten()]).T

        l_prior = torch.tensor([configs.lengthscale], requires_grad=True)
        self.gp_kernel = MaternKernel()
        self.gp_kernel.lengthscale = l_prior

        # Encoder
        # Option 1: FC for q(z)
        # flattened dim. before sampling z_mu & z_var
        # hidden_dim = self.c_bn*self.ny_bn*self.nx_bn 

        # self.encoder = Encoder(configs)
        # self.enc_z_mu = nn.Sequential(
        #     nn.Linear(hidden_dim, configs.latent_dim),
        #     nn.Tanh()
        # )

        # self.enc_z_logvar = nn.Sequential(
        #     nn.Linear(hidden_dim, configs.latent_dim),
        #     nn.Softplus()
        # )

        # Option 2: Conv2d for q(z)
        self.encoder = Encoder(configs)
        self.enc_z_mu = ResidualBlock(self.c_bn, 1, p=configs.drop_rate)
        self.enc_z_covariance = CovarianceGP(self.latent_coords, torch.zeros(self.ny_bn*self.nx_bn), kernel = self.gp_kernel)

        # Decoder
        # Option 1: FC for z->x
        #self.dec_z_to_hidden = nn.Linear(self.batch_size*configs.latent_dim, hidden_dim)
        # Option 2: conv2d for z->x
        self.dec_z_to_hidden = ResidualBlock(1, self.c_bn, p=configs.drop_rate)
        self.decoder = Decoder(configs)
        self.out_layer = nn.Conv2d(configs.c_base*configs.layer_mults[0], c_out, kernel_size=1, stride=1)
        
    def inference(self, x):
        hidden = self.encoder(x)

        # Option 1: FC for q(z)
        #qz_mu = self.enc_z_mu(hidden.view(self.batch_size, -1))
        #qz_logvar = self.enc_z_logvar(hidden.view(self.batch_size, -1))

        # Option 2: Conv2d for q(z) mean
        # TODO: need LKJ prior for the inference term??
        qz_mu = self.enc_z_mu(hidden)
        mvn = self.enc_z_covariance(self.latent_coords)
        qz = mvn.rsample().view(self.ny_bn, self.nx_bn)

        # Estimate cov matrix w/ MC
        qz_cov = self._estimate_cov(mvn.mean, mvn.sample_n(1000))

        inference_terms = ConfigDict()
        inference_terms.qz = qz
        inference_terms.qz_mu = qz_mu
        # inference_terms.qz_logvar = qz_logvar
        inference_terms.qz_cov = qz_cov

        return inference_terms

    def generative(self, qz):
        qz = qz.unsqueeze(0).unsqueeze(0)  # TODO: deal w/ multiple mini-batches?
        hidden = self.dec_z_to_hidden(qz)

        # Option 1: FC for q(z)
        # hidden = self.decoder(hidden.view(self.batch_size, self.c_bn, self.ny_bn, self.nx_bn))
    
        # Option 2: Conv2d for q(z)
        hidden = self.decoder(hidden)
        x_hat = self.out_layer(hidden)
        return x_hat

    def get_loss(self, x, x_hat, inference_terms):
        # Reconstruction loss
        mse = nn.MSELoss(reduction='none')
        loss_NLL = mse(x, x_hat).sum((1,2,3)).mean()
        
        # Option 1: p(z) ~ i.i.d. N(0, 1)
        
        # KL( q(z|x) || p(z) )
        # qz_mu = inference_terms.qz_mu
        # qz_logvar = inference_terms.qz_logvar
        # 
        # pz_mu = torch.zeros_like(qz_mu).to(self.device)
        # pz_std = torch.ones_like(pz_mu) * self.pz_std
        # 
        # loss_KL = kl_divergence(
        #     Normal(qz_mu, torch.exp(0.5*qz_logvar)),
        #     Normal(pz_mu, pz_std)
        # ).sum(-1).mean()

        # Option 2: p(z) ~ MVN(0, I_{k})
        # KL( q(z|x) || p(z) )
        qz_mu = inference_terms.qz_mu
        qz_cov = inference_terms.qz_cov

        pz_mu = torch.zeros(qz_mu.shape[-2]*qz_mu.shape[-1]).to(self.device)
        pz_cov = torch.eye(qz_cov.shape[-1]).to(self.device)
        
        # Compute KL divergence per mini-batch:
        loss_KL = 0.0
        for i in range(self.batch_size):
            qz_mu_batch = qz_mu[i].squeeze().flatten()
            loss_KL_batch = kl_divergence(
                MultivariateNormal(qz_mu_batch, qz_cov),
                MultivariateNormal(pz_mu, pz_cov)
            )
            loss_KL += loss_KL_batch

        loss_KL /= self.batch_size

        loss_configs = ConfigDict()
        loss_configs.tot = loss_NLL + self.beta*loss_KL
        loss_configs.nll = loss_NLL
        loss_configs.kl = loss_KL

        return loss_configs 

    @staticmethod
    def _approx_positive_definite(x, eps=1e-3):
        eigvals, eigvecs = torch.linalg.eigh(x, UPLO='U')
        adj_eigvals = F.relu(eigvals) + eps
        adj_eigvecs, adj_eigvals = torch.real(eigvecs), torch.real(adj_eigvals)
        positive_definite_matrix = adj_eigvecs @ torch.diag(adj_eigvals) @ adj_eigvecs.T
        return positive_definite_matrix
    
    @staticmethod
    def _estimate_cov(mean, samples):
        centered_samples = samples - mean.unsqueeze(0)
        cov = torch.matmul(centered_samples.T, centered_samples) / (samples.size(0) - 1)
        return cov

