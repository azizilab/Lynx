import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.distributions import Normal, kl_divergence
from ml_collections import ConfigDict


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, p=0.1):
        super(ResidualBlock, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, c_out),
            nn.SiLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, c_out)
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

        z = x.reshape(x.shape[0], -1)
        return z
        
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

class BetaVAE(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super(BetaVAE, self).__init__()
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

        # flattened dim. before sampling z_mu & z_var
        hidden_dim = self.batch_size*self.c_bn*self.ny_bn*self.nx_bn 
    
        # Encoder
        self.encoder = Encoder(configs)
        self.enc_z_mu = nn.Sequential(
            nn.Linear(hidden_dim, self.batch_size*configs.latent_dim),
            nn.Tanh()
        )

        self.enc_z_logvar = nn.Sequential(
            nn.Linear(hidden_dim, self.batch_size*configs.latent_dim),
            nn.Softplus()
        )

        # Decoder
        self.dec_z_to_hidden = nn.Linear(self.batch_size*configs.latent_dim, hidden_dim)
        self.decoder = Decoder(configs)
        self.out_layer = nn.Conv2d(configs.c_base*configs.layer_mults[0], c_out, kernel_size=1, stride=1)
        
    def inference(self, x):
        z = self.encoder(x)
        qz_mu = self.enc_z_mu(z.flatten())
        qz_logvar = self.enc_z_logvar(z.flatten())
        qz = Normal(qz_mu, torch.exp(0.5*qz_logvar)).rsample()

        inference_terms = ConfigDict()
        inference_terms.qz = qz
        inference_terms.qz_mu = qz_mu
        inference_terms.qz_logvar = qz_logvar

        return inference_terms

    def generative(self, qz):
        hidden = self.dec_z_to_hidden(qz)
        hidden = hidden.view(self.batch_size, self.c_bn, self.ny_bn, self.nx_bn)
        px_z = self.decoder(hidden)
        x_hat = self.out_layer(px_z)
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
        loss_configs.tot = (1-self.beta)*loss_NLL + self.beta*loss_KL
        loss_configs.nll = loss_NLL
        loss_configs.kl = loss_KL

        return loss_configs 
