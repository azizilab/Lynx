import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x_outs = []
        for layer in self.layers:
            x_out = layer(x)
            x_outs.append(x_out)
            x = self.downscale_layer(x_out)

        z = x.reshape(x.shape[0], -1)
        return x_outs, z
        

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
            ResidualBlock(c_hiddens[i]+c_hiddens[i+1], c_hiddens[i+1], p=p)
            for i in range(len(c_hiddens)-1)
        ])
        
    def forward(self, x_outs, x):
        for layer, x_out in zip(self.layers, x_outs[::-1]):
            x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear') 
            x_concat = torch.cat((x, x_out), axis=1)  # shape: [B, C1+C2, Y, X]
            x = layer(x_concat)
        
        return x


class BetaVAE(nn.Module):
    def __init__(
        self,
        configs,
        prior_std: float = 0.01  
    ):
        super(BetaVAE, self).__init__()
        c_out = configs.c_out
        nlayers = len(configs.layer_mults)
        device = configs.device
        
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.beta = configs.beta  # weights for B-VAE
        self.pz_std = torch.tensor(prior_std).to(device)

        self.c_bn = configs.c_base * configs.layer_mults[-1]

        ny_in, nx_in = configs.ydim, configs.xdim
        self.ny_bn = ny_in // (2**nlayers)
        self.nx_bn = nx_in // (2**nlayers)
        
        # Encoder
        self.encoder = Encoder(configs)
        self.enc_z_mu = nn.Sequential(
            nn.Linear(self.c_bn*self.ny_bn*self.nx_bn, configs.latent_dim),
            nn.Tanh()
        )

        self.enc_z_logvar = nn.Linear(self.c_bn*self.ny_bn*self.nx_bn, configs.latent_dim)

        # Decoder
        # TODO: check non-negative constraint in Decoder
        self.dec_z_to_hidden = nn.Sequential(
            nn.Linear(configs.latent_dim, self.c_bn*self.ny_bn*self.nx_bn, configs.latent_dim),
            nn.Softplus()
        )

        self.decoder = Decoder(configs)
        self.out_layer = nn.Conv2d(configs.c_base*configs.layer_mults[0], c_out, kernel_size=1, stride=1)

    def inference(self, x):
        x_encs, z = self.encoder(x)
        qz_mu = self.enc_z_mu(z.flatten())
        qz_logvar = self.enc_z_logvar(z.flatten())
        qz = Normal(qz_mu, torch.exp(0.5*qz_logvar)).rsample()

        inference_terms = ConfigDict()
        inference_terms.qz = qz
        inference_terms.qz_mu = qz_mu
        inference_terms.qz_logvar = qz_logvar
        inference_terms.x_encs = x_encs

        return inference_terms

    def generative(self, x_encs, qz):
        hidden = self.dec_z_to_hidden(qz)
        hidden = hidden.view(self.batch_size, self.c_bn, self.ny_bn, self.nx_bn)
        px_z = self.decoder( x_encs, hidden)
        x_hat = self.out_layer(px_z)
        return torch.sigmoid(x_hat)

    def get_loss(self, x, x_hat, pz_mu, inference_terms):
        # Reconstruction loss
        mse = nn.MSELoss(reduction='none')
        loss_NLL = mse(x, x_hat).sum((1,2,3)).mean()

        # KL divergence
        pz_mu = pz_mu.squeeze().flatten()
        pz_std = torch.ones_like(pz_mu) * self.pz_std

        qz_mu = inference_terms.qz_mu
        qz_logvar = inference_terms.qz_logvar


        loss_KL = kl_divergence(
            Normal(qz_mu, torch.exp(0.5*qz_logvar)),
            Normal(pz_mu, pz_std)
        ).sum(-1).mean()

        loss_configs = ConfigDict()
        loss_configs.tot = (1-self.beta)*loss_NLL + self.beta*loss_KL
        loss_configs.nll = loss_NLL
        loss_configs.kl = loss_KL

        return loss_configs 
    