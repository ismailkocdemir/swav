#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:44:23 2018

@author: nsde
"""

from torch import nn
import numpy as np
from .utils import CenterCrop, Flatten, BatchReshape


def get_encoder(encoder_name):
    models = {'mlp': mlp_encoder,
              'conv': conv_encoder}
    assert (encoder_name in models), 'Encoder not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[encoder_name]


def get_decoder(decoder_name):
    models = {'mlp': mlp_decoder,
              'conv': conv_decoder}
    assert (decoder_name in models), 'Decoder not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[decoder_name]


class mlp_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(mlp_encoder, self).__init__()
        self.flat_dim = np.prod(input_shape)
        self.encoder_mu = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim),
        )
        self.encoder_var = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim),
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z_mu = self.encoder_mu(x)
        z_var = self.encoder_var(x)
        return z_mu, z_var


class mlp_decoder(nn.Module):
    def __init__(self, output_shape, latent_dim):
        super(mlp_decoder, self).__init__()
        self.flat_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.flat_dim),
            nn.ReLU()
        )
        
    def forward(self, z):
        x_mu = self.decoder_mu(z).reshape(-1, *self.output_shape)
        return x_mu

    
class conv_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(conv_encoder, self).__init__()
        self.encoder_mu = nn.Sequential(
            nn.BatchNorm2d(input_shape[0]),
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            Flatten(),
            nn.Linear(64*7*7, latent_dim)
        )
        self.encoder_var = nn.Sequential(
            nn.BatchNorm2d(input_shape[0]),
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            Flatten(),
            nn.Linear(64*7*7, latent_dim),
        )
        
    def forward(self, x):
        z_mu = self.encoder_mu(x)
        z_var = self.encoder_var(x)
        return z_mu, z_var
    

class conv_decoder(nn.Module):
    def __init__(self, output_shape, latent_dim, outputnonlin):
        super(conv_decoder, self).__init__()
        self.output_shape = output_shape
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, 7*7*1),
            BatchReshape((1, 7, 7)),
            nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            Flatten(),
            nn.Linear(64*25*25, 1*28*28),
            nn.ReLu()
        )
        
    def forward(self, z):
        x_mu = self.decoder_mu(z).reshape(-1, *self.output_shape)
        return x_mu