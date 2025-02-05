import torch
from torch import nn
from torch.nn import functional as F
from .encoder_decoder import * 
from .types_ import *

class VanillaVAE_MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 encoder_type='mlp',
                 decoder_type='conv',
                 with_decoder: bool = True,
                 **kwargs) -> None:
        super(VanillaVAE_MLP, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.with_decoder = with_decoder

        if self.with_decoder:
            output_dim  = input_dim
            self.encoder = get_encoder(encoder_type)(input_dim, latent_dim)
            self.decoder = get_decoder(decoder_type)(output_dim, latent_dim + 6)
        else:
            self.encoder = get_encoder(encoder_type)(input_dim + 6, latent_dim)


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        assert self.with_decoder, 'decoder not found: with_decoder was not set.'
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, theta: Tensor, **kwargs) -> List[Tensor]:
        if self.with_decoder:
            mu, log_var = self.encode(input)
        else:
            input_combined = torch.cat([input, theta], dim=1)
            mu, log_var = self.encode(input_combined)
        
        z = self.reparameterize(mu, log_var)
        
        output = {'mu':mu, 'log_var':log_var, 'z':z}

        if self.with_decoder:
            z_combined = torch.cat([z, theta], dim=1)
            output['recons'] = self.decode(z_combined)
            output['input'] = input
        
        return output

    def loss_function(self,
                    mu,
                    log_var,
                    *args,
                    **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """                                                                 
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        assert self.with_decoder, 'decoder not found: with_decoder was not set.'
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        assert self.with_decoder, 'decoder not found: with_decoder was not set.'
        return self.forward(x)[0]

def get_VAE(vae_name):
    models = {'VanillaVAE_MLP': VanillaVAE_MLP,}
    assert (vae_name in models), 'VAE not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[vae_name]

