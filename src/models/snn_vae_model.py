import torch
import torch.nn as nn
import norse.torch as snn  # Using norse for LIF neurons

from src.models.components.LIFSpike import LIFSpike


class SpikingVAE(nn.Module):
    def __init__(self, latent_dim=64, time_steps=16, img_height=128, img_width=96):
        super(SpikingVAE, self).__init__()
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width

        # ======================================
        # Encoder (Spiking with LIF neurons)
        # ======================================
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            LIFSpike(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            LIFSpike(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            LIFSpike(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            LIFSpike(),
            nn.MaxPool2d(2, 2)
        )
        self.feature_size = self._calculate_feature_size()

        # Latent space layers (mu and logvar for the VAE part)
        self.mu_layer = nn.Linear(self.feature_size, self.latent_dim)
        self.logvar_layer = nn.Linear(self.feature_size, self.latent_dim)

        # Recurrent layer for autoregressive sampling
        self.latent_recurrent = nn.GRUCell(latent_dim, latent_dim)

        # ======================================
        # Decoder (Spiking with LIF neurons)
        # ======================================
        self.expand_fc = nn.Linear(self.latent_dim, 512 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            LIFSpike(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            LIFSpike(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            LIFSpike(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            LIFSpike(),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _calculate_feature_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.img_height, self.img_width)
            x = self.encoder(dummy_input)  # No need to unpack tuples
            feature_size = x.view(1, -1).size(1)
        return feature_size

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # ======================================
        # Encoder
        # ======================================
        spikes = self.encoder(x)

        x_flat = spikes.view(batch_size, -1)
        mu = self.mu_layer(x_flat)
        logvar = self.logvar_layer(x_flat)

        z = torch.zeros(batch_size, self.latent_dim, device=device)
        h = torch.zeros(batch_size, self.latent_dim, device=device)

        reconstructed_output = 0
        mu_accum = 0
        logvar_accum = 0

        # Autoregressive sampling and decoding over time steps
        for t in range(self.time_steps):
            z = self.autoregressive_sampling(mu, z, h)

            # ======================================
            # Decoder (passing through spiking decoder)
            # ======================================
            z_decoded = self.expand_fc(z)
            z_decoded = z_decoded.view(batch_size, 512, 8, 8)

            out_spikes = self.decoder(z_decoded)

            reconstructed_output += out_spikes
            mu_accum += mu
            logvar_accum += logvar

        x_reconstructed = reconstructed_output / self.time_steps
        mu = mu_accum / self.time_steps
        logvar = logvar_accum / self.time_steps

        return x_reconstructed, mu, logvar

    def autoregressive_sampling(self, mu, prev_z, h):
        """
        Autoregressive sampling from latent space.
        Generates new latent variables based on previous time step.
        """
        h = self.latent_recurrent(prev_z, h)
        mu = mu + h
        probabilities = torch.sigmoid(mu)
        z = torch.bernoulli(probabilities)
        return z
