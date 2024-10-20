import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetVAE(nn.Module):
    def __init__(self, z_dim=64):
        super(UNetVAE, self).__init__()

        # Encoder (U-Net)
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Latent space (VAE) - replace fully connected layers with conv layers
        self.conv_mu = nn.Conv2d(256, z_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(256, z_dim, kernel_size=1)

        # Decoder (U-Net)
        self.dec_conv1 = nn.ConvTranspose2d(z_dim, 256, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        mu = self.conv_mu(x)     # Output shape: (batch_size, z_dim, H', W')
        logvar = self.conv_logvar(x)  # Output shape: (batch_size, z_dim, H', W')
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.dec_conv1(z))
        z = F.relu(self.dec_conv2(z))
        z = F.relu(self.dec_conv3(z))
        return torch.sigmoid(self.final_conv(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
