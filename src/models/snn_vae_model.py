import torch
import torch.nn as nn
import norse.torch as snn

import warnings
warnings.filterwarnings(
    "always",
    message="optree.register_pytree_node.*",
    category=UserWarning,
)


class SNNVAE(nn.Module):
    def __init__(self, z_dim=64, T=16):
        """
        SNN Variational Autoencoder model
        :param z_dim: Dimension of the latent space
        :param T: Number of timesteps
        """
        super(SNNVAE, self).__init__()
        self.T = T

        # Encoder layers (same as before)
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.lif1 = snn.LIFCell()

        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif2 = snn.LIFCell()

        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.lif3 = snn.LIFCell()

        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.lif4 = snn.LIFCell()

        self.pool = nn.MaxPool2d(2, 2)

        # Latent space (VAE) with Bernoulli sampling
        self.conv_mu = nn.Conv2d(512, z_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(512, z_dim, kernel_size=1)

        # Decoder layers
        self.dec_conv1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=2, stride=2)
        self.lif_d1 = snn.LIFCell()

        self.dec_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.lif_d2 = snn.LIFCell()

        self.dec_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.lif_d3 = snn.LIFCell()

        self.dec_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.lif_d4 = snn.LIFCell()

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        with torch.no_grad():
            cur1 = self.enc_conv1(x)
            spk1, _ = self.lif1(cur1)
            x1 = self.pool(spk1)

            cur2 = self.enc_conv2(x1)
            spk2, _ = self.lif2(cur2)
            x2 = self.pool(spk2)

            cur3 = self.enc_conv3(x2)
            spk3, _ = self.lif3(cur3)
            x3 = self.pool(spk3)

            cur4 = self.enc_conv4(x3)
            spk4, _ = self.lif4(cur4)
            x4 = self.pool(spk4)

        z_shape = (batch_size, self.conv_mu.out_channels, x4.shape[2], x4.shape[3])

        # Encoder states
        mem1 = self.lif1.initial_state(cur1)
        mem2 = self.lif2.initial_state(cur2)
        mem3 = self.lif3.initial_state(cur3)
        mem4 = self.lif4.initial_state(cur4)

        # Decoder states
        z_dummy = torch.zeros(z_shape, device=device)
        cur_d1 = self.dec_conv1(z_dummy)
        mem_d1 = self.lif_d1.initial_state(cur_d1)

        spk_d1_dummy = torch.zeros_like(cur_d1)
        cur_d2 = self.dec_conv2(spk_d1_dummy)
        mem_d2 = self.lif_d2.initial_state(cur_d2)

        spk_d2_dummy = torch.zeros_like(cur_d2)
        cur_d3 = self.dec_conv3(spk_d2_dummy)
        mem_d3 = self.lif_d3.initial_state(cur_d3)

        spk_d3_dummy = torch.zeros_like(cur_d3)
        cur_d4 = self.dec_conv4(spk_d3_dummy)
        mem_d4 = self.lif_d4.initial_state(cur_d4)

        out_reconstructed = 0
        mu_accum = 0
        logvar_accum = 0

        for t in range(self.T):
            # Encoder
            cur1 = self.enc_conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            x1 = self.pool(spk1)

            cur2 = self.enc_conv2(x1)
            spk2, mem2 = self.lif2(cur2, mem2)
            x2 = self.pool(spk2)

            cur3 = self.enc_conv3(x2)
            spk3, mem3 = self.lif3(cur3, mem3)
            x3 = self.pool(spk3)

            cur4 = self.enc_conv4(x3)
            spk4, mem4 = self.lif4(cur4, mem4)
            x4 = self.pool(spk4)

            mu = self.conv_mu(x4)
            logvar = self.conv_logvar(x4)

            mu_accum += mu
            logvar_accum += logvar

            # Bernoulli sampling
            z = self.bernoulli_sampling(mu)

            # Decoder
            cur_d1 = self.dec_conv1(z)
            spk_d1, mem_d1 = self.lif_d1(cur_d1, mem_d1)

            cur_d2 = self.dec_conv2(spk_d1)
            spk_d2, mem_d2 = self.lif_d2(cur_d2, mem_d2)

            cur_d3 = self.dec_conv3(spk_d2)
            spk_d3, mem_d3 = self.lif_d3(cur_d3, mem_d3)

            cur_d4 = self.dec_conv4(spk_d3)
            spk_d4, mem_d4 = self.lif_d4(cur_d4, mem_d4)

            out = torch.sigmoid(self.final_conv(spk_d4))
            out_reconstructed += out

        x_reconstructed = out_reconstructed / self.T
        mu = mu_accum / self.T
        logvar = logvar_accum / self.T

        return x_reconstructed, mu, logvar

    def bernoulli_sampling(self, mu):
        probabilities = torch.sigmoid(mu)
        return torch.bernoulli(probabilities)
