import torch
import torch.nn as nn
import norse.torch as snn

class SpikingVAE(nn.Module):
    def __init__(self, latent_dim=64, time_steps=16, img_height=128, img_width=96):
        super(SpikingVAE, self).__init__()
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width

        # Encoder layers
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder_lif1 = snn.LIFCell()

        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_lif2 = snn.LIFCell()

        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_lif3 = snn.LIFCell()

        self.encoder_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder_lif4 = snn.LIFCell()

        self.pooling = nn.MaxPool2d(2, 2)

        # Calculate feature size after encoding
        self.feature_size = self._calculate_feature_size()

        # Latent space parameters
        self.mu_layer = nn.Linear(self.feature_size, self.latent_dim)
        self.logvar_layer = nn.Linear(self.feature_size, self.latent_dim)

        # Recurrent layer for autoregressive sampling
        self.latent_recurrent = nn.GRUCell(latent_dim, latent_dim)

        # Decoder layers
        self.decoder_fc = nn.Linear(self.latent_dim, 512 * 8 * 8)

        self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.decoder_lif1 = snn.LIFCell()

        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_lif2 = snn.LIFCell()

        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_lif3 = snn.LIFCell()

        self.decoder_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.decoder_lif4 = snn.LIFCell()

        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.expand_fc = nn.Linear(self.latent_dim, 512 * 8 * 8)

    def _calculate_feature_size(self):
        # Compute the size of the features after the encoder layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.img_height, self.img_width)
            x = self.encoder_conv1(dummy_input)
            spk1, _ = self.encoder_lif1(x)
            x = self.pooling(spk1)

            x = self.encoder_conv2(x)
            spk2, _ = self.encoder_lif2(x)
            x = self.pooling(spk2)

            x = self.encoder_conv3(x)
            spk3, _ = self.encoder_lif3(x)
            x = self.pooling(spk3)

            x = self.encoder_conv4(x)
            spk4, _ = self.encoder_lif4(x)
            x = self.pooling(spk4)

            feature_size = x.view(1, -1).size(1)
        return feature_size

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Encoder forward pass
        x = self.encoder_conv1(x)
        spk1, _ = self.encoder_lif1(x)
        x = self.pooling(spk1)

        x = self.encoder_conv2(x)
        spk2, _ = self.encoder_lif2(x)
        x = self.pooling(spk2)

        x = self.encoder_conv3(x)
        spk3, _ = self.encoder_lif3(x)
        x = self.pooling(spk3)

        x = self.encoder_conv4(x)
        spk4, _ = self.encoder_lif4(x)
        x = self.pooling(spk4)

        x_flat = x.view(batch_size, -1)
        mu = self.mu_layer(x_flat)
        logvar = self.logvar_layer(x_flat)

        # Initialize latent variables and hidden state
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        h = torch.zeros(batch_size, self.latent_dim, device=device)

        # Initialize accumulators
        reconstructed_output = 0
        mu_accum = 0
        logvar_accum = 0

        # Autoregressive sampling and decoding
        for t in range(self.time_steps):
            z = self.autoregressive_sampling(mu, z, h)

            # Decoder
            z_decoded = self.expand_fc(z)
            z_decoded = z_decoded.view(batch_size, 512, 8, 8)

            x = self.decoder_conv1(z_decoded)
            spk_d1, _ = self.decoder_lif1(x)

            x = self.decoder_conv2(spk_d1)
            spk_d2, _ = self.decoder_lif2(x)

            x = self.decoder_conv3(spk_d2)
            spk_d3, _ = self.decoder_lif3(x)

            x = self.decoder_conv4(spk_d3)
            spk_d4, _ = self.decoder_lif4(x)

            out = torch.sigmoid(self.output_conv(spk_d4))
            reconstructed_output += out

            # Update accumulators
            mu_accum += mu
            logvar_accum += logvar

        # Average over time steps
        x_reconstructed = reconstructed_output / self.time_steps
        mu = mu_accum / self.time_steps
        logvar = logvar_accum / self.time_steps

        return x_reconstructed, mu, logvar

    def autoregressive_sampling(self, mu, prev_z, h):
        # Update hidden state
        h = self.latent_recurrent(prev_z, h)
        mu = mu + h
        probabilities = torch.sigmoid(mu)
        z = torch.bernoulli(probabilities)
        return z
