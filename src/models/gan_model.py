import torch
import torch.nn as nn


class GAN(nn.Module):

    def __init__(self, input_channels=3, output_channels=3):
        super(GAN, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        def deconv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # ======================================
        # Encoder
        # ======================================
        self.enc1 = conv_block(input_channels, 64, use_bn=False)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.bottleneck = conv_block(512, 512)

        # ======================================
        # Decoder
        # ======================================
        self.dec1 = deconv_block(512, 512)
        self.dec2 = deconv_block(1024, 256)
        self.dec3 = deconv_block(512, 128)
        self.dec4 = deconv_block(256, 64)

        self.final_layer = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # ======================================
        # Encoder
        # ======================================
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        # ======================================
        # Decoder
        # ======================================
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e4], dim=1))
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        d4 = self.dec4(torch.cat([d3, e2], dim=1))
        out = self.tanh(self.final_layer(torch.cat([d4, e1], dim=1)))
        return out
