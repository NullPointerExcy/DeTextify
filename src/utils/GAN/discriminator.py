import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(input_channels, 64, use_bn=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
