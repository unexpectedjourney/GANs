import numpy as np
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_features, out_features, should_normalize=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features, 0.8)
        self.should_normalize = should_normalize
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.linear(x)
        if self.should_normalize:
            x = self.norm(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            Block(latent_dim, 128, should_normalize=False),
            Block(128, 256),
            Block(256, 512),
            Block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
