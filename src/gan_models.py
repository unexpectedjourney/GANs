import numpy as np
import torch.nn as nn


class GBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        should_normalize=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features, 0.8)
        self.should_normalize = should_normalize
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        z = self.linear(x)
        if self.should_normalize:
            z = self.norm(z)
        z = self.activation(z)
        return z


class DBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout_val=0.3,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_val)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        z = self.linear(x)
        z = self.activation(z)
        z = self.dropout(z)
        return z


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            GBlock(latent_dim, 128, should_normalize=False),
            GBlock(128, 256),
            GBlock(256, 512),
            GBlock(512, 1024),
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
            DBlock(int(np.prod(img_shape)), 1024),
            DBlock(1024, 512),
            DBlock(512, 256),
            DBlock(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
