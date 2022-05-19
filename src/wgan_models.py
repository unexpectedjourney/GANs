import numpy as np
import torch.nn as nn

from .utils import normal_init


class GBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        z = self.linear(x)
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
        self.activation = nn.ReLU()

    def forward(self, x):
        z = self.linear(x)
        z = self.activation(z)
        return z


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            GBlock(latent_dim, 512),
            GBlock(512, 512),
            GBlock(512, 512),
            GBlock(512, 512),
            nn.Linear(512, int(np.prod(self.img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            DBlock(int(np.prod(img_shape)), 512),
            DBlock(512, 512),
            DBlock(512, 512),
            DBlock(512, 512),
            DBlock(512, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
