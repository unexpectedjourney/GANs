import torch.nn as nn


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class GBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        should_normalize=True,
        should_activate=True,
    ):
        super().__init__()
        self.ctnn = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding,
        )
        self.norm = nn.BatchNorm2d(out_channels, 0.8)
        self.should_normalize = should_normalize
        self.activation = nn.ReLU()
        self.should_activate = should_activate

    def forward(self, x):
        z = self.ctnn(x)
        if self.should_normalize:
            z = self.norm(z)
        if self.should_activate:
            z = self.activation(z)
        return z


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        should_normalize=True,
        should_activate=True,
        dropout_val=0.25,
    ):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
        )
        self.norm = nn.BatchNorm2d(out_channels, 0.8)
        self.should_normalize = should_normalize
        self.activation = nn.LeakyReLU(0.2)
        self.should_activate = should_activate
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x):
        z = self.cnn(x)
        if self.should_normalize:
            z = self.norm(z)
        if self.should_activate:
            z = self.activation(z)
        z = self.dropout(z)
        return z


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            GBlock(latent_dim, 128*8, 4, 1, 0),
            GBlock(128*8, 128*4, 4, 2, 1),
            GBlock(128*4, 128*2, 4, 2, 1),
            GBlock(128*2, 128, 4, 2, 1),
            GBlock(128, 3, 1, 1, 0, should_normalize=False, should_activate=False),
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            DBlock(3, 128, 4, 2, 1, should_normalize=False),
            DBlock(128, 128*2, 4, 2, 1),
            DBlock(128*2, 128*4, 4, 2, 1),
            DBlock(128*4, 128*8, 4, 2, 1),
            DBlock(128*8, 1, 2, 2, 0, should_normalize=False, should_activate=False),
            nn.Sigmoid(),
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, img):
        validity = self.model(img)
        validity = validity.squeeze()
        return validity
