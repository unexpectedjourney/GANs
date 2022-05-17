import torch.nn as nn


from src.utils import normal_init


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        output_dim,
        blocks_num,
    ):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, 8*8*hidden_dim)
        layers = []
        for i in range(blocks_num):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1))
            layers.append(nn.ELU(True))

            if i < blocks_num - 1:
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        layers.append(nn.Conv2d(hidden_dim, output_dim, 3, 1, 1))
        layers.append(nn.Tanh())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        h0 = self.l1(x)
        z = self.conv(h0)
        return z


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        blocks_num,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(input_dim, hidden_dim, 3, 1, 1))
        layers.append(nn.ELU(True))
        for i in range(1, blocks_num+1):
            layers.append(nn.Conv2d(i * hidden_dim, i * hidden_dim, 3, 1, 1))
            layers.append(nn.ELU(True))

            if i < blocks_num:
                layers.append(nn.Conv2d(i * hidden_dim, (i+1) * hidden_dim, 3, 2, 1))
            else:
                layers.append(nn.Conv2d(i * hidden_dim, (i+1) * hidden_dim, 3, 1, 1))
            layers.append(nn.ELU(True))

        self.conv = nn.Sequential(*layers)
        self.l1 = nn.Linear(8*8*i*hidden_dim, latent_dim)

    def forward(self, x):
        z = self.conv(x)
        z = self.l1(z)
        return z


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        output_dim,
        blocks_num,
    ):
        super().__init__()
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, blocks_num)

    def forward(self, z):
        x = self.decoder(z)
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        output_dim,
        blocks_num,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim, blocks_num)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, blocks_num)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
