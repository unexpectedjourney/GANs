from collections import OrderedDict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import utils

from models import Discriminator, Generator


class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def train_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        z = torch.randn(imgs.shape[0], latent_dim)
        z = z.type_as(imgs)

        if optimizer_idx == 0:
            self.generated_imgs = self(z)

            sample_imgs = self.generated_imgs[:6]
            grid = utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({
                "loss": g_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })

        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({
                "loss": d_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        sample_imgs = self(z)
        grid = utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
