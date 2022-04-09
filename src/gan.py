from collections import OrderedDict

import wandb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import utils

from models import Discriminator, Generator


class GAN(pl.LightningModule):
    def __init__(
            self,
            channels,
            width,
            height,
            conf,
    ):
        super().__init__()
        self.save_hyperparameters()
        data_shape = (channels, width, height)

        self.conf = conf
        self.latent_dim = conf.get("latent_dim", 100)
        train_params = conf.get("train", {})
        self.lr = train_params.get("lr")
        self.b1 = train_params.get("b1")
        self.b2 = train_params.get("b2")

        self.generator = Generator(
            latent_dim=self.latent_dim,
            img_shape=data_shape
        )
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.latent_dim)

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def train_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        if optimizer_idx == 0:
            self.generated_imgs = self(z)

            sample_imgs = self.generated_imgs[:6]
            grid = utils.make_grid(sample_imgs)
            self.logger.experiment.log({
                "generated_images": [
                    wandb.Image(grid)
                ]
            })
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({
                "loss": g_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
            self.log(
                'g_loss',
                g_loss,
                on_step=True,
                on_epoch=True,
                logger=True,
            )
        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({
                "loss": d_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            })
            self.log(
                'd_loss',
                d_loss,
                on_step=True,
                on_epoch=True,
                logger=True,
            )
        return output

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        sample_imgs = self(z)
        grid = utils.make_grid(sample_imgs)
        self.logger.experiment.log({
            "generated_images": [
                wandb.Image(grid)
            ]
        })
