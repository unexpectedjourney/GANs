import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import utils

from src.lsgan_models import Discriminator, Generator


class LSGAN(pl.LightningModule):
    def __init__(
            self,
            conf,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conf = conf
        self.latent_dim = conf.get("latent_dim", 100)
        train_params = conf.get("train", {})
        self.lr = train_params.get("lr")
        self.b1 = train_params.get("b1")
        self.b2 = train_params.get("b2")

        self.generator = Generator(
            latent_dim=self.latent_dim,
        )
        self.generator.weight_init(mean=0.0, std=0.02)
        self.discriminator = Discriminator()
        self.discriminator.weight_init(mean=0.0, std=0.02)

        self.validation_z = torch.randn(8, self.latent_dim, 1, 1)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        z = torch.randn(imgs.shape[0], self.latent_dim, 1, 1)
        z = z.type_as(imgs)
        loss = 0
        if optimizer_idx == 0:
            valid = torch.ones(imgs.size(0))
            valid = valid.type_as(imgs)

            loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log('g_loss', loss, logger=True)
        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0))
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            fake = torch.zeros(imgs.size(0))
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(self.discriminator(self(z)), fake)

            loss = (real_loss + fake_loss) / 2
            self.log('d_loss', loss, logger=True)
        return loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        return [opt_g, opt_d], []

    def training_epoch_end(self, outputs):
        z = self.validation_z.to(self.device)
        self.generated_imgs = self(z)
        sample_imgs = self.generated_imgs[:6]
        grid = utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(
            "generated_images",
            grid,
            self.current_epoch,
        )
