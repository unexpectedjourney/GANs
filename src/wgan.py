import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import utils

from src.wgan_models import Discriminator, Generator


class WGAN(pl.LightningModule):
    def __init__(
            self,
            conf,
    ):
        super().__init__()
        self.save_hyperparameters()
        data_shape = (3, 32, 32)

        self.conf = conf
        self.latent_dim = conf.get("latent_dim", 100)
        train_params = conf.get("train", {})
        self.lr = train_params.get("lr")

        self.generator = Generator(
            latent_dim=self.latent_dim,
            img_shape=data_shape
        )
        self.generator.weight_init(mean=0.0, std=0.02)

        self.discriminator = Discriminator(img_shape=data_shape)
        self.discriminator.weight_init(mean=0.0, std=0.02)

        self.validation_z = torch.randn(8, self.latent_dim)

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        clip_value = 0.01

        loss = 0
        if optimizer_idx == 0:
            loss = -torch.mean(self.discriminator(self(z)))
            self.log('g_loss', loss, logger=True)
        if optimizer_idx == 1:
            loss = -torch.mean(self.discriminator(imgs)) + torch.mean(self.discriminator(self(z)))
            print(
                torch.mean(self.discriminator(imgs)),
                torch.mean(self.discriminator(self(z))),
                flush=True
            )
            for p in self.discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            self.log('d_loss', loss, logger=True)
        return loss

    def configure_optimizers(self):
        n_critic = 5
        opt_g = torch.optim.RMSprop(
            self.generator.parameters(), lr=self.lr,
        )

        opt_d = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=self.lr,
        )
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': n_critic}
        )

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
