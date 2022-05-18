import torch
import pytorch_lightning as pl
from torchvision import utils

from src.began_models import Discriminator, Generator


class BEGAN(pl.LightningModule):
    def __init__(
            self,
            conf,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conf = conf
        self.latent_dim = conf.get("latent_dim", 128)
        self.hidden_dim = conf.get("hidden_dim", 128)
        self.blocks_num = conf.get("blocks_num", 3)
        train_params = conf.get("train", {})
        self.lr = train_params.get("lr")
        self.b1 = train_params.get("b1")
        self.b2 = train_params.get("b2")
        self.k = 0
        self.gamma = 0.75
        self.lambda_k = 0.001
        self.channels = 3

        self.generator = Generator(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.channels,
            blocks_num=self.blocks_num,
        )
        self.generator.weight_init(mean=0.0, std=0.02)

        self.discriminator = Discriminator(
            input_dim=self.channels,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.channels,
            blocks_num=self.blocks_num,
        )
        self.discriminator.weight_init(mean=0.0, std=0.02)

        self.validation_z = -2 * torch.randn(8, self.latent_dim) + 1

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return torch.mean(torch.abs(y_hat - y))

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        z = torch.randn(imgs.shape[0], self.latent_dim, 1, 1)
        z = z.type_as(imgs)
        loss = 0
        if optimizer_idx == 0:
            gen_imgs = self.generator(z)
            loss = self.adversarial_loss(self.discriminator(gen_imgs), gen_imgs)
            self.log('g_loss', loss, logger=True)
        if optimizer_idx == 1:
            gen_imgs = self.generator(z).detach()
            g_loss = self.adversarial_loss(self.discriminator(gen_imgs), gen_imgs)
            r_loss = self.adversarial_loss(self.discriminator(imgs), imgs)

            loss = r_loss - self.k * g_loss
            diff = torch.mean(self.gamma * r_loss - g_loss)
            self.k = self.k + self.lambda_k * diff.item()
            self.k = min(max(0, self.k), 1)

            M = (r_loss + torch.abs(diff)).item()
            self.log('d_loss', loss, logger=True)
            self.log('M', M, logger=True)
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
