import os

import fire
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.gan import GAN
from src.dcgan import DCGAN
from src.began import BEGAN
from src.wgan import WGAN
from src.utils import load_config
from src.data import BirdDataModule

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)


def main(gan_type=None):
    conf = load_config("configs/default_config.yaml")
    if gan_type == "gan":
        conf = load_config("configs/gan_config.yaml")
        model = GAN(conf)
    if gan_type == "dcgan":
        conf = load_config("configs/dcgan_config.yaml")
        model = DCGAN(conf)
    if gan_type == "began":
        conf = load_config("configs/began_config.yaml")
        model = BEGAN(conf)
    if gan_type == "wgan":
        conf = load_config("configs/wgan_config.yaml")
        model = WGAN(conf)
    else:
        print("Bye")
        return

    print(gan_type)

    train_params = conf.get("train", {})
    epochs = train_params.get("epochs", 10)
    batch_size = conf.get("batch_size", 32)
    dm = BirdDataModule(batch_size=batch_size, num_workers=NUM_WORKERS)

    filename = gan_type + '_state-{epoch:02d}-{g_loss:.2f}'
    checkpoint_callback = ModelCheckpoint(
        monitor='g_loss',
        dirpath='states',
        filename=filename
    )
    tb_logger = TensorBoardLogger("logs", name=gan_type)

    trainer = pl.Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=epochs,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
        ],

    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    fire.Fire(main)
