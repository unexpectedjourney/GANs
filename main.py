import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.gan import GAN
from src.dcgan import DCGAN
from src.utils import load_config
from src.data import BirdDataModule

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)


def main():
    conf = load_config("configs/default_config.yaml")
    train_params = conf.get("train", {})
    epochs = train_params.get("epochs", 10)
    batch_size = conf.get("batch_size", 32)
    dm = BirdDataModule(batch_size=batch_size, num_workers=NUM_WORKERS)

    gan_type = conf.get("gan_type")
    if gan_type == "gan":
        model = GAN(conf)
    if gan_type == "dcgan":
        model = DCGAN(conf)
    print(gan_type)

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
    main()
