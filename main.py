import os

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.gan import GAN
from src.utils import load_config
from src.data import BirdDataModule

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)


def main():
    conf = load_config("configs/default_config.yaml")
    epochs = conf.get("epochs", 10)
    batch_size = conf.get("batch_size", 32)
    checkpoint_callback = ModelCheckpoint(
        monitor='g_loss',
        dirpath='states',
        filename='state-{epoch:02d}-{val_loss:.2f}'
    )
    wandb_logger = WandbLogger(project='gan', job_type='train')

    dm = BirdDataModule(batch_size=batch_size, num_workers=NUM_WORKERS)
    model = GAN(*dm.size(), conf)
    trainer = pl.Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
        ],

    )
    trainer.fit(model, dm)
    wandb.finish()


if __name__ == "__main__":
    main()
