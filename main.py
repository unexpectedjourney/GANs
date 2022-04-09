import os

import torch
import pytorch_lightning as pl

from src.gan import GAN
from src.utils import load_config

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)


def main():
    conf = load_config("configs/default_config.yaml")
    epochs = conf.get("epochs", 10)
    dm = MNISTDataModule()
    model = GAN(*dm.size(), conf)
    trainer = pl.Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=epochs,
        progress_bar_refresh_rate=20
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
