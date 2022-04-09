import os

import torch
import pytorch_lightning as pl

from gan import GAN

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)
EPOCHS = 5

dm = MNISTDataModule()
model = GAN(*dm.size())
trainer = pl.Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=EPOCHS,
    progress_bar_refresh_rate=20
)
trainer.fit(model, dm)
