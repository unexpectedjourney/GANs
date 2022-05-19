import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class BirdDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, "image_path"]
        img = Image.open(image_path)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.df.shape[0]


class BirdDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.df = pd.read_csv("data/data.csv")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_subset = self.df[self.df.split == "train"].reset_index(drop=True)
            val_subset = self.df[self.df.split == "valid"].reset_index(drop=True)
            self.train_data = BirdDataset(train_subset, self.transform)
            self.val_data = BirdDataset(val_subset, self.transform)

        if stage == "test" or stage is None:
            test_subset = self.df[self.df.split == "test"].reset_index(drop=True)
            self.test_data = BirdDataset(test_subset, self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
