import hydra
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.datasets.ae_dataset import ZeroShotDataset


class ZeroShotDataModule(LightningDataModule):
    def __init__(self, train, val, test, batch_size=32):
        super().__init__()

        self.train_dataset = ZeroShotDataset(train)
        self.val_dataset = ZeroShotDataset(val)
        self.test_dataset = ZeroShotDataset(test)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


