import os
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.seg_dataset import SegDataset
from torchvision.transforms import transforms as T


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers=4,
        random_seed=42,
        img_transform=None,
        mask_transform=None,
        **kwargs
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        if img_transform is None:
            self.img_transform = T.Compose(
                [
                    T.Resize((512, 512)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.img_transform = img_transform
        if mask_transform is None:
            self.mask_transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
        else:
            self.mask_transform = mask_transform
        self.kwargs = kwargs
        self.num_batches_train = None

    def setup(self, stage: str) -> None:
        self.dataset_class = SegDataset
        self.train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        self.valid_df = pd.read_csv(os.path.join(self.data_dir, "val.csv"))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        # Load data
        if stage == "fit":
            self.train_dataset = self.dataset_class(
                data_dir=self.data_dir,
                df=self.train_df,
                img_transform=self.img_transform,
                mask_transform=self.mask_transform,
                **self.kwargs,
            )
            self.valid_dataset = self.dataset_class(
                data_dir=self.data_dir,
                df=self.valid_df,
                img_transform=self.img_transform,
                mask_transform=self.mask_transform,
                **self.kwargs,
            )

        if stage == "test":
            self.test_dataset = self.dataset_class(
                data_dir=self.data_dir,
                df=self.test_df,
                img_transform=self.img_transform,
                mask_transform=self.mask_transform,
                **self.kwargs,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        self.num_batches_train = len(dataloader)
        return dataloader

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


if __name__ == "__main__":
    data_module = DataModule(
        data_dir="data",
        batch_size=4,
        num_workers=8,
        random_seed=42,
    )
    data_module.setup(stage="fit")
    print(len(data_module.train_dataset), len(data_module.valid_dataset))
    print(len(data_module.train_dataloader()), len(data_module.val_dataloader()))
    data, mask = next(iter(data_module.train_dataloader()))
    print(data.shape, mask.shape)
