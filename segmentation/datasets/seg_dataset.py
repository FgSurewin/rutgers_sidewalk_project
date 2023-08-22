import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import albumentations as A


class SegDataset(Dataset):
    def __init__(
        self,
        data_dir,
        df,
        origin_img_folder="JPEGImages",
        is_augment=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.df = df
        self.img_names = df["original_name"].values
        self.mask_names = df["mask_file_name"].values
        self.is_augment = is_augment

        # Get image paths
        original_img_dir = os.path.join(self.data_dir, origin_img_folder)
        assert os.path.exists(original_img_dir), f"{original_img_dir} does not exist."
        self.original_img_paths = [
            os.path.join(original_img_dir, img_name) for img_name in self.img_names
        ]

        # Get mask paths
        mask_dir = os.path.join(self.data_dir, "SegmentationMask")
        assert os.path.exists(mask_dir), f"{mask_dir} does not exist."
        self.mask_paths = [
            os.path.join(mask_dir, img_name) for img_name in self.mask_names
        ]

        # Define common transformations for both image and mask
        self.common_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
            ]
        )

        # Define individual resize transformations
        self.img_resize_transform = A.Resize(512, 512, interpolation=cv2.INTER_LINEAR)
        self.mask_resize_transform = A.Resize(512, 512, interpolation=cv2.INTER_NEAREST)

        # Define image normalization transformation
        self.img_nor_transform = A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.kwargs = kwargs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.original_img_paths[index]
        mask_path = self.mask_paths[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img_np = np.array(img)
        mask_np = np.array(mask)

        # Apply individual resize transformations
        img_resize = self.img_resize_transform(image=img_np)["image"]
        mask_resize = self.mask_resize_transform(image=mask_np)["image"]

        # Apply common transformations
        if self.is_augment:
            transformed = self.common_transform(image=img_resize, mask=mask_resize)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            img = img_resize
            mask = mask_resize

        # Apply image normalization
        img = self.img_nor_transform(image=img)["image"]
        # img = self.img_nor_transform(image=img_resize)["image"]

        # Convert to tensor
        img = torch.tensor(np.transpose(img, (2, 0, 1)))
        mask = torch.tensor(mask).long()
        # mask = torch.tensor(mask_resize).long()

        # mask = mask.long()

        return img, mask


if __name__ == "__main__":
    data_dir = os.path.join("data", "sidewalk")
    seg_dataset = SegDataset(
        data_dir=data_dir,
        df=pd.read_csv(os.path.join(data_dir, "train.csv")),
        is_augment=False,
    )
    print(len(seg_dataset))

    img, mask = seg_dataset[0]
    print(img.shape, mask.shape)
    print("img dtype: ", img.dtype)
    print("mask dtype: ", mask.dtype)

    # Plot the image
    # to_pil_image = T.ToPILImage()
    img_np = img.permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(img_np)
    plt.show()
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    plt.imshow(mask_np)
    plt.show()
