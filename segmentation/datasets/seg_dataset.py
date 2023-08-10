import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T


class SegDataset(Dataset):
    def __init__(
        self,
        data_dir,
        df,
        img_transform=None,
        mask_transform=None,
        origin_img_folder="JPEGImages",
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.df = df
        self.img_names = df["original_name"].values
        self.mask_names = df["mask_file_name"].values

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.original_img_paths[index]
        mask_path = self.mask_paths[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = self.img_transform(img)
        mask = self.mask_transform(mask).squeeze(0)
        mask = mask.long()

        return img, mask


if __name__ == "__main__":
    data_dir = os.path.join("data", "sidewalk")
    seg_dataset = SegDataset(
        data_dir=data_dir,
        df=pd.read_csv(os.path.join(data_dir, "train.csv")),
    )
    print(len(seg_dataset))

    img, mask = seg_dataset[0]
    print(img.shape, mask.shape)
    print("img dtype: ", img.dtype)
    print("mask dtype: ", mask.dtype)

    # Plot the image
    # to_pil_image = T.ToPILImage()
    # img_pil = to_pil_image(img)
    # plt.imshow(img_pil)
    # plt.show()
    # mask_pil = to_pil_image(mask)
    # plt.imshow(mask_pil)
    # plt.show()
