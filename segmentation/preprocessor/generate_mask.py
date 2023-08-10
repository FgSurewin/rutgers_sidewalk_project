import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from pprint import pprint
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils.path_utils import PathUtils
from sklearn.model_selection import train_test_split


class MaskGenerator:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

        # Find mask images
        self.mask_imgs = self.find_img_paths(
            self.data_dir, "SegmentationClass", ext="png"
        )
        self.mask_file_names = [os.path.basename(img) for img in self.mask_imgs]

        # Find original images
        self.original_imgs = self.find_img_paths(
            self.data_dir, "JPEGImages", ext=["jpg", "tif"]
        )
        self.original_file_names = [os.path.basename(img) for img in self.original_imgs]

        # Parse label file
        label_file_path = os.path.join(self.data_dir, "labelmap.txt")
        self.label_to_rgb, self.rgb_to_label, self.label_to_id = self.parse_label_file(
            label_file_path
        )
        self.save_label_info()

        # Generate dataframe
        self.df = self.data_dataframe()

    def save_label_info(self):
        label_info = {
            "label_to_rgb": self.label_to_rgb,
            "label_to_id": self.label_to_id,
        }
        # Save label info to json file
        label_info_path = os.path.join(self.data_dir, "label_info.json")
        with open(label_info_path, "w") as f:
            json.dump(label_info, f)

    def check_subfolders(self, root_path):
        subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
        return subfolders

    def get_img_patten(self, file_path, ext="png"):
        subfolders = self.check_subfolders(file_path)
        if len(subfolders) == 0:
            return f"*.{ext}"
        else:
            return os.path.join("**", self.get_img_patten(subfolders[0], ext=ext))

    def find_img_paths(self, data_dir, folder_name, ext="png"):
        data_path = os.path.join(data_dir, folder_name)
        seg_imgs = []
        if type(ext) == str:
            patten = self.get_img_patten(data_path, ext=ext)
            finding_patten = os.path.join(data_path, patten)
            seg_imgs = glob(finding_patten)
        elif type(ext) == list:
            for ext_ in ext:
                patten = self.get_img_patten(data_path, ext=ext_)
                finding_patten = os.path.join(data_path, patten)
                seg_imgs += glob(finding_patten)
        return seg_imgs

    def parse_label_file(self, file_path):
        assert os.path.exists(file_path), f"{file_path} does not exist."
        label_to_rgb = {}
        rgb_to_label = {}
        label_to_id = {}
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("#"):  # Skip comments
                    continue
                label, color, _, _ = line.split(":")
                color = tuple(map(int, color.split(",")))
                label_to_rgb[label] = list(color)
                rgb_to_label[color] = int(
                    i - 1
                )  # Assign a class index based on the line number
                label_to_id[label] = i - 1
        return label_to_rgb, rgb_to_label, label_to_id

    def rgb_to_target(self, mask_path, rgb_to_label):
        mask = np.array(Image.open(mask_path))
        h, w, _ = mask.shape
        target = np.zeros((h, w), dtype="int")
        for rgb, label in rgb_to_label.items():
            target[(mask == rgb).all(axis=-1)] = label
        return target

    def split_train_val(self, data, val_ratio=0.2):
        train_data, val_data = train_test_split(
            data, test_size=val_ratio, random_state=42
        )
        return train_data, val_data

    def data_dataframe(self):
        # Create mask file names dataframe
        self.mask_file_names_df = pd.DataFrame(
            self.mask_file_names, columns=["mask_file_name"]
        )
        self.mask_file_names_df["img_name"] = (
            self.mask_file_names_df["mask_file_name"]
            .apply(lambda x: str(x)[:-4])
            .astype(str)
        )

        # Create original file names dataframe
        self.original_file_names_df = pd.DataFrame(
            self.original_file_names, columns=["original_name"]
        )
        self.original_file_names_df["img_name"] = (
            self.original_file_names_df["original_name"]
            .apply(lambda x: str(x)[:-4])
            .astype(str)
        )

        # Join mask and original file names dataframe
        self.final_df = pd.merge(
            self.mask_file_names_df,
            self.original_file_names_df,
            on="img_name",
            how="left",
        )

        return self.final_df

    def save_train_val_data(self, df):
        train_df, test_df = self.split_train_val(df, val_ratio=0.2)
        train_df, val_df = self.split_train_val(train_df, val_ratio=0.2)
        train_df.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.data_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.data_dir, "test.csv"), index=False)

    def process(self):
        # Create output folder
        output_dir = os.path.join(self.data_dir, "SegmentationMask")
        PathUtils.create_dir(output_dir)

        # Convert RGB mask to target mask
        print("Converting RGB mask to target mask...")
        for mask_img in tqdm(self.mask_imgs):
            target = self.rgb_to_target(mask_img, self.rgb_to_label)
            target = Image.fromarray(target.astype("uint8"))
            target.save(os.path.join(output_dir, os.path.basename(mask_img)))

        # Save train/val/test data
        self.save_train_val_data(self.df)


if __name__ == "__main__":
    data_dir = os.path.join("data", "sidewalk")
    mask_generator = MaskGenerator(data_dir)
    mask_generator.process()
    # print(len(mask_generator.df))
    # pprint(mask_generator.df.head(5))
    # pprint(mask_generator.rgb_to_label)
    # pprint(mask_generator.label_to_rgb)
    # pprint(mask_generator.label_to_id)
    # mask_generator.process()
