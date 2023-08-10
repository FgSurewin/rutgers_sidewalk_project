import os
from glob import glob

root_dir = "data"
data_dir = "test"
PATH = os.path.join(root_dir, data_dir)
folder_name = "SegmentationClass"

# root_path = os.path.join(data_dir, folder_name)

# Check if any subfolder inside the root_path
# subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]


# print(list(os.scandir(root_path)))
# print(subfolders)


def get_img_patten(file_path, ext="png"):
    subfolders = [f.path for f in os.scandir(file_path) if f.is_dir()]
    if len(subfolders) == 0:
        return f"*.{ext}"
    else:
        return os.path.join("**", get_img_patten(subfolders[0], ext=ext))


patten = get_img_patten(PATH, ext="jpg")
print(patten)
glob_path = os.path.join(PATH, patten)
print("glob_path: ", glob_path)
seg_imgs = glob(glob_path)
# print(os.path.join(PATH, patten))
print(seg_imgs)
