# Dataset Manifest Generation

This is the instructions for generating the dataset manifest for the images.

The codebase is written by CVAT. For more detailed information, please check this webpage. [link](https://opencv.github.io/cvat/docs/manual/advanced/dataset_manifest/)

## Environment Setup

To avoide the conflict with the existing python environment, we recommend to create a new virtual environment for this task.

```bash
# SHELL:bash
python3 -m venv .env
. .env/bin/activate
pip install -U pip
pip install -r requirements.in
```


## Generate Dataset Manifest

Examples
```bash
python create.py --output-dir ./output ./Images/
```
In the above example, the dataset manifest will be generated in the `./output` folder. The images are in the `./Images` folder.
(Please note that the `./output` folder should be created **before** running the above command.)


### Script Arguments

```bash
usage: create.py [-h] [--force] [--output-dir .] source

positional arguments:
  source                Source paths

optional arguments:
  -h, --help            show this help message and exit
  --force               Use this flag to prepare the manifest file for video data
                        if by default the video does not meet the requirements
                        and a manifest file is not prepared
  --output-dir OUTPUT_DIR
                        Directory where the manifest file will be saved
```

## Dataset Manifest Format
```
{"version":"1.0"}
{"type":"images"}
{"name":"image1","extension":".jpg","width":720,"height":405,"meta":{"related_images":[]},"checksum":"548918ec4b56132a5cff1d4acabe9947"}
{"name":"image2","extension":".jpg","width":183,"height":275,"meta":{"related_images":[]},"checksum":"4b4eefd03cc6a45c1c068b98477fb639"}
{"name":"image3","extension":".jpg","width":301,"height":167,"meta":{"related_images":[]},"checksum":"0e454a6f4a13d56c82890c98be063663"}

```