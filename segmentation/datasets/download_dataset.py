import requests
from tqdm import tqdm
import argparse


def download_file(url, filename):
    response = requests.get(url, stream=True)

    # Get the total file size
    total_size = int(response.headers.get("Content-Length", 0))

    # Initialize a tqdm progress bar
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(filename, "wb") as file:
        for data in response.iter_content(chunk_size=8192):
            # Update the progress bar
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    # If the total size does not match the downloaded size, raise an error
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR, something went wrong")


if __name__ == "__main__":
    argpaser = argparse.ArgumentParser()
    argpaser.add_argument(
        "--url", type=str, required=True, help="URL to download the dataset"
    )
    argpaser.add_argument(
        "--filename", type=str, required=True, help="Filename to save the dataset"
    )

    args = argpaser.parse_args()
    download_file(str(args.url), str(args.filename))
