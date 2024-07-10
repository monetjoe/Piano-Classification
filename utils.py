import os
import torch
import zipfile
import requests
from tqdm import tqdm

LOGS_DIR = "./logs"
MODEL_DIR = "./model"


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def url_download(url: str, fname: str, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"Downloading: {url}")
            resp = requests.get(url, stream=True)
            # Check the response status code (raise an exception if it's not in the range 200-299)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(fname, "wb") as file, tqdm(
                desc=fname,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            print(f"Download of {url} completed.")
            return

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP error occurred: {errh}")
            retry_count += 1
            continue
        except requests.exceptions.ConnectionError as errc:
            print(f"Connection error occurred: {errc}")
            retry_count += 1
            continue
        except requests.exceptions.Timeout as errt:
            print(f"Timeout error occurred: {errt}")
            retry_count += 1
            continue
        except Exception as err:
            print(f"Other error occurred: {err}")
            retry_count += 1
            continue

    else:
        print(
            f"Error: the operation could not be completed after {max_retries} retries."
        )
        exit()


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, "r")
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print("This is not zip")


def toCUDA(x):
    if hasattr(x, "cuda"):
        if torch.cuda.is_available():
            return x.cuda()

    return x
