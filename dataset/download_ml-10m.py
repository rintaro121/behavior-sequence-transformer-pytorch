import os
import zipfile
from io import BytesIO

import requests

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
BASE_DIR = "./dataset"
DATASET_DIR = os.path.join(BASE_DIR, "ml-1m")
OUTPUT_PATH = os.path.join(BASE_DIR, "sequential_dataset.csv")


def download_and_extract_zip(url, extract_path):
    response = requests.get(url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    zip_file.extractall(extract_path)


if __name__ == "__main__":
    download_and_extract_zip(URL, BASE_DIR)
