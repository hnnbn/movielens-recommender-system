from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "ml-100k.zip"
EXTRACTED_DIR = RAW_DIR / "ml-100k"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    else:
        print(f"Using existing archive: {ZIP_PATH}")

    if EXTRACTED_DIR.exists():
        print(f"Removing existing extracted directory: {EXTRACTED_DIR}")
        shutil.rmtree(EXTRACTED_DIR)

    print(f"Extracting to {RAW_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(RAW_DIR)

    print("Done.")
    print(f"Dataset path: {EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
