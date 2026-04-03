"""
Download and prepare the NIH Chest X-Ray14 dataset.
Source: https://nihcc.app.box.com/v/ChestXray-NIHCC

Usage:
    python data/scripts/download_nih_chestxray.py --output_dir data/raw/nih_chestxray
"""
import argparse
import os
import urllib.request
from pathlib import Path

CSV_URL = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/metadata.csv"  # placeholder

DATASET_URLS = [
    # Official Box links — paste direct URLs here after accepting NIH data use agreement
    # https://nihcc.app.box.com/v/ChestXray-NIHCC
]


def download_nih(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}")
    print("TODO: Add direct download URLs from NIH Box after accepting the data use agreement.")
    print("Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/raw/nih_chestxray")
    args = parser.parse_args()
    download_nih(args.output_dir)
