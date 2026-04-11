"""
dataset.py  —  NIH ChestX-ray14 multi-label dataset
=====================================================
Accepts two CSV formats:

  1. Pre-encoded (from scripts/prepare_csv.py):
     Columns: Image Index | image_path | Atelectasis | Cardiomegaly | …
     Labels are already 0/1 integers — used directly.

  2. Raw NIH format (Data_Entry_2017.csv):
     Columns: Image Index | Finding Labels (pipe-separated) | …
     Labels are parsed from the pipe-separated string.

img_dir is only used as a fallback when image_path is not in the CSV.
It can be a flat directory or a root containing images_001…images_012.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


def _build_image_lookup(img_dir: str) -> dict:
    """
    Walk img_dir recursively and return a dict mapping
    filename -> full absolute path.  Handles both flat and
    multi-subfolder layouts (images_001 … images_012).
    """
    lookup = {}
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                lookup[fname] = os.path.join(root, fname)
    return lookup


class ChestXrayDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to CSV.  Accepts both the pre-encoded format produced by
        scripts/prepare_csv.py (one-hot label columns) and the raw NIH
        Data_Entry_2017.csv format (pipe-separated Finding Labels).
    img_dir  : str
        Root directory for images.  Used for the recursive lookup when
        image_path is absent from the CSV.
    transform : callable, optional
        torchvision transforms applied to each PIL image.
    """

    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform
        self.classes   = CLASSES

        # ── image path resolution ─────────────────────────────────────────
        # If the CSV already has an image_path column (from prepare_csv.py),
        # use it directly and skip the expensive directory scan.
        if "image_path" in self.df.columns:
            self._img_lookup = dict(
                zip(self.df["Image Index"], self.df["image_path"])
            )
            print(f"[dataset] Using image_path column from CSV ({len(self._img_lookup):,} entries).")
        else:
            print(f"[dataset] Scanning image directory: {img_dir}")
            self._img_lookup = _build_image_lookup(img_dir)
            print(f"[dataset] Found {len(self._img_lookup):,} images across all subdirectories.")

        # ── label encoding ────────────────────────────────────────────────
        self.labels = np.zeros((len(self.df), len(CLASSES)), dtype=np.float32)

        # Format 1: one-hot columns already present (prepare_csv.py output)
        if all(c in self.df.columns for c in CLASSES):
            self.labels = self.df[CLASSES].values.astype(np.float32)

        # Format 2: raw pipe-separated Finding Labels column
        elif "Finding Labels" in self.df.columns:
            for i, findings in enumerate(self.df["Finding Labels"]):
                if pd.isna(findings) or findings == "No Finding":
                    continue
                for label in str(findings).split("|"):
                    label = label.strip()
                    if label in CLASS2IDX:
                        self.labels[i, CLASS2IDX[label]] = 1.0

        else:
            raise ValueError(
                f"CSV '{csv_path}' has neither one-hot class columns nor a "
                "'Finding Labels' column.  Run scripts/prepare_csv.py first."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["Image Index"]

        img_path = self._img_lookup.get(img_name)
        if img_path is None:
            raise FileNotFoundError(
                f"Image '{img_name}' not found. "
                "Check that img_dir points to the archive root containing "
                "images_001 … images_012 subdirectories."
            )

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

    def get_class_weights(self):
        """Return inverse-frequency weights per class, normalized to mean=1."""
        pos = self.labels.sum(axis=0)
        neg = len(self.labels) - pos
        weights = neg / (pos + 1e-6)
        return weights / weights.mean()
