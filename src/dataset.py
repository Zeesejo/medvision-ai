"""
dataset.py  —  NIH ChestX-ray14 multi-label dataset
=====================================================
Expects a CSV with columns:
  Image Index  |  Finding Labels  (pipe-separated, e.g. "Atelectasis|Effusion")

img_dir can be either:
  - a flat directory containing all .png files directly, OR
  - a root directory containing subdirectories (e.g. images_001 ... images_012)
    in which case images are located recursively.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

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
        Path to CSV file with columns 'Image Index' and 'Finding Labels'.
    img_dir  : str
        Root directory that contains the images, either directly or inside
        subdirectories such as images_001 … images_012.
    transform : callable, optional
        torchvision transforms applied to each PIL image.
    """

    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform
        self.classes   = CLASSES

        # Build a fast filename → path lookup that works for split subfolders
        print(f"[dataset] Scanning image directory: {img_dir}")
        self._img_lookup = _build_image_lookup(img_dir)
        print(f"[dataset] Found {len(self._img_lookup):,} images across all subdirectories.")

        # Pre-compute binary label vectors
        self.labels = np.zeros((len(self.df), len(CLASSES)), dtype=np.float32)
        for i, findings in enumerate(self.df["Finding Labels"]):
            if pd.isna(findings) or findings == "No Finding":
                continue
            for label in str(findings).split("|"):
                label = label.strip()
                if label in CLASS2IDX:
                    self.labels[i, CLASS2IDX[label]] = 1.0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["Image Index"]

        # Resolve path via lookup dict (handles subdirectories)
        img_path = self._img_lookup.get(img_name)
        if img_path is None:
            raise FileNotFoundError(
                f"Image '{img_name}' not found under '{self.img_dir}'. "
                "Check that img_dir points to the archive root containing "
                "images_001 … images_012 subdirectories."
            )

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

    def get_class_weights(self) -> torch.Tensor:
        """
        Return inverse-frequency pos_weights per class, normalized to mean=1.

        Suitable for direct use as `pos_weight` in nn.BCEWithLogitsLoss.
        Normalization prevents loss explosion for rare classes (e.g. Hernia).
        """
        pos = self.labels.sum(axis=0)
        neg = len(self.labels) - pos
        weights = neg / (pos + 1e-6)
        weights = weights / weights.mean()  # normalize to mean=1
        return torch.tensor(weights, dtype=torch.float32)
