"""
dataset.py  —  NIH ChestX-ray14 multi-label dataset
=====================================================
Expects a CSV with columns:
  Image Index  |  Finding Labels  (pipe-separated, e.g. "Atelectasis|Effusion")
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


class ChestXrayDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to CSV file with columns 'Image Index' and 'Finding Labels'.
    img_dir  : str
        Directory containing the .png images.
    transform : callable, optional
        torchvision transforms applied to each PIL image.
    """

    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform
        self.classes   = CLASSES

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
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

    def get_class_weights(self):
        """Return inverse-frequency weights per class for weighted sampling."""
        pos = self.labels.sum(axis=0)
        neg = len(self.labels) - pos
        weights = neg / (pos + 1e-6)
        return weights
