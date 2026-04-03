"""
NIH ChestX-ray14 Data Pipeline
================================
Dataset   : NIH ChestX-ray14 (112,120 frontal-view X-rays, 14 disease labels)
Download  : https://nihcc.app.box.com/v/ChestXray-NIHCC
            OR via kaggle: `kaggle datasets download -d nih-chest-xrays/data`

Expected directory layout:
    data/
    └── chestxray14/
        ├── images/          # all .png files
        ├── Data_Entry_2017_v2020.csv
        ├── train_val_list.txt
        └── test_list.txt
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(DISEASE_LABELS)   # 14
IMG_SIZE    = 224                   # ViT / ResNet standard input

# ImageNet stats (used for pretrained models)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
def get_transforms(split: str = "train", img_size: int = IMG_SIZE) -> A.Compose:
    """
    Returns albumentations transform pipeline.
    split: 'train' | 'val' | 'test'
    """
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),          # histogram equalisation — good for X-rays
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:  # val / test — no augmentation
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ChestXray14Dataset(Dataset):
    """
    Multi-label classification dataset for NIH ChestX-ray14.

    Args:
        root_dir   : path to `data/chestxray14/`
        file_list  : list of image filenames to use (from official split files)
        transform  : albumentations Compose pipeline
        label_col  : column name in CSV containing pipe-separated findings
    """

    def __init__(
        self,
        root_dir: str,
        file_list: list,
        transform: A.Compose = None,
        label_col: str = "Finding Labels",
    ):
        self.img_dir   = os.path.join(root_dir, "images")
        self.transform = transform
        self.labels    = DISEASE_LABELS

        # Load metadata CSV
        csv_path = os.path.join(root_dir, "Data_Entry_2017_v2020.csv")
        df = pd.read_csv(csv_path)
        df = df[df["Image Index"].isin(set(file_list))].reset_index(drop=True)

        self.image_names = df["Image Index"].tolist()

        # Build binary label matrix  [N, 14]
        self.label_matrix = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
        for i, findings in enumerate(df[label_col]):
            for finding in findings.split("|"):
                finding = finding.strip()
                if finding in DISEASE_LABELS:
                    self.label_matrix[i, DISEASE_LABELS.index(finding)] = 1.0

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image    = np.array(Image.open(img_path).convert("RGB"))  # H x W x 3
        label    = self.label_matrix[idx]                          # [14]

        if self.transform:
            image = self.transform(image=image)["image"]           # [3, H, W] tensor

        return image, torch.tensor(label, dtype=torch.float32)

    def get_pos_weights(self) -> torch.Tensor:
        """
        Compute positive class weights for BCEWithLogitsLoss to handle class imbalance.
        weight_i = (N - pos_i) / pos_i
        """
        pos = self.label_matrix.sum(axis=0).clip(min=1)
        neg = len(self) - pos
        return torch.tensor(neg / pos, dtype=torch.float32)


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------
def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
    img_size: int = IMG_SIZE,
    seed: int = 42,
) -> dict:
    """
    Returns dict with keys: 'train', 'val', 'test'
    Uses official NIH train/val list and test list.

    Args:
        root_dir    : path to `data/chestxray14/`
        batch_size  : images per batch
        num_workers : DataLoader workers (set to 0 on Windows if errors occur)
        val_split   : fraction of train_val list to use as validation
        img_size    : resize resolution
        seed        : random seed for reproducibility
    """
    # Read official split files
    def _read_list(fname):
        path = os.path.join(root_dir, fname)
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    train_val_files = _read_list("train_val_list.txt")
    test_files      = _read_list("test_list.txt")

    # Split train_val into train / val
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_split,
        random_state=seed,
    )

    # Build datasets
    datasets = {
        "train": ChestXray14Dataset(root_dir, train_files, get_transforms("train",  img_size)),
        "val":   ChestXray14Dataset(root_dir, val_files,   get_transforms("val",    img_size)),
        "test":  ChestXray14Dataset(root_dir, test_files,  get_transforms("test",   img_size)),
    }

    # Build loaders
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    loaders["pos_weights"] = datasets["train"].get_pos_weights()
    loaders["num_samples"] = {
        k: len(datasets[k]) for k in ["train", "val", "test"]
    }

    return loaders


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/chestxray14"
    loaders = get_dataloaders(root, batch_size=8, num_workers=0)

    print(f"Train : {loaders['num_samples']['train']:,} images")
    print(f"Val   : {loaders['num_samples']['val']:,} images")
    print(f"Test  : {loaders['num_samples']['test']:,} images")
    print(f"Pos weights shape: {loaders['pos_weights'].shape}")

    imgs, labels = next(iter(loaders["train"]))
    print(f"Batch — images: {imgs.shape}  labels: {labels.shape}")
    print(f"Label example : {labels[0]}")
    print("Sanity check passed ✓")
