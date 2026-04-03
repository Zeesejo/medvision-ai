"""NIH ChestX-ray14 Dataset Loader.

Dataset source:
    https://nihcc.app.box.com/v/ChestXray-NIHCC
    or via Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data

Expected folder structure:
    data/chestxray14/
        Data_Entry_2017.csv
        images/
            00000001_000.png
            00000001_001.png
            ...
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

# All 14 disease labels in the dataset
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

NUM_CLASSES = len(CLASS_NAMES)


def get_transforms(split: str, image_size: int = 224) -> A.Compose:
    """Return albumentations transform pipeline for a given split."""
    mean = [0.485, 0.456, 0.406]  # ImageNet stats (chest X-rays are grayscale→RGB)
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose([
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.4),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:  # val / test — deterministic
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class ChestXrayDataset(Dataset):
    """Multi-label classification dataset for NIH ChestX-ray14."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_indices: List[int],
        transform: Optional[A.Compose] = None,
    ):
        self.df        = pd.read_csv(csv_path).iloc[image_indices].reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labels    = self._encode_labels()

        logger.info(f"Dataset split: {len(self.df)} samples")

    def _encode_labels(self) -> np.ndarray:
        """Convert pipe-separated label strings to multi-hot vectors."""
        labels = np.zeros((len(self.df), NUM_CLASSES), dtype=np.float32)
        for idx, finding in enumerate(self.df["Finding Labels"]):
            for disease in finding.split("|"):
                disease = disease.strip()
                if disease in CLASS_NAMES:
                    labels[idx, CLASS_NAMES.index(disease)] = 1.0
        return labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.df.iloc[idx]["Image Index"]
        img_path = self.image_dir / img_name

        # Load as RGB (X-rays are grayscale, convert for pretrained RGB backbones)
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_dataloaders(
    csv_path: str,
    image_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from a CSV file.

    Args:
        csv_path:    Path to Data_Entry_2017.csv
        image_dir:   Path to images/ folder
        image_size:  Resize target (default 224)
        batch_size:  Samples per batch
        num_workers: Dataloader worker processes
        train_split: Fraction of data for training
        val_split:   Fraction of data for validation
        seed:        Random seed for reproducibility

    Returns:
        (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(csv_path)
    indices = list(range(len(df)))

    test_size = 1.0 - train_split - val_split
    assert test_size > 0, "train_split + val_split must be < 1.0"

    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - train_split), random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(test_size / (val_split + test_size)),
        random_state=seed
    )

    logger.info(f"Split sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    datasets = {
        "train": ChestXrayDataset(csv_path, image_dir, train_idx, get_transforms("train", image_size)),
        "val":   ChestXrayDataset(csv_path, image_dir, val_idx,   get_transforms("val",   image_size)),
        "test":  ChestXrayDataset(csv_path, image_dir, test_idx,  get_transforms("test",  image_size)),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }

    return loaders["train"], loaders["val"], loaders["test"]
