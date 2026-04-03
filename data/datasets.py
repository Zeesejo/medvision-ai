"""
PyTorch Dataset wrappers for NIH Chest X-Ray and MedMNIST.
"""
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


class NIHChestXray(Dataset):
    """
    NIH Chest X-Ray14 multi-label classification dataset.

    Args:
        root: Path to dataset root (contains images/ and Data_Entry_2017.csv)
        split: 'train' | 'val' | 'test'
        transform: torchvision transforms
        return_metadata: if True, also return dict with patient_id, age, gender
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_metadata = return_metadata

        csv_path = self.root / "Data_Entry_2017.csv"
        split_file = self.root / f"{split}_list.txt"

        df = pd.read_csv(csv_path)
        with open(split_file) as f:
            split_files = set(f.read().splitlines())

        self.df = df[df["Image Index"].isin(split_files)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_labels(self, label_str: str) -> np.ndarray:
        labels = np.zeros(len(DISEASE_LABELS), dtype=np.float32)
        for disease in label_str.split("|"):
            disease = disease.strip()
            if disease in DISEASE_LABELS:
                labels[DISEASE_LABELS.index(disease)] = 1.0
        return labels

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root / "images" / row["Image Index"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = self._encode_labels(row["Finding Labels"])

        if self.return_metadata:
            meta = {
                "patient_id": row["Patient ID"],
                "age": row["Patient Age"],
                "gender": row["Patient Gender"],
                "filename": row["Image Index"],
            }
            return image, labels, meta

        return image, labels
