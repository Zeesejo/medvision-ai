"""
NIH ChestX-ray14 Dataset
========================
Loads images + multi-label targets from Data_Entry_2017.csv.
Returns DataLoaders for train / val / test splits.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(CLASS_NAMES)


class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels          # np.ndarray [N, 14] float32
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def _build_transforms(image_size: int, split: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == 'train':
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


def _parse_labels(finding_labels: pd.Series) -> np.ndarray:
    """Convert pipe-separated label strings to binary matrix."""
    labels = np.zeros((len(finding_labels), NUM_CLASSES), dtype=np.float32)
    for i, findings in enumerate(finding_labels):
        if findings != 'No Finding':
            for finding in findings.split('|'):
                finding = finding.strip()
                if finding in CLASS_NAMES:
                    labels[i, CLASS_NAMES.index(finding)] = 1.0
    return labels


def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    images_subdir: str = 'images',
    labels_file: str = 'Data_Entry_2017.csv',
    train_val_list: str = 'train_val_list.txt',
    test_list: str = 'test_list.txt',
) -> Dict:
    """
    Returns dict with keys: 'train', 'val', 'test', 'pos_weights', 'class_names'
    """
    # ---- Load metadata ----
    df = pd.read_csv(os.path.join(data_dir, labels_file))
    df = df[['Image Index', 'Finding Labels']].copy()

    images_dir = os.path.join(data_dir, images_subdir)
    df['path'] = df['Image Index'].apply(lambda x: os.path.join(images_dir, x))
    df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)

    # ---- Train / val / test split from official lists ----
    with open(os.path.join(data_dir, train_val_list)) as f:
        train_val_files = set(f.read().splitlines())
    with open(os.path.join(data_dir, test_list)) as f:
        test_files = set(f.read().splitlines())

    train_val_df = df[df['Image Index'].isin(train_val_files)].reset_index(drop=True)
    test_df      = df[df['Image Index'].isin(test_files)].reset_index(drop=True)

    # 90/10 split of train_val for train/val
    val_size  = int(0.1 * len(train_val_df))
    val_df    = train_val_df.iloc[:val_size].reset_index(drop=True)
    train_df  = train_val_df.iloc[val_size:].reset_index(drop=True)

    # ---- Parse labels ----
    train_labels = _parse_labels(train_df['Finding Labels'])
    val_labels   = _parse_labels(val_df['Finding Labels'])
    test_labels  = _parse_labels(test_df['Finding Labels'])

    # ---- Positive weights for BCE loss ----
    pos_counts  = train_labels.sum(axis=0)
    neg_counts  = len(train_labels) - pos_counts
    pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)

    # ---- Build datasets ----
    train_dataset = ChestXrayDataset(train_df['path'].tolist(), train_labels, _build_transforms(image_size, 'train'))
    val_dataset   = ChestXrayDataset(val_df['path'].tolist(),   val_labels,   _build_transforms(image_size, 'val'))
    test_dataset  = ChestXrayDataset(test_df['path'].tolist(),  test_labels,  _build_transforms(image_size, 'test'))

    print(f"Dataset sizes — Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    # ---- DataLoaders ----
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'pos_weights':  pos_weights,
        'class_names':  CLASS_NAMES,
    }
    return loaders
