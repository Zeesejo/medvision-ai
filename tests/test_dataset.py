import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
N = 20


def make_fake_csv(tmp_path):
    """Create a minimal CSV that ChestXrayDataset can consume."""
    data = {
        'Image Index': [f'img_{i:04d}.png' for i in range(N)],
        'Finding Labels': ['No Finding'] * N,
    }
    for cls in CLASS_NAMES:
        data[cls] = np.random.randint(0, 2, N)
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'labels.csv'
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetClassWeights:
    """Unit tests for get_class_weights() — no real images needed."""

    def _make_dataset(self, tmp_path):
        csv_path = make_fake_csv(tmp_path)
        from dataset import ChestXrayDataset
        with patch('dataset.ChestXrayDataset.__getitem__', return_value=(torch.zeros(3, 224, 224), torch.zeros(14))):
            ds = ChestXrayDataset.__new__(ChestXrayDataset)
            # Manually inject labels array
            labels = np.random.randint(0, 2, (N, len(CLASS_NAMES))).astype(np.float32)
            # Ensure at least 1 positive per class to avoid div-by-zero
            labels[0, :] = 1.0
            ds.labels = labels
            return ds

    def test_weights_shape(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        weights = ds.get_class_weights()
        assert weights.shape == (len(CLASS_NAMES),), "Weights shape mismatch"

    def test_weights_mean_approx_one(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        weights = ds.get_class_weights()
        assert abs(float(np.mean(weights)) - 1.0) < 1e-4, "Weights should be normalized to mean=1"

    def test_weights_all_positive(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        weights = ds.get_class_weights()
        assert np.all(weights > 0), "All weights must be positive"

    def test_weights_no_nan(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        weights = ds.get_class_weights()
        assert not np.any(np.isnan(weights)), "Weights must not contain NaN"
