import pytest
import numpy as np
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
N_CLASSES = len(CLASS_NAMES)
N_SAMPLES = 200


# ---------------------------------------------------------------------------
# compute_aucs
# ---------------------------------------------------------------------------

class TestComputeAucs:

    def _make_data(self, perfect=False):
        np.random.seed(42)
        labels = np.random.randint(0, 2, (N_SAMPLES, N_CLASSES)).astype(np.float32)
        if perfect:
            probs = labels.copy()
        else:
            probs = np.random.rand(N_SAMPLES, N_CLASSES).astype(np.float32)
        return labels, probs

    def test_returns_dict(self):
        from evaluate import compute_aucs
        labels, probs = self._make_data()
        result = compute_aucs(labels, probs)
        assert isinstance(result, dict)

    def test_all_classes_present(self):
        from evaluate import compute_aucs
        labels, probs = self._make_data()
        result = compute_aucs(labels, probs)
        for cls in CLASS_NAMES:
            assert cls in result, f"Missing AUC for class: {cls}"

    def test_auc_range(self):
        from evaluate import compute_aucs
        labels, probs = self._make_data()
        result = compute_aucs(labels, probs)
        for cls, auc in result.items():
            assert 0.0 <= auc <= 1.0, f"AUC out of range for {cls}: {auc}"

    def test_perfect_classifier_auc_one(self):
        from evaluate import compute_aucs
        labels, probs = self._make_data(perfect=True)
        # Ensure each class has both positive and negative examples
        labels[0, :] = 1
        labels[1, :] = 0
        result = compute_aucs(labels, probs)
        for cls, auc in result.items():
            assert auc >= 0.99, f"Perfect classifier should score ~1.0 for {cls}, got {auc:.4f}"


# ---------------------------------------------------------------------------
# plot_roc_curves (smoke test — no display needed)
# ---------------------------------------------------------------------------

class TestPlotRocCurves:

    def test_saves_file(self, tmp_path):
        from evaluate import plot_roc_curves
        np.random.seed(0)
        labels = np.random.randint(0, 2, (N_SAMPLES, N_CLASSES)).astype(np.float32)
        labels[0, :] = 1  # ensure at least one positive
        labels[1, :] = 0  # ensure at least one negative
        probs = np.random.rand(N_SAMPLES, N_CLASSES).astype(np.float32)
        save_path = str(tmp_path / 'roc.png')
        plot_roc_curves(labels, probs, save_path=save_path)
        assert os.path.exists(save_path), "ROC curve PNG was not saved"

    def test_dynamic_grid_no_crash(self, tmp_path):
        """Grid must not crash for any class count 1..20."""
        from evaluate import plot_roc_curves
        import math
        # Patch CLASS_NAMES temporarily inside evaluate module
        import evaluate as ev
        original = ev.CLASS_NAMES
        try:
            for n in [1, 5, 14, 20]:
                ev.CLASS_NAMES = [f'Class{i}' for i in range(n)]
                labels = np.random.randint(0, 2, (50, n)).astype(np.float32)
                labels[0, :] = 1
                labels[1, :] = 0
                probs = np.random.rand(50, n).astype(np.float32)
                save_path = str(tmp_path / f'roc_{n}.png')
                plot_roc_curves(labels, probs, save_path=save_path)
        finally:
            ev.CLASS_NAMES = original
