"""Evaluation metrics for multi-label classification."""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    hamming_loss,
)


def compute_auc(targets: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Compute per-class and mean AUC-ROC.

    Args:
        targets: (N, C) binary ground-truth labels
        probs:   (N, C) predicted probabilities

    Returns:
        Dict with per-class AUCs and 'mean_auc'
    """
    from src.data.chestxray import CLASS_NAMES

    aucs = {}
    for i, cls in enumerate(CLASS_NAMES):
        # Skip classes with only one label present in this batch
        if len(np.unique(targets[:, i])) < 2:
            aucs[cls] = float("nan")
        else:
            aucs[cls] = roc_auc_score(targets[:, i], probs[:, i])

    valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
    aucs["mean_auc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return aucs


def compute_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute a full suite of multi-label metrics.

    Args:
        targets:   (N, C) binary ground-truth labels
        probs:     (N, C) predicted probabilities
        threshold: Decision threshold for binary predictions

    Returns:
        Dict with mean_auc, mean_ap, macro_f1, micro_f1, hamming_loss
    """
    preds = (probs >= threshold).astype(int)

    auc_dict = compute_auc(targets, probs)

    return {
        "mean_auc":    auc_dict["mean_auc"],
        "mean_ap":     float(average_precision_score(targets, probs, average="macro")),
        "macro_f1":    float(f1_score(targets, preds, average="macro",  zero_division=0)),
        "micro_f1":    float(f1_score(targets, preds, average="micro",  zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, preds)),
    }
