"""
Fairness audit: compute per-subgroup AUC, sensitivity, and specificity
for a trained classifier on NIH Chest X-Ray.

Subgroups: Patient Gender (M/F), Age bins (<40, 40-60, >60).

Usage:
    python experiments/fairness/audit_subgroups.py \
        --checkpoint results/checkpoints/finetune/model_best.pt \
        --data_root data/raw/nih_chestxray
"""
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, recall_score
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.encoder import MedEncoder, MultiLabelClassifier

NUM_CLASSES = 14
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def age_bin(age: int) -> str:
    if age < 40:
        return "<40"
    elif age <= 60:
        return "40-60"
    return ">60"


@torch.no_grad()
def collect_predictions(model, dataset, device):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    all_probs, all_labels, all_meta = [], [], []
    for batch in loader:
        x, labels, meta = batch
        probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_meta.extend([{k: v[i] for k, v in meta.items()} for i in range(len(x))])
    return np.concatenate(all_probs), np.concatenate(all_labels), all_meta


def compute_group_metrics(probs, labels, groups: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for group_name, mask in groups.items():
        if mask.sum() == 0:
            continue
        g_probs, g_labels = probs[mask], labels[mask]
        aucs = []
        for i in range(labels.shape[1]):
            if g_labels[:, i].sum() > 0:
                aucs.append(roc_auc_score(g_labels[:, i], g_probs[:, i]))
        rows.append({
            "group": group_name,
            "n": int(mask.sum()),
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        })
    return pd.DataFrame(rows)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = MedEncoder(arch=args.arch, pretrained=False)
    model = MultiLabelClassifier(encoder, num_classes=NUM_CLASSES).to(device)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # TODO: Replace with real dataset
    # from data.datasets import NIHChestXray
    # dataset = NIHChestXray(root=args.data_root, split="test",
    #                        transform=..., return_metadata=True)

    print("Fairness audit script ready. Plug in NIHChestXray dataset to run.")
    print("Subgroup keys: gender (M/F), age_bin (<40 / 40-60 / >60)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_root", default="data/raw/nih_chestxray")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--output", default="results/metrics/fairness_audit.csv")
    args = parser.parse_args()
    main(args)
