"""
MedVision-AI — Evaluation Script
==================================
Loads best checkpoint and runs full test set evaluation.
Outputs:
  - Per-class AUC table
  - Mean AUC
  - ROC curve plot  (results/roc_curves.png)
  - Confusion matrix per class (results/confusion_matrices.png)

Usage:
    python src/evaluate.py
    python src/evaluate.py --checkpoint results/checkpoints/resnet50_best.pth
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from tqdm import tqdm
from torch.amp import autocast  # L1-001: replaced deprecated torch.cuda.amp.autocast

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.classifier import build_model
from src.data.dataset import get_dataloaders, CLASS_NAMES


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, loader, device, use_amp=True):
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in tqdm(loader, desc='Evaluating'):
        imgs = imgs.to(device)
        with autocast(device_type=device, enabled=use_amp):
            logits = model(imgs)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())  # L1-002: explicit .cpu() before .numpy()
    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_aucs(labels, probs):
    aucs = {}
    for i, name in enumerate(CLASS_NAMES):
        if len(np.unique(labels[:, i])) > 1:
            aucs[name] = roc_auc_score(labels[:, i], probs[:, i])
        else:
            aucs[name] = float('nan')
    valid = [v for v in aucs.values() if not np.isnan(v)]
    aucs['Mean AUC'] = np.mean(valid)
    return aucs


def plot_roc_curves(labels, probs, save_path='results/roc_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # L1-005: dynamic grid instead of hardcoded 3x5 (14 classes wasted 1 slot)
    n = len(CLASS_NAMES)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, name in enumerate(CLASS_NAMES):
        if len(np.unique(labels[:, i])) < 2:
            axes[i].text(0.5, 0.5, 'No positive samples', ha='center')
            axes[i].set_title(name)
            continue
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        auc = roc_auc_score(labels[:, i], probs[:, i])
        axes[i].plot(fpr, tpr, color='#01696f', lw=2, label=f'AUC={auc:.3f}')
        axes[i].plot([0,1],[0,1],'--', color='gray', lw=1)
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel('FPR', fontsize=8)
        axes[i].set_ylabel('TPR', fontsize=8)
        axes[i].legend(fontsize=8)
        axes[i].set_xlim([0,1])
        axes[i].set_ylim([0,1])

    # Hide any remaining empty subplots
    for j in range(len(CLASS_NAMES), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('ROC Curves — NIH ChestX-ray14', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'ROC curves saved → {save_path}')


def print_auc_table(aucs):
    print('\n' + '='*45)
    print(f'{"Disease":<28} {"AUC":>8}')
    print('-'*45)
    for name, auc in aucs.items():
        if name != 'Mean AUC':
            marker = ' ✓' if auc >= 0.80 else ''
            print(f'{name:<28} {auc:>8.4f}{marker}')
    print('-'*45)
    print(f'{"Mean AUC":<28} {aucs["Mean AUC"]:>8.4f}')
    print('='*45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Load model ----
    ckpt_path = args.checkpoint or str(
        Path(cfg['logging']['save_dir']) / f"{cfg['model']['backbone']}_best.pth"
    )
    model = build_model(
        backbone        = cfg['model']['backbone'],
        pretrained      = False,
        checkpoint_path = ckpt_path,
        device          = device,
    )

    # ---- Load test data ----
    loaders = get_dataloaders(
        data_dir    = cfg['data']['data_dir'],
        image_size  = cfg['data']['image_size'],
        batch_size  = cfg['training']['batch_size'],
        num_workers = cfg['data']['num_workers'],
        labels_file = cfg['data']['labels_file'],
        train_val_list = cfg['data']['train_val_list'],
        test_list   = cfg['data']['test_list'],
    )

    # ---- Inference ----
    labels, probs = run_inference(model, loaders['test'], device)

    # ---- AUC table ----
    aucs = compute_aucs(labels, probs)
    print_auc_table(aucs)

    # ---- ROC curves ----
    plot_roc_curves(labels, probs)

    print('\nEvaluation complete!')


if __name__ == '__main__':
    main()
