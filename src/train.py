"""
MedVision-AI — Training Script
================================
Usage:
    python src/train.py                        # uses config.yaml defaults
    python src/train.py --backbone vit_base_patch16_224
    python src/train.py --epochs 50 --lr 3e-4
"""

import os
import sys
import argparse
import time
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.classifier import build_model
from src.models.losses import WeightedBCELoss, FocalLoss, AsymmetricLoss
from src.data.dataset import get_dataloaders


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
def load_config(path: str = 'config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='MedVision-AI Trainer')
    parser.add_argument('--config',   default='config.yaml')
    parser.add_argument('--backbone', default=None)
    parser.add_argument('--epochs',   type=int, default=None)
    parser.add_argument('--lr',       type=float, default=None)
    parser.add_argument('--batch',    type=int, default=None)
    parser.add_argument('--loss',     default=None, choices=['bce', 'focal', 'asl'])
    parser.add_argument('--data_dir', default=None)
    return parser.parse_args()


# ─────────────────────────────────────────
# Loss factory
# ─────────────────────────────────────────
def build_loss(loss_name: str, pos_weights: torch.Tensor):
    if loss_name == 'bce':
        return WeightedBCELoss(pos_weights)
    elif loss_name == 'focal':
        return FocalLoss(gamma=2.0)
    elif loss_name == 'asl':
        return AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    raise ValueError(f'Unknown loss: {loss_name}')


# ─────────────────────────────────────────
# AUC metric
# ─────────────────────────────────────────
def compute_auc(all_labels: np.ndarray, all_probs: np.ndarray, class_names: list) -> dict:
    aucs = {}
    for i, name in enumerate(class_names):
        if len(np.unique(all_labels[:, i])) > 1:
            aucs[name] = roc_auc_score(all_labels[:, i], all_probs[:, i])
        else:
            aucs[name] = float('nan')
    valid = [v for v in aucs.values() if not np.isnan(v)]
    aucs['mean'] = np.mean(valid) if valid else 0.0
    return aucs


# ─────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, grad_clip):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc='  Train', leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device, enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


# ─────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, class_names):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in tqdm(loader, desc='  Eval ', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(device_type=device, enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    aucs = compute_auc(all_labels, all_probs, class_names)
    return total_loss / len(loader), aucs


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    # CLI overrides
    if args.backbone: cfg['model']['backbone']         = args.backbone
    if args.epochs:   cfg['training']['epochs']        = args.epochs
    if args.lr:       cfg['training']['learning_rate'] = args.lr
    if args.batch:    cfg['training']['batch_size']    = args.batch
    if args.loss:     cfg['training']['loss']          = args.loss
    if args.data_dir: cfg['data']['data_dir']          = args.data_dir

    # ── Device ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nDevice : {device}')
    if device == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    # ── Data ──
    loaders = get_dataloaders(
        data_dir      = cfg['data']['data_dir'],
        image_size    = cfg['data']['image_size'],
        batch_size    = cfg['training']['batch_size'],
        num_workers   = cfg['data']['num_workers'],
        images_subdir = cfg['data']['images_subdir'],
        labels_file   = cfg['data']['labels_file'],
        train_val_list= cfg['data']['train_val_list'],
        test_list     = cfg['data']['test_list'],
    )
    class_names = loaders['class_names']

    # ── Model ──
    model = build_model(
        backbone        = cfg['model']['backbone'],
        pretrained      = cfg['model']['pretrained'],
        dropout         = cfg['model']['dropout'],
        freeze_backbone = cfg['model']['freeze_backbone'],
        device          = device,
    )

    # ── Loss ──
    criterion = build_loss(cfg['training']['loss'], loaders['pos_weights'].to(device))

    # ── Optimiser ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg['training']['learning_rate'],
        weight_decay = cfg['training']['weight_decay'],
    )

    # ── Scheduler ──
    epochs = cfg['training']['epochs']
    if cfg['training']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif cfg['training']['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    # FIX L1-003: GradScaler must use dynamic device, not hardcoded 'cuda'.
    # Also disable AMP if not on CUDA to prevent ValueError on CPU machines.
    use_amp = cfg['training']['amp'] and device == 'cuda'
    scaler  = GradScaler(device, enabled=use_amp)

    save_dir  = Path(cfg['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    best_auc       = 0.0
    patience_count = 0
    patience       = cfg['training']['early_stopping_patience']

    print(f'\nStarting training — {epochs} epochs\n')
    print(f'{"Epoch":>6} {"Train Loss":>12} {"Val Loss":>10} {"Val AUC":>10} {"Best AUC":>10}')
    print('-' * 55)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, loaders['train'], optimizer, criterion,
            scaler, device, use_amp, cfg['training']['grad_clip']
        )
        val_loss, val_aucs = evaluate(
            model, loaders['val'], criterion, device,
            use_amp, class_names
        )

        if scheduler:
            scheduler.step()

        mean_auc = val_aucs['mean']
        elapsed  = time.time() - t0
        print(f'{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {mean_auc:>10.4f} {best_auc:>10.4f}  [{elapsed:.0f}s]')

        # ── Save best checkpoint ──
        if mean_auc > best_auc:
            best_auc = mean_auc
            patience_count = 0
            ckpt_path = save_dir / f'{cfg["model"]["backbone"]}_best.pth'
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc':          mean_auc,
                'val_aucs':         val_aucs,
                'config':           cfg,
            }, ckpt_path)
            print(f'         ✓ Saved checkpoint → {ckpt_path}')
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f'\nEarly stopping at epoch {epoch} (patience={patience})')
                break

    # ── Final test evaluation ──
    print('\nRunning test set evaluation...')
    test_loss, test_aucs = evaluate(
        model, loaders['test'], criterion, device,
        use_amp, class_names
    )
    print(f'\nTest Loss : {test_loss:.4f}')
    print(f'Test AUC  : {test_aucs["mean"]:.4f}')
    print('\nPer-class AUC:')
    for name, auc in test_aucs.items():
        if name != 'mean':
            print(f'  {name:<25} {auc:.4f}')

    print('\nTraining complete!')


if __name__ == '__main__':
    main()
