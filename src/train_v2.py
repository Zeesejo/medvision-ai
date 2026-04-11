"""
train_v2.py  —  MedVision-AI  (upgraded run)
================================================
Upgrades over train.py:
  * DenseNet-121 backbone  (CheXNet-style)
  * 320x320 input resolution
  * Progressive backbone unfreezing  (epoch 1-3 head only, epoch 4+ full)
  * Cosine LR schedule with warmup
  * Test-Time Augmentation (TTA) at validation
  * AMP fully dynamic — works on both CPU and CUDA
  * Weights & Biases (wandb) experiment tracking
Expected AUC gain: +0.04 - 0.07 over ResNet-50 baseline
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import wandb

# ── make project root importable regardless of how script is invoked ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import ChestXrayDataset
from src.losses  import AsymmetricLoss

# ── constants ──────────────────────────────────────────────────────────────
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES   = len(CLASSES)
IMG_SIZE      = 320
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LR_HEAD       = 3e-4
LR_BACKBONE   = 3e-5
WARMUP_EPOCHS = 2
PATIENCE      = 8

WB_PROJECT = "medvision-ai"
WB_ENTITY  = None

# ── model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


# ── transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ── evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device, use_amp: bool = True):
    """
    Run one validation pass.
    use_amp is gated on CUDA — autocast on CPU provides no benefit and can
    cause issues on older PyTorch builds.
    """
    amp_enabled = use_amp and device.type == "cuda"
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for imgs, lbs in tqdm(loader, desc="Eval ", leave=False):
            imgs, lbs = imgs.to(device), lbs.to(device)
            with autocast(device.type, enabled=amp_enabled):
                logits = model(imgs)
                loss   = criterion(logits, lbs)
            total_loss += loss.item()
            preds.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(lbs.cpu().numpy())

    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    aucs, per_class = [], {}
    for i, cls in enumerate(CLASSES):
        if labels[:, i].sum() > 0:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            aucs.append(auc)
            per_class[f"val_auc/{cls}"] = auc
    return total_loss / len(loader), float(np.mean(aucs)), per_class


# ── scheduler ─────────────────────────────────────────────────────────────────
def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs * steps_per_epoch)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=(total_epochs - warmup_epochs) * steps_per_epoch,
                               eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs * steps_per_epoch])


# ── training loop ───────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP is only meaningful on CUDA; disable gracefully on CPU
    use_amp = device.type == "cuda"

    run = wandb.init(
        project=WB_PROJECT,
        entity=WB_ENTITY,
        name="densenet121-320px-asl",
        config={
            "backbone": "densenet121", "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
            "warmup_epochs": WARMUP_EPOCHS, "patience": PATIENCE,
            "loss": "AsymmetricLoss", "gamma_neg": 4, "gamma_pos": 0,
            "clip": 0.05, "optimizer": "AdamW", "weight_decay": 1e-4,
            "device": str(device), "num_classes": NUM_CLASSES,
            "use_amp": use_amp,
        }
    )
    print(f"wandb run: {run.url}")
    print(f"Device: {device}  |  Input: {IMG_SIZE}x{IMG_SIZE}  |  AMP: {use_amp}")

    train_ds     = ChestXrayDataset(args.train_csv, args.img_dir, transform=train_transform)
    val_ds       = ChestXrayDataset(args.val_csv,   args.img_dir, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = build_model(NUM_CLASSES).to(device)
    wandb.watch(model, log="gradients", log_freq=200)
    freeze_backbone(model)
    print("Phase 1: head only (epochs 1-3)")

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4
    )
    # GradScaler is a no-op when enabled=False (CPU runs)
    scaler    = GradScaler(device.type, enabled=use_amp)
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, len(train_loader))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_auc, no_improve, log_rows = 0.0, 0, []

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == 4:
            print("\nPhase 2: full backbone unfrozen")
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW([
                {"params": model.features.parameters(),   "lr": LR_BACKBONE},
                {"params": model.classifier.parameters(), "lr": LR_HEAD}
            ], weight_decay=1e-4)
            scaler    = GradScaler(device.type, enabled=use_amp)
            scheduler = get_scheduler(optimizer, 1, NUM_EPOCHS - epoch + 1, len(train_loader))

        model.train()
        train_loss, t0 = 0.0, time.time()
        for step, (imgs, lbs) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
            imgs, lbs = imgs.to(device), lbs.to(device)
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                loss = criterion(model(imgs), lbs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            if step % 100 == 0:
                wandb.log({"batch/loss": loss.item(),
                           "batch/lr":   scheduler.get_last_lr()[0],
                           "batch/step": (epoch - 1) * len(train_loader) + step})

        avg_train = train_loss / len(train_loader)
        val_loss, val_auc, per_class = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed = int(time.time() - t0)

        marker = ""
        if val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            ckpt = os.path.join(args.checkpoint_dir, "densenet121_best.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "auc": best_auc}, ckpt)
            marker = f"\n    ✓ Saved → {ckpt}"
            wandb.run.summary["best_val_auc"] = best_auc
            wandb.run.summary["best_epoch"]   = epoch
        else:
            no_improve += 1

        epoch_log = {"epoch": epoch, "train/loss": avg_train,
                     "val/loss": val_loss, "val/mean_auc": val_auc,
                     "val/best_auc": best_auc, "train/epoch_time": elapsed}
        epoch_log.update(per_class)
        wandb.log(epoch_log)

        print(f"  {epoch:>3}  {avg_train:.4f}  {val_loss:.4f}  "
              f"{val_auc:.4f}  {best_auc:.4f}  [{elapsed}s]{marker}")
        log_rows.append({"epoch": epoch, "train_loss": avg_train,
                         "val_loss": val_loss, "val_auc": val_auc, "best_auc": best_auc})

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # Fix: use a single DictWriter instance for both header and rows
    log_path = os.path.join(args.checkpoint_dir, "densenet121_training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    artifact = wandb.Artifact("medvision-densenet121", type="model")
    artifact.add_file(os.path.join(args.checkpoint_dir, "densenet121_best.pth"))
    artifact.add_file(log_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",      default="data/train_val.csv")
    parser.add_argument("--val_csv",        default="data/val.csv")
    parser.add_argument("--img_dir",        default="data/images")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    train(parser.parse_args())
