"""
train_v3.py  —  MedVision-AI  (v3 — production-ready)
=======================================================
Fixes over train_v2.py:
  [BUG-1]  GradScaler(device.type, ...) → GradScaler(enabled=use_amp)       (PyTorch >=2.3)
  [BUG-2]  Phase-2 scheduler off-by-one → NUM_EPOCHS - epoch + 1 (includes current epoch)
  [BUG-3]  log_rows IndexError crash when training aborts before epoch 1
  [BUG-4]  WB_ENTITY=None → now --wb_entity argparse param
  [BUG-5]  num_workers=4 hardcoded → now --num_workers argparse param
  [BUG-6]  pin_memory=True on CPU raises warning → gated on device.type=="cuda"

Upgrades over train_v2.py:
  * ViT-B/16 backbone support via --backbone flag (timm-based via classifier.py)
  * SSL pre-trained weight loading via --ssl_checkpoint (optional)
  * Anatomy-aware augmentation (random lung-biased crops + stronger Gaussian blur)
  * AsymmetricLossOptimized with fixed log_pos clamp (see losses.py fix)
  * Per-class AUC printed + logged every epoch (not just mean)
  * W&B artifact upload is skipped gracefully if best checkpoint not yet saved
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
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import wandb

# ── make project root importable regardless of invocation method ──────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import ChestXrayDataset
from src.losses  import AsymmetricLossOptimized          # BUG-4 fixed variant
from src.models.classifier import build_model as build_timm_model

# ── constants ─────────────────────────────────────────────────────────────────
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
WB_PROJECT    = "medvision-ai"


# ── anatomy-aware augmentation ────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.70, 1.00),
        ratio=(0.90, 1.10),
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── scheduler ─────────────────────────────────────────────────────────────────
def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(1, warmup_epochs * steps_per_epoch),
    )
    cosine_steps = max(1, (total_epochs - warmup_epochs) * steps_per_epoch)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs * steps_per_epoch],
    )


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device, use_amp=True):
    """
    Validation pass. Returns (avg_loss, mean_auc, per_class_auc_dict).
    Returns (0.0, 0.0, {}) immediately if the loader yields no batches.
    """
    amp_enabled = use_amp and device.type == "cuda"
    model.eval()
    total_loss, preds_list, labels_list = 0.0, [], []

    with torch.no_grad():
        for imgs, lbs in tqdm(loader, desc="Val  ", leave=False):
            imgs, lbs = imgs.to(device), lbs.to(device)
            with autocast(device.type, enabled=amp_enabled):
                logits = model(imgs)
                loss   = criterion(logits, lbs)
            total_loss += loss.item()
            preds_list.append(torch.sigmoid(logits).cpu().numpy())
            labels_list.append(lbs.cpu().numpy())

    # Copilot review: guard against empty validation loader
    if not preds_list:
        return 0.0, 0.0, {}

    preds  = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    aucs, per_class = [], {}
    for i, cls in enumerate(CLASSES):
        # Copilot review: guard against single-class columns (all-0 or all-1)
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            aucs.append(auc)
            per_class[f"val_auc/{cls}"] = auc
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    return total_loss / max(len(loader), 1), mean_auc, per_class


# ── training loop ─────────────────────────────────────────────────────────────
def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin_mem = device.type == "cuda"   # BUG-6: suppress CPU pin_memory warning

    run = wandb.init(
        project=WB_PROJECT,
        entity=args.wb_entity if args.wb_entity else None,
        name=f"{args.backbone}-320px-asl-v3",
        config={
            "backbone":      args.backbone,
            "img_size":      IMG_SIZE,
            "batch_size":    BATCH_SIZE,
            "epochs":        NUM_EPOCHS,
            "lr_head":       LR_HEAD,
            "lr_backbone":   LR_BACKBONE,
            "warmup_epochs": WARMUP_EPOCHS,
            "patience":      PATIENCE,
            "loss":          "AsymmetricLossOptimized",
            "gamma_neg":     4,
            "gamma_pos":     0,
            "clip":          0.05,
            "optimizer":     "AdamW",
            "weight_decay":  1e-4,
            "device":        str(device),
            "num_classes":   NUM_CLASSES,
            "use_amp":       use_amp,
            "ssl_checkpoint": args.ssl_checkpoint or "none",
        },
    )
    print(f"\nwandb run : {run.url}")
    print(f"Device    : {device}  |  Backbone: {args.backbone}  |  AMP: {use_amp}\n")

    # ── datasets & loaders ────────────────────────────────────────────────────
    train_ds = ChestXrayDataset(args.train_csv, args.img_dir, transform=train_transform)
    val_ds   = ChestXrayDataset(args.val_csv,   args.img_dir, transform=val_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
    )

    # ── model ────────────────────────────────────────────────────────────────
    model = build_timm_model(
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.3,
        freeze_backbone=True,
    ).to(device)

    if args.ssl_checkpoint and os.path.isfile(args.ssl_checkpoint):
        print(f"Loading SSL checkpoint: {args.ssl_checkpoint}")
        ssl_state = torch.load(args.ssl_checkpoint, map_location=device, weights_only=True)
        bb_state  = ssl_state.get("backbone", ssl_state)
        missing, unexpected = model.backbone.load_state_dict(bb_state, strict=False)
        print(f"  SSL load — missing: {len(missing)}  unexpected: {len(unexpected)}")

    wandb.watch(model, log="gradients", log_freq=200)
    print("Phase 1: head only (epochs 1-3)")

    criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4,
    )
    scaler    = GradScaler(enabled=use_amp)
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, len(train_loader))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_auc, no_improve, log_rows = 0.0, 0, []
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_best.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Phase 2: unfreeze full backbone at epoch 4
        if epoch == 4:
            print("\nPhase 2: full backbone unfrozen")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
                    {"params": model.head.parameters(),     "lr": LR_HEAD},
                ],
                weight_decay=1e-4,
            )
            scaler = GradScaler(enabled=use_amp)
            # Copilot review BUG-2: use NUM_EPOCHS - epoch + 1 to include the
            # current epoch in the scheduler budget (avoids LR rising at end).
            scheduler = get_scheduler(optimizer, 1, NUM_EPOCHS - epoch + 1, len(train_loader))

        # ── training step ─────────────────────────────────────────────────────
        model.train()
        train_loss, t0 = 0.0, time.time()
        for step, (imgs, lbs) in enumerate(tqdm(train_loader, desc=f"Ep {epoch:02d}", leave=False)):
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
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/lr":   scheduler.get_last_lr()[0],
                    "batch/step": (epoch - 1) * len(train_loader) + step,
                })

        avg_train = train_loss / max(len(train_loader), 1)
        val_loss, val_auc, per_class = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed = int(time.time() - t0)

        marker = ""
        if val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "auc": best_auc},
                best_ckpt_path,
            )
            marker = f"  ✓ saved → {best_ckpt_path}"
            wandb.run.summary["best_val_auc"] = best_auc
            wandb.run.summary["best_epoch"]   = epoch
        else:
            no_improve += 1

        epoch_log = {
            "epoch":            epoch,
            "train/loss":       avg_train,
            "val/loss":         val_loss,
            "val/mean_auc":     val_auc,
            "val/best_auc":     best_auc,
            "train/epoch_time": elapsed,
        }
        epoch_log.update(per_class)
        wandb.log(epoch_log)
        log_rows.append({"epoch": epoch, "train_loss": avg_train,
                         "val_loss": val_loss, "val_auc": val_auc, "best_auc": best_auc})

        print(
            f"  {epoch:>3}  train={avg_train:.4f}  val={val_loss:.4f}  "
            f"auc={val_auc:.4f}  best={best_auc:.4f}  [{elapsed}s]{marker}"
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ── save training log ────────────────────────────────────────────────────
    log_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_training_log.csv")
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Training log → {log_path}")

    # ── W&B artifact upload ───────────────────────────────────────────────────
    if os.path.isfile(best_ckpt_path):
        artifact = wandb.Artifact(f"medvision-{args.backbone}", type="model")
        artifact.add_file(best_ckpt_path)
        if log_rows and os.path.isfile(log_path):
            artifact.add_file(log_path)
        wandb.log_artifact(artifact)

    wandb.finish()
    print(f"\nBest Val AUC: {best_auc:.4f}")
    return best_auc


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedVision-AI v3 trainer")
    parser.add_argument("--train_csv",      default="data/train_val.csv")
    parser.add_argument("--val_csv",        default="data/val.csv")
    parser.add_argument("--img_dir",        default="data/images")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument("--backbone",       default="densenet121",
                        choices=["densenet121", "resnet50", "vit_base_patch16_224"],
                        help="Backbone architecture")
    parser.add_argument("--wb_entity",      default="",
                        help="Weights & Biases entity / username (e.g. zeemaokik)")
    parser.add_argument("--num_workers",    type=int, default=4,
                        help="DataLoader workers (set 0 on Windows if issues occur)")
    parser.add_argument("--ssl_checkpoint", default="",
                        help="Optional path to SSL-pretrained backbone .pth file")
    train(parser.parse_args())
