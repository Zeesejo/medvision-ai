"""Training entry point for MedVision-AI.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/vit_experiment.yaml
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.data.chestxray import get_dataloaders
from src.models.classifier import MedVisionClassifier
from src.utils.metrics import compute_metrics
from src.utils.logger import setup_logger

logger = setup_logger("medvision.train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    name    = cfg["training"]["scheduler"]
    epochs  = cfg["training"]["epochs"]
    warmup  = cfg["training"].get("warmup_epochs", 0)

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=1e-6
        )
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    raise ValueError(f"Unknown scheduler: {name}")


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, epoch, cfg
) -> float:
    model.train()
    total_loss = 0.0
    log_interval = cfg["logging"]["log_interval"]

    for step, (images, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if cfg["training"]["mixed_precision"]:
            with autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            logger.info(f"  step {step+1}/{len(loader)} | loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model, loader, criterion, device
):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_targets = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.cpu().numpy())

    all_probs   = np.concatenate(all_probs,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics     = compute_metrics(all_targets, all_probs)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MedVision-AI")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # W&B
    if cfg["logging"]["use_wandb"] and WANDB_AVAILABLE:
        wandb.init(project=cfg["logging"]["project"], config=cfg, name=cfg["model"]["backbone"])

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path    = cfg["data"]["csv_path"],
        image_dir   = str(Path(cfg["data"]["root"]) / "images"),
        image_size  = cfg["data"]["image_size"],
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        train_split = cfg["data"]["train_split"],
        val_split   = cfg["data"]["val_split"],
        seed        = cfg["project"]["seed"],
    )

    # Model
    model = MedVisionClassifier(
        backbone    = cfg["model"]["backbone"],
        num_classes = cfg["model"]["num_classes"],
        pretrained  = cfg["model"]["pretrained"],
        dropout     = cfg["model"]["dropout"],
    ).to(device)

    # Loss — pos_weight to handle class imbalance (NIH dataset is heavily imbalanced)
    # Approximate positive frequency per class: ~5-10%, so weight ~9-19
    pos_weight = torch.ones(cfg["model"]["num_classes"]) * 10.0
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler(enabled=cfg["training"]["mixed_precision"])

    # Resume
    start_epoch  = 1
    best_auc     = 0.0
    patience_cnt = 0
    save_dir     = Path(cfg["logging"]["save_dir"]) / cfg["model"]["backbone"]
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']} (best_auc={best_auc:.4f})")

    # Training loop
    patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, cfg)
        val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"| val_auc={val_metrics['mean_auc']:.4f} "
            f"| val_f1={val_metrics['macro_f1']:.4f}"
        )

        # Scheduler step
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["mean_auc"])
        else:
            scheduler.step()

        # W&B logging
        if cfg["logging"]["use_wandb"] and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "train_loss": train_loss, **{f"val/{k}": v for k, v in val_metrics.items()}})

        # Checkpoint
        ckpt = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_auc":        best_auc,
            "config":          cfg,
        }
        torch.save(ckpt, save_dir / "last.pth")

        if val_metrics["mean_auc"] > best_auc:
            best_auc     = val_metrics["mean_auc"]
            patience_cnt = 0
            torch.save(ckpt, save_dir / "best.pth")
            logger.info(f"  *** New best AUC: {best_auc:.4f} — checkpoint saved ***")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Final test evaluation
    logger.info("\n--- Final Test Evaluation ---")
    best_ckpt = torch.load(save_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test results: {test_metrics}")

    if cfg["logging"]["use_wandb"] and WANDB_AVAILABLE:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
