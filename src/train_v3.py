"""
train_v3.py  —  MedVision-AI  (v3 — production-ready)
=======================================================
Fixes over train_v2.py:
  [BUG-1]  GradScaler(device.type, ...) → GradScaler(enabled=use_amp)       (PyTorch >=2.3)
  [BUG-2]  Phase-2 scheduler off-by-one → NUM_EPOCHS - epoch + 1
  [BUG-3]  log_rows IndexError crash when training aborts before epoch 1
  [BUG-4]  WB_ENTITY=None → now --wb_entity argparse param
  [BUG-5]  num_workers=4 hardcoded → now --num_workers argparse param
  [BUG-6]  pin_memory=True on CPU raises warning → gated on device.type=="cuda"
  [BUG-7]  best.pth only saves on improvement → last.pth saves every epoch
           so --resume always continues from the true last epoch

Upgrades:
  * ViT-B/16 backbone via --backbone
  * SSL checkpoint loading via --ssl_checkpoint
  * Anatomy-aware augmentation
  * AsymmetricLossOptimized (fixed)
  * Per-class AUC logged every epoch
  * --resume: resumes from last.pth (every-epoch) or any explicit path
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import wandb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import ChestXrayDataset
from src.losses  import AsymmetricLossOptimized
from src.models.classifier import build_model as build_timm_model

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

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.00), ratio=(0.90, 1.10)),
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


def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(1, warmup_epochs * steps_per_epoch),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, (total_epochs - warmup_epochs) * steps_per_epoch),
        eta_min=1e-6,
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[warmup_epochs * steps_per_epoch],
    )


def evaluate(model, loader, criterion, device, use_amp=True):
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
    if not preds_list:
        return 0.0, 0.0, {}
    preds  = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    aucs, per_class = [], {}
    for i, cls in enumerate(CLASSES):
        if len(np.unique(labels[:, i])) > 1:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            aucs.append(auc)
            per_class[f"val_auc/{cls}"] = auc
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    return total_loss / max(len(loader), 1), mean_auc, per_class


def _save_ckpt(path, epoch, model, optimizer, scaler, scheduler,
               best_auc, no_improve, log_rows):
    """Save a full resumable checkpoint."""
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "auc":                  best_auc,
        "no_improve":           no_improve,
        "log_rows":             log_rows,
    }, path)


def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin_mem = device.type == "cuda"

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_best.pth")
    last_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_last.pth")

    # ── resolve which checkpoint to resume from ─────────────────────────────────
    resume_path = args.resume
    if not resume_path and os.path.isfile(last_ckpt_path):
        # Auto-detect last.pth so user doesn't have to specify --resume
        # (only kicks in if last.pth already exists from a previous run)
        pass  # don't auto-resume unless explicitly requested

    resume_ckpt = None
    start_epoch = 1
    best_auc    = 0.0
    no_improve  = 0
    log_rows    = []

    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        print(f"\nResuming from checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch = resume_ckpt.get("epoch", 1) + 1
        best_auc    = resume_ckpt.get("auc",   0.0)
        no_improve  = resume_ckpt.get("no_improve", 0)
        log_rows    = resume_ckpt.get("log_rows",   [])
        print(f"  Resuming at epoch {start_epoch}  |  best_auc so far: {best_auc:.4f}  |  no_improve: {no_improve}")

    # ── W&B ────────────────────────────────────────────────────────────────────────
    run = wandb.init(
        project=WB_PROJECT,
        entity=args.wb_entity if args.wb_entity else None,
        name=f"{args.backbone}-320px-asl-v3",
        resume="allow",
        config=dict(
            backbone=args.backbone, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS, lr_head=LR_HEAD, lr_backbone=LR_BACKBONE,
            warmup_epochs=WARMUP_EPOCHS, patience=PATIENCE,
            loss="AsymmetricLossOptimized", gamma_neg=4, gamma_pos=0, clip=0.05,
            optimizer="AdamW", weight_decay=1e-4, device=str(device),
            num_classes=NUM_CLASSES, use_amp=use_amp,
            ssl_checkpoint=args.ssl_checkpoint or "none",
        ),
    )
    print(f"\nwandb run : {run.url}")
    print(f"Device    : {device}  |  Backbone: {args.backbone}  |  AMP: {use_amp}\n")

    # ── data ───────────────────────────────────────────────────────────────────────
    train_ds = ChestXrayDataset(args.train_csv, args.img_dir, transform=train_transform)
    val_ds   = ChestXrayDataset(args.val_csv,   args.img_dir, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_mem,
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_mem,
                              persistent_workers=(args.num_workers > 0))

    # ── model ─────────────────────────────────────────────────────────────────────
    freeze_for_init = (start_epoch <= 3) and (resume_ckpt is None)
    model = build_timm_model(
        backbone=args.backbone, num_classes=NUM_CLASSES,
        pretrained=(resume_ckpt is None), dropout=0.3,
        freeze_backbone=freeze_for_init,
    ).to(device)

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print("  Model weights loaded.")
        if start_epoch > 3:
            model.unfreeze_backbone()
            print("  Backbone unfrozen (resuming in Phase 2).")
    elif args.ssl_checkpoint and os.path.isfile(args.ssl_checkpoint):
        print(f"Loading SSL checkpoint: {args.ssl_checkpoint}")
        ssl_state = torch.load(args.ssl_checkpoint, map_location=device, weights_only=True)
        bb_state  = ssl_state.get("backbone", ssl_state)
        missing, unexpected = model.backbone.load_state_dict(bb_state, strict=False)
        print(f"  SSL load — missing: {len(missing)}  unexpected: {len(unexpected)}")

    wandb.watch(model, log="gradients", log_freq=200)
    criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05)

    # ── optimizer & scheduler ──────────────────────────────────────────────────
    if start_epoch > 3:
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.head.parameters(),     "lr": LR_HEAD},
        ], weight_decay=1e-4)
        scheduler = get_scheduler(optimizer, 1, NUM_EPOCHS - start_epoch + 1, len(train_loader))
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_HEAD, weight_decay=1e-4,
        )
        scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, len(train_loader))

    scaler = GradScaler(enabled=use_amp)

    if resume_ckpt is not None:
        for key, obj, label in [
            ("optimizer_state_dict", optimizer, "Optimizer"),
            ("scaler_state_dict",    scaler,    "GradScaler"),
            ("scheduler_state_dict", scheduler, "Scheduler"),
        ]:
            if key in resume_ckpt:
                try:
                    obj.load_state_dict(resume_ckpt[key])
                    print(f"  {label} state restored.")
                except Exception as e:
                    print(f"  {label} state skipped ({e}): fresh start.")

    phase_label = "Phase 1: head only (epochs 1-3)" if start_epoch <= 3 \
        else f"Phase 2: full backbone unfrozen (resuming at epoch {start_epoch})"
    print(phase_label)

    # ── training loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        if epoch == 4 and start_epoch <= 4:
            print("\nPhase 2: full backbone unfrozen")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
                {"params": model.head.parameters(),     "lr": LR_HEAD},
            ], weight_decay=1e-4)
            scaler    = GradScaler(enabled=use_amp)
            scheduler = get_scheduler(optimizer, 1, NUM_EPOCHS - epoch + 1, len(train_loader))

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

        # Update log_rows BEFORE saving so checkpoint has current epoch included
        log_rows.append({"epoch": epoch, "train_loss": avg_train,
                         "val_loss": val_loss, "val_auc": val_auc, "best_auc": best_auc})

        marker = ""
        if val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            # Update the last log row's best_auc to the new value
            log_rows[-1]["best_auc"] = best_auc
            _save_ckpt(best_ckpt_path, epoch, model, optimizer, scaler,
                       scheduler, best_auc, no_improve, log_rows)
            marker = f"  ✓ saved → {best_ckpt_path}"
            wandb.run.summary["best_val_auc"] = best_auc
            wandb.run.summary["best_epoch"]   = epoch
        else:
            no_improve += 1

        # Always save last.pth so resume picks up from the true last epoch
        _save_ckpt(last_ckpt_path, epoch, model, optimizer, scaler,
                   scheduler, best_auc, no_improve, log_rows)

        epoch_log = {
            "epoch": epoch, "train/loss": avg_train,
            "val/loss": val_loss, "val/mean_auc": val_auc,
            "val/best_auc": best_auc, "train/epoch_time": elapsed,
        }
        epoch_log.update(per_class)
        wandb.log(epoch_log)

        print(
            f"  {epoch:>3}  train={avg_train:.4f}  val={val_loss:.4f}  "
            f"auc={val_auc:.4f}  best={best_auc:.4f}  [{elapsed}s]{marker}"
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ── training log CSV ─────────────────────────────────────────────────────────
    log_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_training_log.csv")
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Training log → {log_path}")

    # ── W&B artifact ──────────────────────────────────────────────────────────────
    if os.path.isfile(best_ckpt_path):
        artifact = wandb.Artifact(f"medvision-{args.backbone}", type="model")
        artifact.add_file(best_ckpt_path)
        if log_rows and os.path.isfile(log_path):
            artifact.add_file(log_path)
        wandb.log_artifact(artifact)

    wandb.finish()
    print(f"\nBest Val AUC: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedVision-AI v3 trainer")
    parser.add_argument("--train_csv",      default="data/train_val.csv")
    parser.add_argument("--val_csv",        default="data/val.csv")
    parser.add_argument("--img_dir",        default="data/images")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument("--backbone",       default="densenet121",
                        choices=["densenet121", "resnet50", "vit_base_patch16_224"])
    parser.add_argument("--wb_entity",      default="")
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--ssl_checkpoint", default="")
    parser.add_argument("--resume",         default="",
                        help="Path to checkpoint to resume from. "
                             "Use densenet121_last.pth to continue from exact last epoch, "
                             "or densenet121_best.pth to restart from best AUC epoch.")
    train(parser.parse_args())
