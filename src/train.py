"""Canonical MedVision-AI training and evaluation entry point.

Example:
    python src/train.py --config configs/baseline_densenet121_asl.yaml

Each run writes a self-contained directory with the resolved config, environment
snapshot, training history, best checkpoint, test metrics, and raw predictions.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import CLASS_NAMES
from src.dataset import ChestXrayDataset
from src.losses import get_loss
from src.models.classifier import build_model
from src.reproducibility import environment_snapshot, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reproducible MedVision-AI baseline")
    parser.add_argument(
        "--config",
        default="configs/baseline_densenet121_asl.yaml",
        help="Experiment YAML configuration",
    )
    parser.add_argument("--run_name", default=None, help="Optional output-directory override")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_transforms(image_size: int) -> tuple[T.Compose, T.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = T.Compose(
        [
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transform


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def build_loaders(cfg: dict, seed: int) -> dict[str, DataLoader]:
    data_cfg = cfg["data"]
    train_transform, eval_transform = build_transforms(int(data_cfg["image_size"]))

    datasets = {
        "train": ChestXrayDataset(data_cfg["train_csv"], data_cfg["img_dir"], transform=train_transform),
        "val": ChestXrayDataset(data_cfg["val_csv"], data_cfg["img_dir"], transform=eval_transform),
        "test": ChestXrayDataset(data_cfg["test_csv"], data_cfg["img_dir"], transform=eval_transform),
    }

    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = torch.cuda.is_available()

    return {
        split: DataLoader(
            dataset,
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            worker_init_fn=seed_worker,
            generator=generator,
        )
        for split, dataset in datasets.items()
    }


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    per_class = {}
    auc_values = []
    ap_values = []

    for index, class_name in enumerate(CLASS_NAMES):
        y_true = labels[:, index]
        y_prob = probs[:, index]
        if len(np.unique(y_true)) < 2:
            auc = None
            ap = None
        else:
            auc = float(roc_auc_score(y_true, y_prob))
            ap = float(average_precision_score(y_true, y_prob))
            auc_values.append(auc)
            ap_values.append(ap)

        per_class[class_name] = {"auroc": auc, "auprc": ap}

    return {
        "macro_auroc": float(np.mean(auc_values)) if auc_values else None,
        "macro_auprc": float(np.mean(ap_values)) if ap_values else None,
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    model.eval()
    total_loss = 0.0
    labels_list = []
    probs_list = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += float(loss.item())
        labels_list.append(labels.cpu().numpy())
        probs_list.append(torch.sigmoid(logits).cpu().numpy())

    labels_np = np.concatenate(labels_list, axis=0)
    probs_np = np.concatenate(probs_list, axis=0)
    metrics = compute_metrics(labels_np, probs_np)
    return total_loss / max(len(loader), 1), labels_np, probs_np, metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def save_predictions(path: Path, labels: np.ndarray, probs: np.ndarray) -> None:
    data = {}
    for index, class_name in enumerate(CLASS_NAMES):
        data[f"label_{class_name}"] = labels[:, index].astype(np.int8)
        data[f"prob_{class_name}"] = probs[:, index]
    pd.DataFrame(data).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("project", {}).get("seed", 42))
    deterministic = bool(cfg.get("project", {}).get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"

    experiment_id = cfg.get("project", {}).get("experiment_id", "experiment")
    run_name = args.run_name or f"{experiment_id}_seed{seed}"
    output_dir = Path(cfg["logging"].get("output_root", "results/runs")) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(args.config, output_dir / "config.yaml")
    write_json(output_dir / "environment.json", environment_snapshot())
    write_json(
        output_dir / "run_manifest.json",
        {
            "experiment_id": experiment_id,
            "run_name": run_name,
            "seed": seed,
            "deterministic": deterministic,
            "device": str(device),
            "config_source": str(Path(args.config).resolve()),
        },
    )

    loaders = build_loaders(cfg, seed)
    model = build_model(
        backbone=cfg["model"]["backbone"],
        pretrained=bool(cfg["model"].get("pretrained", True)),
        dropout=float(cfg["model"].get("dropout", 0.3)),
        freeze_backbone=bool(cfg["model"].get("freeze_backbone", False)),
        device=str(device),
    )
    criterion = get_loss(cfg)

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    epochs = int(cfg["training"]["epochs"])
    scheduler_name = cfg["training"].get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name in {None, "none"}:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    scaler = GradScaler(device=device.type, enabled=use_amp)
    patience = int(cfg["training"].get("early_stopping_patience", epochs))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    best_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    best_checkpoint = output_dir / "best.pth"

    print(f"Device      : {device}")
    print(f"Run         : {run_name}")
    print(f"Backbone    : {cfg['model']['backbone']}")
    print(f"Seed        : {seed}")
    print(f"Output      : {output_dir}\n")

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            scaler,
            device,
            use_amp,
            grad_clip,
        )
        val_loss, _, _, val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            use_amp,
        )
        if scheduler is not None:
            scheduler.step()

        val_auc = val_metrics["macro_auroc"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auroc": val_auc,
            "val_macro_auprc": val_metrics["macro_auprc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - started,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"AUROC={val_auc:.4f} | AUPRC={val_metrics['macro_auprc']:.4f}"
        )

        if val_auc is not None and val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "seed": seed,
                    "val_metrics": val_metrics,
                },
                best_checkpoint,
            )
            write_json(output_dir / "best_validation_metrics.json", val_metrics)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_labels, test_probs, test_metrics = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        use_amp,
    )
    test_metrics["loss"] = test_loss
    test_metrics["best_validation_epoch"] = best_epoch
    write_json(output_dir / "test_metrics.json", test_metrics)
    save_predictions(output_dir / "test_predictions.csv", test_labels, test_probs)

    print("\nFinal test results")
    print(f"Macro AUROC: {test_metrics['macro_auroc']:.4f}")
    print(f"Macro AUPRC: {test_metrics['macro_auprc']:.4f}")
    print(f"Artifacts  : {output_dir}")


if __name__ == "__main__":
    main()
