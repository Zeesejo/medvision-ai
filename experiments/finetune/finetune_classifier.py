"""
Fine-tune a pretrained encoder for multi-label chest X-ray classification.
Supports linear probing and full fine-tuning modes.

Usage:
    python experiments/finetune/finetune_classifier.py \
        --checkpoint results/checkpoints/simclr/simclr_epoch0100.pt \
        --mode linear_probe \
        --label_fraction 0.1
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.encoder import MedEncoder, MultiLabelClassifier

NUM_CLASSES = 14  # NIH Chest X-Ray14


def get_transforms(split: str, size: int = 224):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.sigmoid(model(x)).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    # Compute mean AUC across classes (skip classes with no positive samples)
    aucs = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
    return float(np.mean(aucs))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="medvision-finetune", config=vars(args))

    encoder = MedEncoder(arch=args.arch, pretrained=False)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # Load only encoder weights from SimCLR checkpoint
        state = {k.replace("encoder.", ""): v for k, v in ckpt["model_state_dict"].items()
                 if k.startswith("encoder.")}
        encoder.backbone.load_state_dict(state, strict=False)
        print(f"Loaded encoder from {args.checkpoint}")

    if args.mode == "linear_probe":
        for p in encoder.parameters():
            p.requires_grad = False

    model = MultiLabelClassifier(encoder, num_classes=NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TODO: Replace with NIHChestXray dataset
    # from data.datasets import NIHChestXray
    # train_ds = NIHChestXray(root=args.data_root, split="train", transform=get_transforms("train"))

    print("INFO: Using MedMNIST ChestMNIST as placeholder. Replace with NIH Chest X-Ray when available.")

    for epoch in range(1, args.epochs + 1):
        # Placeholder training loop
        print(f"Epoch {epoch}/{args.epochs} — (plug in real dataset to train)")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="full", choices=["linear_probe", "full"])
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--data_root", default="data/raw/nih_chestxray")
    args = parser.parse_args()
    main(args)
