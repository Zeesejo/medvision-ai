"""
SimCLR self-supervised pretraining on NIH Chest X-Ray or MedMNIST.

Usage:
    python experiments/ssl/pretrain_simclr.py --config experiments/configs/ssl_simclr.yaml
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.encoder import MedEncoder
from models.ssl_heads import SimCLRProjectionHead, NTXentLoss


class SimCLRDataTransform:
    """Produces two augmented views of the same image for contrastive learning."""

    def __init__(self, size: int = 224) -> None:
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size) | 1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SimCLRModel(nn.Module):
    def __init__(self, arch: str = "resnet50", proj_dim: int = 128) -> None:
        super().__init__()
        self.encoder = MedEncoder(arch=arch, pretrained=False)
        self.projector = SimCLRProjectionHead(self.encoder.feat_dim, out_dim=proj_dim)

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2


def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project="medvision-ssl", config=config, mode=config.get("wandb_mode", "online"))

    # TODO: Replace with NIHChestXray dataset when data is downloaded
    # from data.datasets import NIHChestXray
    # dataset = NIHChestXray(root=config["data_root"], split="train", transform=SimCLRDataTransform())

    # Placeholder: use MedMNIST ChestMNIST for quick testing
    try:
        import medmnist
        from medmnist import ChestMNIST
        dataset = ChestMNIST(split="train", download=True, size=64,
                             transform=SimCLRDataTransform(size=64))
    except ImportError:
        raise ImportError("pip install medmnist")

    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    model = SimCLRModel(arch=config.get("arch", "resnet50")).to(device)
    criterion = NTXentLoss(temperature=config.get("temperature", 0.07))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 3e-4),
                                  weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("epochs", 100))

    save_dir = Path(config.get("save_dir", "results/checkpoints/simclr"))
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.get("epochs", 100) + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            (x1, x2), _ = batch
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1, x2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(loader)
        wandb.log({"train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0]}, step=epoch)
        print(f"Epoch [{epoch}/{config['epochs']}]  Loss: {avg_loss:.4f}")

        if epoch % config.get("save_every", 10) == 0:
            ckpt_path = save_dir / f"simclr_epoch{epoch:04d}.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "loss": avg_loss}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    default_config = {
        "arch": "resnet50",
        "batch_size": 256,
        "epochs": 100,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "temperature": 0.07,
        "num_workers": 4,
        "save_every": 10,
        "save_dir": "results/checkpoints/simclr",
        "wandb_mode": "online",
    }
    train(default_config)
