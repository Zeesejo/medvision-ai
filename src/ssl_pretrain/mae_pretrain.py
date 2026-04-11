"""
mae_pretrain.py  —  Lightweight Masked Autoencoder (MAE) pretraining
=====================================================================
Novel contribution: Anatomy-Guided Masked Autoencoding for Chest X-Ray.

Core idea:
  Standard MAE (He et al., 2022) masks random 75% of patches and trains
  the encoder to reconstruct them.  Our extension adds two novelties:

  1. Anatomy-biased masking: lung-region patches (centre-bottom 50% of image)
     are masked with 2x higher probability than periphery, forcing the encoder
     to learn clinically-relevant features inside the lung fields.

  2. Frequency-domain reconstruction loss: MSE in pixel space + FFT magnitude
     loss to preserve high-frequency texture (nodules, fibrosis patterns).

Usage:
  python src/ssl_pretrain/mae_pretrain.py \
      --img_dir  data/images \
      --save_dir results/ssl \
      --epochs   100 \
      --batch_size 64
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import wandb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IMG_SIZE   = 224
PATCH_SIZE = 16
MASK_RATIO = 0.75


class UnlabeledCXRDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.transform = transform
        self.paths = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(root, f))
        print(f"[SSL] Found {len(self.paths):,} images for pretraining")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def anatomy_guided_mask(num_patches: int, img_size: int, patch_size: int,
                        mask_ratio: float = 0.75) -> torch.Tensor:
    """
    Returns a boolean mask (True = masked) for num_patches patches.
    Lung region (centre 60% of patches) has 2x masking probability.
    Novel contribution: biased masking toward lung fields.
    """
    n  = int(math.sqrt(num_patches))
    weights = torch.ones(num_patches)
    row_lo, row_hi = int(0.20 * n), int(0.80 * n)
    col_lo, col_hi = int(0.15 * n), int(0.85 * n)
    for r in range(n):
        for c in range(n):
            if row_lo <= r < row_hi and col_lo <= c < col_hi:
                weights[r * n + c] = 2.0
    num_masked = int(num_patches * mask_ratio)
    weights    = weights / weights.sum()
    indices    = torch.multinomial(weights, num_masked, replacement=False)
    mask       = torch.zeros(num_patches, dtype=torch.bool)
    mask[indices] = True
    return mask


class LightweightMAE(nn.Module):
    """
    Patch-based MAE using ResNet-50 encoder + lightweight MLP decoder.
    Copilot review fix: masking is now fully vectorized in patch space
    (no Python loops), keeping all ops on GPU.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim   = 3 * patch_size * patch_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, self.patch_dim * self.num_patches),
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, patch_dim)"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        imgs = imgs.reshape(B, C, H // p, p, W // p, p)
        imgs = imgs.permute(0, 2, 4, 1, 3, 5)
        return imgs.reshape(B, (H // p) * (W // p), C * p * p)

    def unpatchify(self, patches: torch.Tensor, B: int) -> torch.Tensor:
        """(B, N, patch_dim) -> (B, 3, H, W)"""
        p  = self.patch_size
        n  = self.img_size // p
        x  = patches.reshape(B, n, n, 3, p, p)
        x  = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, 3, self.img_size, self.img_size)

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor):
        """
        imgs : (B, 3, H, W)
        mask : (N,) bool  -- True = masked patch
        Returns (pred_patches, target_patches, mask)
        """
        B = imgs.size(0)
        target_patches = self.patchify(imgs)   # (B, N, patch_dim)

        # Copilot review fix: vectorized masking in patch space -- no Python loops.
        masked_patches = target_patches.clone()
        masked_patches[:, mask, :] = 0.0
        masked_imgs = self.unpatchify(masked_patches, B)

        feats = self.encoder(masked_imgs)
        feats = self.encoder_proj(feats)              # (B, 1024)
        pred  = self.decoder(feats)                   # (B, N*patch_dim)
        pred  = pred.reshape(B, self.num_patches, self.patch_dim)

        return pred, target_patches, mask


def freq_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                             mask: torch.Tensor, freq_weight: float = 0.1) -> torch.Tensor:
    """
    Novel frequency-domain reconstruction loss.
    L = MSE(pred, target) + freq_weight * |FFT(pred) - FFT(target)| on masked patches.
    """
    pred_m   = pred[:, mask]
    target_m = target[:, mask]
    mse = F.mse_loss(pred_m, target_m)
    pred_fft   = torch.fft.rfft(pred_m,   dim=-1)
    target_fft = torch.fft.rfft(target_m, dim=-1)
    freq_loss  = F.l1_loss(pred_fft.abs(), target_fft.abs())
    return mse + freq_weight * freq_loss


def pretrain(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin_mem = device.type == "cuda"

    wandb.init(
        project="medvision-ai",
        entity=args.wb_entity if args.wb_entity else None,
        name="mae-ssl-pretrain",
        config=vars(args),
    )

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds     = UnlabeledCXRDataset(args.img_dir, transform=tfm)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem,
        persistent_workers=(args.num_workers > 0),
    )

    mae   = LightweightMAE(IMG_SIZE, PATCH_SIZE).to(device)
    opt   = torch.optim.AdamW(mae.parameters(), lr=1.5e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

    for epoch in range(1, args.epochs + 1):
        mae.train()
        total_loss = 0.0
        for imgs in tqdm(loader, desc=f"SSL Ep {epoch:03d}", leave=False):
            imgs = imgs.to(device)
            mask = anatomy_guided_mask(num_patches, IMG_SIZE, PATCH_SIZE, MASK_RATIO).to(device)
            opt.zero_grad()
            with autocast(device.type, enabled=use_amp):
                pred, target, mask_ = mae(imgs, mask)
                loss = freq_reconstruction_loss(pred, target, mask_)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
        avg = total_loss / max(len(loader), 1)
        sched.step()
        wandb.log({"ssl/loss": avg, "ssl/epoch": epoch, "ssl/lr": sched.get_last_lr()[0]})
        print(f"  SSL Ep {epoch:3d}  loss={avg:.5f}")

        if epoch % 20 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"mae_pretrained_ep{epoch}.pth")
            torch.save({"epoch": epoch, "backbone": mae.encoder.state_dict()}, ckpt_path)
            print(f"  Saved SSL checkpoint -> {ckpt_path}")

    final_path = os.path.join(args.save_dir, "mae_pretrained.pth")
    torch.save({"epoch": args.epochs, "backbone": mae.encoder.state_dict()}, final_path)
    print(f"SSL pretraining complete. Backbone saved -> {final_path}")

    artifact = wandb.Artifact("medvision-ssl-backbone", type="model")
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedVision-AI MAE SSL Pretraining")
    parser.add_argument("--img_dir",    default="data/images")
    parser.add_argument("--save_dir",   default="results/ssl")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--wb_entity",  default="")
    pretrain(parser.parse_args())
