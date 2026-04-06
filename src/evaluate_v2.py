"""
evaluate_v2.py  —  Full evaluation with per-class AUC + Grad-CAM export
=======================================================================
Usage:
  python src/evaluate_v2.py --checkpoint results/checkpoints/densenet121_best.pth

Outputs:
  results/eval/per_class_auc.csv
  results/eval/gradcam/  (one heatmap PNG per class per sample)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import ChestXrayDataset

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
IMG_SIZE = 320


def build_model(num_classes, checkpoint_path, device):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, AUC {ckpt['auc']:.4f})")
    return model


def get_gradcam(model, img_tensor, class_idx):
    """Compute Grad-CAM for a single image and class."""
    gradients, activations = [], []

    def fwd_hook(m, i, o):
        activations.append(o.detach())

    def bwd_hook(m, gi, go):
        gradients.append(go[0].detach())

    # Hook onto last DenseNet denseblock
    handle_fwd = model.features.denseblock4.register_forward_hook(fwd_hook)
    handle_bwd = model.features.denseblock4.register_full_backward_hook(bwd_hook)

    img_tensor = img_tensor.unsqueeze(0).requires_grad_(True)
    output     = model(img_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grads = gradients[0]          # (1, C, H, W)
    acts  = activations[0]        # (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def save_gradcam_overlay(original_img_path, cam, save_path, alpha=0.45):
    img  = cv2.imread(original_img_path)
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heat = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heat = np.uint8(255 * heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    cv2.imwrite(save_path, overlay)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(len(CLASSES), args.checkpoint, device)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds     = ChestXrayDataset(args.test_csv, args.img_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbs in tqdm(loader, desc="Evaluating"):
            logits = model(imgs.to(device))
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(lbs.numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # ── per-class AUC
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    print("\nPer-class AUC:")
    print("-" * 40)
    for i, cls in enumerate(CLASSES):
        if labels[:, i].sum() > 0:
            auc = roc_auc_score(labels[:, i], preds[:, i])
        else:
            auc = float("nan")
        print(f"  {cls:<22} {auc:.4f}")
        rows.append({"class": cls, "auc": auc})

    mean_auc = np.nanmean([r["auc"] for r in rows])
    print("-" * 40)
    print(f"  {'Mean AUC':<22} {mean_auc:.4f}")

    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"class": "MEAN", "auc": mean_auc}
    csv_path = os.path.join(args.output_dir, "per_class_auc.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # ── Grad-CAM for top-N samples per class
    if args.gradcam:
        print("\nGenerating Grad-CAM heatmaps...")
        gradcam_dir = os.path.join(args.output_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)

        df_test = pd.read_csv(args.test_csv)
        img_paths = df_test["Image Index"].tolist()

        for cls_idx, cls_name in enumerate(CLASSES):
            pos_indices = np.where(labels[:, cls_idx] == 1)[0][:args.gradcam_n]
            cls_dir = os.path.join(gradcam_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            for idx in pos_indices:
                img_path = os.path.join(args.img_dir, img_paths[idx])
                img_tensor, _ = ds[idx]
                img_tensor = img_tensor.to(device)
                cam = get_gradcam(model, img_tensor, cls_idx)
                save_path = os.path.join(cls_dir, f"sample_{idx}.png")
                save_gradcam_overlay(img_path, cam, save_path)
        print(f"Grad-CAM heatmaps saved → {gradcam_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="results/checkpoints/densenet121_best.pth")
    parser.add_argument("--test_csv",    default="data/test.csv")
    parser.add_argument("--img_dir",     default="data/images")
    parser.add_argument("--output_dir",  default="results/eval")
    parser.add_argument("--gradcam",     action="store_true", help="Generate Grad-CAM overlays")
    parser.add_argument("--gradcam_n",   type=int, default=5, help="Samples per class for Grad-CAM")
    args = parser.parse_args()
    evaluate(args)
