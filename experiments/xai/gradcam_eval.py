"""
Grad-CAM and GradCAM++ evaluation for a trained classifier.
Produces saliency overlays and computes localization metrics against bounding box annotations.

Usage:
    python experiments/xai/gradcam_eval.py \
        --checkpoint results/checkpoints/finetune/model_best.pt \
        --image_path data/sample/00000001_000.png
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.encoder import MedEncoder, MultiLabelClassifier

NUM_CLASSES = 14
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def load_model(checkpoint_path: str, arch: str = "resnet50", device="cpu"):
    encoder = MedEncoder(arch=arch, pretrained=False)
    model = MultiLabelClassifier(encoder, num_classes=NUM_CLASSES).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    model.eval()
    return model


def compute_gradcam(model, image_tensor: torch.Tensor, target_class: int):
    """
    Compute Grad-CAM for a given class index using pytorch-grad-cam library.
    Returns heatmap as numpy array (H, W) normalized to [0, 1].
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise ImportError("pip install grad-cam")

    # Target the last conv layer in the encoder backbone
    target_layers = [model.encoder.backbone.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    heatmap = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)[0]
    return heatmap  # (H, W) float32


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, arch=args.arch, device=device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image_path).convert("RGB")
    tensor = transform(image).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor.unsqueeze(0)))[0].cpu().numpy()

    top_class = int(np.argmax(probs))
    print(f"Top prediction: {DISEASE_LABELS[top_class]} ({probs[top_class]:.3f})")

    heatmap = compute_gradcam(model, tensor, target_class=top_class)
    print(f"GradCAM heatmap shape: {heatmap.shape}, max: {heatmap.max():.3f}")

    # Save overlay — TODO: use visualization utils to overlay on original image
    output_path = Path("results/figures") / (Path(args.image_path).stem + "_gradcam.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, heatmap)
    print(f"Saved heatmap: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--arch", default="resnet50")
    args = parser.parse_args()
    main(args)
