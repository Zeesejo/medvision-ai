"""
Gradio web UI for MedVision AI.
Features: image upload, prediction, Grad-CAM overlay, confidence bar chart.

Run:
    python ui/app.py
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
CHECKPOINT = "results/checkpoints/finetune/model_best.pt"


def load_model():
    from models.encoder import MedEncoder, MultiLabelClassifier
    encoder = MedEncoder(arch="resnet50", pretrained=True)
    model = MultiLabelClassifier(encoder, num_classes=14)
    if Path(CHECKPOINT).exists():
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {CHECKPOINT}")
    else:
        print("No checkpoint found — using ImageNet-pretrained encoder (demo mode).")
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = load_model()


def predict(image: Image.Image):
    """Run inference and return top-5 predictions + Grad-CAM heatmap."""
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))[0].numpy()

    top5_idx = np.argsort(probs)[::-1][:5]
    predictions = {DISEASE_LABELS[i]: float(probs[i]) for i in top5_idx}

    # Grad-CAM overlay (requires grad-cam package)
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layers = [model.encoder.backbone.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        heatmap = cam(input_tensor=tensor,
                      targets=[ClassifierOutputTarget(int(top5_idx[0]))])[0]
        rgb = np.array(image.convert("RGB").resize((224, 224))) / 255.0
        overlay = show_cam_on_image(rgb.astype(np.float32), heatmap, use_rgb=True)
        overlay_pil = Image.fromarray(overlay)
    except Exception:
        overlay_pil = image  # fallback

    return predictions, overlay_pil


def launch():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("pip install gradio")

    with gr.Blocks(title="MedVision AI — Chest X-Ray Analysis") as demo:
        gr.Markdown("## 🩺 MedVision AI — Chest X-Ray Disease Classifier")
        gr.Markdown(
            "Upload a chest X-ray. The model predicts pathology probabilities "
            "and shows a Grad-CAM explanation overlay.  \n"
            "**⚠️ Research demo only — not for clinical use.**"
        )
        with gr.Row():
            input_img = gr.Image(type="pil", label="Input X-Ray")
            overlay_img = gr.Image(type="pil", label="Grad-CAM Overlay")
        with gr.Row():
            output_labels = gr.Label(num_top_classes=5, label="Top-5 Predictions")
        btn = gr.Button("Analyze")
        btn.click(fn=predict, inputs=input_img, outputs=[output_labels, overlay_img])

    demo.launch(share=False)


if __name__ == "__main__":
    launch()
