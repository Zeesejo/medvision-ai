"""
MedVision-AI — Gradio Web Interface
=====================================
Upload a chest X-ray → get disease predictions + Grad-CAM heatmap.

Usage:
    python ui/app.py
    python ui/app.py --checkpoint results/checkpoints/resnet50_best.pth

Requires:
    pip install gradio opencv-python-headless
"""

import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.classifier import build_model
from src.xai.gradcam import GradCAM, overlay_heatmap, get_image_tensor
from src.data.dataset import CLASS_NAMES

# ── Config ──
def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

cfg    = load_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load model once at startup ──
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None)
args, _ = parser.parse_known_args()

CKPT = args.checkpoint or str(
    Path(cfg['logging']['save_dir']) / f"{cfg['model']['backbone']}_best.pth"
)

print(f'Loading model from {CKPT}...')
model = build_model(
    backbone        = cfg['model']['backbone'],
    pretrained      = False,
    checkpoint_path = CKPT,
    device          = device,
)
model.eval()
gradcam = GradCAM(model, target_layer='layer4')
print('Model ready.')


# ── Inference function ──
def predict(image: Image.Image, top_k: int = 5, cam_class: str = 'Auto (top prediction)'):
    if image is None:
        return None, "Please upload a chest X-ray image."

    # Preprocess
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # Sort by probability
    indices   = np.argsort(probs)[::-1]
    top_class = indices[0]

    # Grad-CAM
    if cam_class == 'Auto (top prediction)':
        cam_idx = int(top_class)
    else:
        cam_idx = CLASS_NAMES.index(cam_class)

    heatmap    = gradcam(tensor, class_idx=cam_idx)
    overlay    = overlay_heatmap(image.convert('RGB'), heatmap, alpha=0.45)

    # Build results text
    lines = [f"{'Disease':<28} {'Probability':>12}\n" + "-"*42]
    for i in indices[:top_k]:
        bar  = "█" * int(probs[i] * 20)
        flag = " ⚠️" if probs[i] > 0.5 else ""
        lines.append(f"{CLASS_NAMES[i]:<28} {probs[i]:>8.1%}  {bar}{flag}")

    result_text = "\n".join(lines)
    result_text += f"\n\nGrad-CAM shown for: {CLASS_NAMES[cam_idx]}"

    return overlay, result_text


# ── Gradio UI ──
with gr.Blocks(
    title="MedVision-AI — Chest X-Ray Analysis",
    theme=gr.themes.Soft(primary_hue="teal"),
) as demo:

    gr.Markdown("""
    # 🫑 MedVision-AI — Chest X-Ray Disease Detection
    Upload a chest X-ray to detect 14 pathologies with AI.
    Grad-CAM heatmap shows **which regions** the model focuses on.
    > ⚠️ For research purposes only. Not a medical device.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input  = gr.Image(type='pil', label='Upload Chest X-Ray')
            top_k      = gr.Slider(1, 14, value=5, step=1, label='Show top-K predictions')
            cam_class  = gr.Dropdown(
                choices=['Auto (top prediction)'] + CLASS_NAMES,
                value='Auto (top prediction)',
                label='Grad-CAM target class'
            )
            btn = gr.Button('Analyse', variant='primary')

        with gr.Column(scale=1):
            img_output  = gr.Image(label='Grad-CAM Heatmap')
            text_output = gr.Textbox(label='Predictions', lines=10, max_lines=20)

    btn.click(
        fn=predict,
        inputs=[img_input, top_k, cam_class],
        outputs=[img_output, text_output]
    )

    gr.Examples(
        examples=[],
        inputs=img_input,
    )

    gr.Markdown("""
    ---
    **Model:** ResNet50 pretrained on ImageNet, fine-tuned on NIH ChestX-ray14 (112,120 images)
    **Classes:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
    Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia
    """)


if __name__ == '__main__':
    demo.launch(share=False, server_port=7860)
