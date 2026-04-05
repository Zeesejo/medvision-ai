"""
Grad-CAM for ChestX-ray14 Multi-Label Classifier
=================================================
Generates class-specific heatmaps overlaid on chest X-rays.
Used for paper figures and the Gradio UI.

Usage:
    from src.xai.gradcam import GradCAM, overlay_heatmap
    cam = GradCAM(model, target_layer='layer4')  # resnet50
    heatmap = cam(image_tensor, class_idx=2)     # class 2 = Effusion
    vis = overlay_heatmap(original_image, heatmap)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms as T


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Works with ResNet50 and ViT backbones via timm.
    """

    def __init__(self, model, target_layer: str = 'layer4'):
        self.model  = model
        self.device = next(model.parameters()).device
        self._features = None
        self._grads    = None
        self._hook_handles = []
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: str):
        # Navigate to the target layer in the backbone
        layer = dict(self.model.backbone.named_modules()).get(target_layer)
        if layer is None:
            available = [n for n, _ in self.model.backbone.named_modules()]
            raise ValueError(
                f"Layer '{target_layer}' not found.\n"
                f"Available layers: {available[-10:]}"  # show last 10
            )

        def fwd_hook(module, input, output):
            self._features = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._hook_handles.append(layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def __call__(
        self,
        image_tensor: torch.Tensor,   # [1, 3, H, W]
        class_idx: int,
    ) -> np.ndarray:
        """
        Returns a normalised heatmap [H, W] in range [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)          # [1, 14]
        score  = logits[0, class_idx]

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Pool gradients over spatial dims
        weights  = self._grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam      = (weights * self._features).sum(dim=1)        # [1, H, W]
        cam      = F.relu(cam).squeeze(0).cpu().numpy()         # [H, W]

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on the original X-ray image.

    Args:
        original_image : PIL Image (grayscale or RGB)
        heatmap        : [H, W] float32 array in [0, 1]
        alpha          : heatmap opacity
        colormap       : OpenCV colormap

    Returns:
        PIL Image with heatmap overlay
    """
    # Resize heatmap to image size
    img_w, img_h = original_image.size
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_uint8, (img_w, img_h))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)  # BGR
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Convert original to RGB numpy
    img_rgb = np.array(original_image.convert('RGB'))

    # Blend
    blended = (1 - alpha) * img_rgb + alpha * heatmap_colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def get_image_tensor(
    image_path: str,
    image_size: int = 224,
) -> tuple:
    """
    Load image and return (tensor [1,3,H,W], original PIL Image).
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor, img


if __name__ == '__main__':
    # Quick test with a dummy model
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.classifier import build_model

    model = build_model(backbone='resnet50', pretrained=False, device='cpu')
    cam   = GradCAM(model, target_layer='layer4')

    dummy = torch.randn(1, 3, 224, 224)
    hmap  = cam(dummy, class_idx=0)
    print(f'Heatmap shape : {hmap.shape}')   # (7, 7)
    print(f'Heatmap range : [{hmap.min():.3f}, {hmap.max():.3f}]')
    cam.remove_hooks()
    print('GradCAM test passed ✓')
