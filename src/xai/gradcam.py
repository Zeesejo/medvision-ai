"""Grad-CAM explainability wrapper using pytorch-grad-cam library.

Install: pip install grad-cam
Docs:    https://github.com/jacobgil/pytorch-grad-cam
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import cv2

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    EigenCAM,
    ScoreCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

logger = logging.getLogger(__name__)

CAM_METHODS = {
    "gradcam":      GradCAM,
    "gradcam++":    GradCAMPlusPlus,
    "eigencam":     EigenCAM,
    "scorecam":     ScoreCAM,
}


class GradCAMExplainer:
    """Generate saliency heatmaps for a MedVisionClassifier prediction."""

    def __init__(
        self,
        model: nn.Module,
        method: str = "gradcam",
        device: Optional[torch.device] = None,
    ):
        if method not in CAM_METHODS:
            raise ValueError(f"method must be one of {list(CAM_METHODS.keys())}")

        self.model  = model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        target_layers = [model.target_layer]
        self.cam = CAM_METHODS[method](model=model, target_layers=target_layers)
        logger.info(f"GradCAMExplainer | method={method} | target_layer={model.target_layer}")

    def explain(
        self,
        image_tensor: torch.Tensor,
        class_idx: int,
        original_image: Optional[np.ndarray] = None,
    ) -> dict:
        """Generate a CAM heatmap for a single image and class.

        Args:
            image_tensor:   (1, 3, H, W) normalised input tensor
            class_idx:      Target class index (0-13 for ChestX-ray14)
            original_image: (H, W, 3) float32 [0,1] for overlay — if None, skip overlay

        Returns:
            dict with keys:
                'heatmap'  : (H, W) float32 CAM map in [0, 1]
                'overlay'  : (H, W, 3) float32 RGB overlay — only if original_image provided
                'class_idx': int
        """
        targets      = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = self.cam(input_tensor=image_tensor.to(self.device), targets=targets)
        heatmap       = grayscale_cam[0]  # (H, W)

        result = {"heatmap": heatmap, "class_idx": class_idx}

        if original_image is not None:
            overlay = show_cam_on_image(original_image.astype(np.float32), heatmap, use_rgb=True)
            result["overlay"] = overlay

        return result

    def explain_topk(
        self,
        image_tensor: torch.Tensor,
        k: int = 3,
        original_image: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """Generate CAM heatmaps for the top-k predicted classes.

        Returns:
            List of result dicts sorted by predicted probability (descending)
        """
        with torch.no_grad():
            probs = self.model.get_probabilities(image_tensor.to(self.device))
            topk  = torch.topk(probs[0], k=k)

        results = []
        for prob, cls_idx in zip(topk.values.cpu().tolist(), topk.indices.cpu().tolist()):
            r = self.explain(image_tensor, cls_idx, original_image)
            r["probability"] = prob
            results.append(r)

        return results
