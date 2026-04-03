"""SHAP-based explainability using captum + shap libraries.

Uses captum's GradientShap which scales well to image inputs.
For a full SHAP DeepExplainer, use the `shap` library directly.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from captum.attr import GradientShap, IntegratedGradients

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Attribution maps via GradientShap / IntegratedGradients (captum)."""

    METHODS = {"gradient_shap": GradientShap, "integrated_gradients": IntegratedGradients}

    def __init__(
        self,
        model: nn.Module,
        method: str = "gradient_shap",
        device: Optional[torch.device] = None,
    ):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {list(self.METHODS.keys())}")

        self.model  = model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.method_name = method
        self.explainer   = self.METHODS[method](self.model)
        logger.info(f"SHAPExplainer | method={method}")

    def explain(
        self,
        image_tensor: torch.Tensor,
        class_idx: int,
        n_samples: int = 50,
    ) -> np.ndarray:
        """Compute pixel-level attribution map.

        Args:
            image_tensor: (1, 3, H, W) normalised input tensor
            class_idx:    Target class index
            n_samples:    Number of baseline samples (GradientShap only)

        Returns:
            attribution: (H, W) float32 — mean absolute attribution across channels
        """
        image_tensor = image_tensor.to(self.device).requires_grad_(True)

        if self.method_name == "gradient_shap":
            # Baselines: random Gaussian noise (shape same as input)
            baselines  = torch.randn(n_samples, *image_tensor.shape[1:]).to(self.device)
            attribution = self.explainer.attribute(
                image_tensor,
                baselines=baselines,
                target=class_idx,
                n_samples=n_samples,
                stdevs=0.09,
            )
        else:  # integrated_gradients
            baseline   = torch.zeros_like(image_tensor)
            attribution = self.explainer.attribute(
                image_tensor,
                baselines=baseline,
                target=class_idx,
                n_steps=50,
            )

        # Aggregate across colour channels -> (H, W)
        attr_map = attribution[0].cpu().detach().numpy()  # (3, H, W)
        attr_map = np.mean(np.abs(attr_map), axis=0)       # (H, W)
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
        return attr_map.astype(np.float32)
