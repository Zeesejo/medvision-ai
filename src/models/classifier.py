"""MedVision Classifier.

Supports two backbone families:
  - ResNet-50  (torchvision)  — fast, well-understood, Grad-CAM friendly
  - ViT-Base/16 (timm)        — stronger accuracy, attention map XAI

All heads output raw logits for BCEWithLogitsLoss (multi-label).
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

logger = logging.getLogger(__name__)


class MedVisionClassifier(nn.Module):
    """Flexible multi-label classifier for chest X-ray pathology detection."""

    SUPPORTED_BACKBONES = {
        "resnet50",
        "resnet18",
        "densenet121",
        "vit_base_patch16_224",
        "vit_small_patch16_224",
        "efficientnet_b3",
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone '{backbone}' not supported. Choose from: {self.SUPPORTED_BACKBONES}")

        self.backbone_name = backbone
        self.num_classes   = num_classes

        self.encoder, in_features = self._build_encoder(backbone, pretrained)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        logger.info(f"MedVisionClassifier | backbone={backbone} | classes={num_classes} | pretrained={pretrained}")

    def _build_encoder(self, backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Instantiate backbone and strip its original head. Return (encoder, feature_dim)."""

        # ---------- ResNet family ----------
        if backbone == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model   = tv_models.resnet50(weights=weights)
            in_feat = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_feat

        if backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model   = tv_models.resnet18(weights=weights)
            in_feat = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_feat

        if backbone == "densenet121":
            weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model   = tv_models.densenet121(weights=weights)
            in_feat = model.classifier.in_features
            model.classifier = nn.Identity()
            return model, in_feat

        # ---------- ViT / EfficientNet via timm ----------
        model   = timm.create_model(backbone, pretrained=pretrained, num_classes=0)  # num_classes=0 removes head
        in_feat = model.num_features
        return model, in_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) image tensor

        Returns:
            logits: (B, num_classes) — raw scores (apply sigmoid for probabilities)
        """
        features = self.encoder(x)          # (B, in_features)
        logits   = self.classifier(features) # (B, num_classes)
        return logits

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method — returns sigmoid-activated probabilities."""
        return torch.sigmoid(self.forward(x))

    @property
    def target_layer(self) -> nn.Module:
        """Return the last conv/attention layer for Grad-CAM targeting."""
        if self.backbone_name.startswith("resnet"):
            return self.encoder.layer4[-1]
        if self.backbone_name == "densenet121":
            return self.encoder.features.denseblock4.denselayer16
        if self.backbone_name.startswith("vit"):
            return self.encoder.blocks[-1].norm1
        raise NotImplementedError(f"target_layer not defined for {self.backbone_name}")
