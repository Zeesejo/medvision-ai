"""
Backbone encoder wrappers (ResNet, ViT) for medical image SSL and fine-tuning.
"""
from typing import Optional

import torch
import torch.nn as nn
import timm


class MedEncoder(nn.Module):
    """
    Flexible backbone encoder built on timm.

    Args:
        arch: timm model name, e.g. 'resnet50', 'vit_base_patch16_224'
        pretrained: load ImageNet weights
        out_dim: output feature dimension (None = model default)
        freeze_bn: freeze BatchNorm layers (useful for small batch SSL)
    """

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = False,
        out_dim: Optional[int] = None,
        freeze_bn: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features

        if out_dim is not None and out_dim != self.feat_dim:
            self.proj = nn.Linear(self.feat_dim, out_dim)
        else:
            self.proj = nn.Identity()

        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class MultiLabelClassifier(nn.Module):
    """
    Encoder + classification head for multi-label medical image classification.

    Args:
        encoder: MedEncoder backbone
        num_classes: number of output labels
        dropout: dropout rate before classifier
    """

    def __init__(self, encoder: MedEncoder, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.classifier(feats)
