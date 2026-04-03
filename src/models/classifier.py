"""
ChestX-ray14 Multi-Label Classifier
=====================================
Supports two backbone options:
  - resnet50   : Fast baseline, strong AUC, easy to explain with Grad-CAM
  - vit_base   : Vision Transformer, better long-range features, paper novelty

Both share the same multi-label head and training interface.
"""

import torch
import torch.nn as nn
import timm
from typing import Literal

NUM_CLASSES = 14


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------
class ChestXrayClassifier(nn.Module):
    """
    Multi-label classifier for NIH ChestX-ray14.

    Args:
        backbone  : 'resnet50' | 'vit_base_patch16_224'
        num_classes: number of output labels (default 14)
        pretrained : load ImageNet weights
        dropout   : dropout rate before classifier head
        freeze_backbone: freeze all backbone layers (linear probe mode)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes   = num_classes

        # ---- Load backbone via timm ----
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # remove original head
            global_pool="avg",      # global average pooling
        )
        feature_dim = self.backbone.num_features

        # ---- Classification head ----
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),   # raw logits — no sigmoid here
        )

        # ---- Optional: freeze backbone for linear probe ----
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---- Weight initialisation for head ----
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits [B, num_classes]. Apply sigmoid for probabilities."""
        features = self.backbone(x)   # [B, feature_dim]
        logits   = self.head(features) # [B, num_classes]
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone embeddings (useful for t-SNE / UMAP visualisation)."""
        return self.backbone(x)

    def unfreeze_backbone(self, unfreeze_last_n_layers: int = None):
        """
        Unfreeze backbone layers for fine-tuning.
        If unfreeze_last_n_layers is None, unfreeze everything.
        """
        if unfreeze_last_n_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            layers = list(self.backbone.children())
            for layer in layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------
def build_model(
    backbone: str = "resnet50",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    checkpoint_path: str = None,
    device: str = "cuda",
) -> ChestXrayClassifier:
    """
    Build and optionally load a pretrained checkpoint.

    Args:
        backbone       : timm model name
        num_classes    : output classes
        pretrained     : use ImageNet weights
        dropout        : head dropout
        freeze_backbone: linear probe mode
        checkpoint_path: path to .pth checkpoint to resume from
        device         : 'cuda' | 'cpu'

    Returns:
        model on the specified device
    """
    model = ChestXrayClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        # Support both raw state_dict and checkpoint dicts
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")

    model = model.to(device)
    params = model.count_parameters()
    print(f"Model : {backbone}")
    print(f"Params: {params['total']:,} total | {params['trainable']:,} trainable")
    return model


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for backbone in ["resnet50", "vit_base_patch16_224"]:
        print(f"--- {backbone} ---")
        model = build_model(backbone=backbone, pretrained=False, device=device)
        dummy = torch.randn(4, 3, 224, 224).to(device)
        logits = model(dummy)
        print(f"Output shape : {logits.shape}"   )  # [4, 14]
        print(f"Features     : {model.get_features(dummy).shape}\n")

    print("Model sanity check passed ✓")
