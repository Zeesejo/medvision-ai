"""
ChestX-ray14 multi-label classifier.

Backbones are provided by timm and share one classification head so backbone
comparisons use the same output interface.
"""

from typing import Optional

import timm
import torch
import torch.nn as nn

from src.constants import CLASS_NAMES, NUM_CLASSES


class ChestXrayClassifier(nn.Module):
    """Multi-label classifier for NIH ChestX-ray14."""

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
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._init_head()

    def _init_head(self) -> None:
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits with shape ``[batch, num_classes]``."""
        return self.head(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self, unfreeze_last_n_layers: Optional[int] = None) -> None:
        if unfreeze_last_n_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        all_modules = list(self.backbone.named_modules())
        for _, layer in all_modules[-unfreeze_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(
    backbone: str = "resnet50",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> ChestXrayClassifier:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ChestXrayClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")

    model = model.to(device)
    params = model.count_parameters()
    print(f"Model : {backbone}")
    print(f"Params: {params['total']:,} total | {params['trainable']:,} trainable")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for backbone in ["resnet50", "densenet121", "vit_base_patch16_224"]:
        print(f"--- {backbone} ---")
        model = build_model(backbone=backbone, pretrained=False, device=device)
        dummy = torch.randn(2, 3, 224, 224, device=device)
        logits = model(dummy)
        print(f"Output shape : {logits.shape}")
        print(f"Features     : {model.get_features(dummy).shape}\n")

    print(f"Classes: {len(CLASS_NAMES)}")
    print("Model sanity check passed")
