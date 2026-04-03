"""
Loss functions for multi-label chest X-ray classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with per-class positive weights.
    Handles severe class imbalance in ChestX-ray14.

    Usage:
        pos_weights = loaders['pos_weights'].to(device)  # from get_dataloaders()
        criterion = WeightedBCELoss(pos_weights)
        loss = criterion(logits, targets)
    """

    def __init__(self, pos_weights: torch.Tensor = None):
        super().__init__()
        self.pos_weights = pos_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weights.to(logits.device) if self.pos_weights is not None else None,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Down-weights easy negatives — helps with extreme class imbalance.

    Args:
        gamma : focusing parameter (2.0 is standard)
        alpha : scalar weight for positive class
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) — state-of-the-art for multi-label classification.
    From: 'Asymmetric Loss For Multi-Label Classification' (Ben-Baruch et al., ICCV 2021)
    Penalises false negatives more than false positives.

    Args:
        gamma_neg : focusing for negatives (default 4)
        gamma_pos : focusing for positives (default 0)
        clip      : probability margin to shift negatives
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs    = torch.sigmoid(logits)
        probs_neg = (probs - self.clip).clamp(min=0)   # shift negatives

        loss_pos = targets       * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        loss = loss_pos + loss_neg

        # Asymmetric focusing
        with torch.no_grad():
            p_t = probs * targets + probs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            weight = (1 - p_t) ** gamma

        loss = -weight * loss
        return loss.mean()
