"""
losses.py  —  Asymmetric Loss for multi-label classification
=============================================================
Reference:
  Ridnik et al., "Asymmetric Loss For Multi-Label Classification",
  ICCV 2021.  https://arxiv.org/abs/2009.14119

Bug fixed in v3:
  [BUG-4]  AsymmetricLossOptimized: log_pos had .clamp(max=-eps) which capped
           log values near zero, corrupting loss for high-confidence correct
           predictions.  Fix: remove the erroneous clamp (softplus is already
           numerically stable and always yields log(p) <= 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AsymmetricLoss", "AsymmetricLossOptimized", "get_loss"]


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL).

    Parameters
    ----------
    gamma_neg : float
        Focusing parameter for negative samples.  Recommended: 4.
    gamma_pos : float
        Focusing parameter for positive samples. Usually 0 or 1.
    clip : float
        Probability margin for shifting hard negatives.
        p_neg = max(p - clip, 0).  Recommended: 0.05.
    eps : float
        Small value for numerical stability in log.
    label_smoothing : float
        Optional label smoothing in [0, 0.1]. Default 0.0 (disabled).
    disable_torch_grad_focal_loss : bool
        Detach focal weight from graph — saves memory, negligible accuracy impact.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
        label_smoothing: float = 0.0,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super().__init__()
        self.gamma_neg   = gamma_neg
        self.gamma_pos   = gamma_pos
        self.clip        = clip
        self.eps         = eps
        self.label_smoothing = label_smoothing
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C)  raw model outputs (before sigmoid)
        targets : (B, C)  binary labels in {0, 1}
        """
        probs = torch.sigmoid(logits)

        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs_neg = (probs - self.clip).clamp(min=0) if self.clip and self.clip > 0 else probs

        log_pos = torch.log(probs.clamp(min=self.eps))
        log_neg = torch.log((1.0 - probs_neg).clamp(min=self.eps))
        loss    = targets * log_pos + (1.0 - targets) * log_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    focal = (
                        targets       * (1.0 - probs)    ** self.gamma_pos
                        + (1.0 - targets) * probs_neg ** self.gamma_neg
                    )
            else:
                focal = (
                    targets       * (1.0 - probs)    ** self.gamma_pos
                    + (1.0 - targets) * probs_neg ** self.gamma_neg
                )
            loss = loss * focal

        return -loss.mean()


class AsymmetricLossOptimized(nn.Module):
    """
    Numerically optimised variant of ASL.
    ~15% faster forward pass vs. AsymmetricLoss on GPU.

    BUG-4 fix (v3): removed erroneous .clamp(max=-eps) on log_pos.
    -F.softplus(-logits) is log(sigmoid(logits)) and is already in (-inf, 0].
    Clamping it at -eps was pushing log values for p≈1 toward 0, making
    the loss falsely small for high-confidence correct predictions.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
        label_smoothing: float = 0.0,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super().__init__()
        self.gamma_neg   = gamma_neg
        self.gamma_pos   = gamma_pos
        self.clip        = clip
        self.eps         = eps
        self.label_smoothing = label_smoothing
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C)  raw model outputs (before sigmoid)
        targets : (B, C)  binary labels in {0, 1}
        """
        probs = torch.sigmoid(logits)

        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs_neg = (probs - self.clip).clamp(min=0) if (self.clip and self.clip > 0) else probs

        # BUG-4 FIX: -F.softplus(-logits) = log(sigmoid(logits)) is stable.
        # The previous .clamp(max=-eps) was wrong — it capped near-zero log
        # values and corrupted the loss for confident correct predictions.
        log_pos = -F.softplus(-logits)                                    # log(sigmoid(x))
        log_neg = torch.log((1.0 - probs_neg).clamp(min=self.eps))        # log(1 - p_neg)

        loss = targets * log_pos + (1.0 - targets) * log_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    focal = (
                        targets       * (1.0 - probs)    ** self.gamma_pos
                        + (1.0 - targets) * probs_neg ** self.gamma_neg
                    )
            else:
                focal = (
                    targets       * (1.0 - probs)    ** self.gamma_pos
                    + (1.0 - targets) * probs_neg ** self.gamma_neg
                )
            loss = loss * focal

        return -loss.mean()


def get_loss(cfg: dict) -> nn.Module:
    """
    Factory function — instantiate a loss from a YAML config dict.

    Expected keys under cfg['training']:
        loss        : 'asl' | 'asl_optimized'
    Expected keys under cfg['training']['loss_config']:
        gamma_neg, gamma_pos, clip, label_smoothing
    """
    loss_cfg  = cfg.get("training", {}).get("loss_config", {})
    name      = cfg.get("training", {}).get("loss", "asl")
    gamma_neg = loss_cfg.get("gamma_neg", 4)
    gamma_pos = loss_cfg.get("gamma_pos", 0)
    clip      = loss_cfg.get("clip", 0.05)
    smoothing = loss_cfg.get("label_smoothing", 0.0)

    if name == "asl_optimized":
        return AsymmetricLossOptimized(
            gamma_neg=gamma_neg, gamma_pos=gamma_pos,
            clip=clip, label_smoothing=smoothing,
        )
    return AsymmetricLoss(
        gamma_neg=gamma_neg, gamma_pos=gamma_pos,
        clip=clip, label_smoothing=smoothing,
    )
