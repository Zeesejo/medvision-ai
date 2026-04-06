"""
losses.py  —  Asymmetric Loss for multi-label classification
=============================================================
Reference:
  Ridnik et al., "Asymmetric Loss For Multi-Label Classification",
  ICCV 2021.  https://arxiv.org/abs/2009.14119
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL).

    Parameters
    ----------
    gamma_neg : float
        Focusing parameter for negative samples (hard negatives down-weighted).
        Recommended: 4.
    gamma_pos : float
        Focusing parameter for positive samples. Usually 0 or 1.
    clip : float
        Probability margin for shifting hard negatives.
        p_neg = max(p - clip, 0).  Recommended: 0.05.
    eps : float
        Small value for numerical stability in log.
    disable_torch_grad_focal_loss : bool
        If True, detach the focal weight from the computation graph
        (saves memory, negligible accuracy impact).
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C)  raw model outputs (before sigmoid)
        targets : (B, C)  binary labels in {0, 1}
        """
        probs = torch.sigmoid(logits)

        # Asymmetric clip: shift negative probabilities down
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs - self.clip).clamp(min=0)
        else:
            probs_neg = probs

        # Log-probabilities
        log_pos = torch.log(probs.clamp(min=self.eps))
        log_neg = torch.log((1.0 - probs_neg).clamp(min=self.eps))

        # Binary cross-entropy base
        loss = targets * log_pos + (1.0 - targets) * log_neg

        # Asymmetric focusing weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt_pos = probs
                    pt_neg = probs_neg
                    focal_pos = (1.0 - pt_pos) ** self.gamma_pos
                    focal_neg = pt_neg ** self.gamma_neg
                    focal = targets * focal_pos + (1.0 - targets) * focal_neg
            else:
                pt_pos    = probs
                pt_neg    = probs_neg
                focal_pos = (1.0 - pt_pos) ** self.gamma_pos
                focal_neg = pt_neg ** self.gamma_neg
                focal     = targets * focal_pos + (1.0 - targets) * focal_neg

            loss = loss * focal

        return -loss.mean()
