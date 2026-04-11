"""
tests/test_losses.py
====================
Unit tests for AsymmetricLoss and AsymmetricLossOptimized.

New in v3 (Copilot review): added focused tests for AsymmetricLossOptimized
to protect the BUG-4 fix (removed erroneous log_pos clamp) from regression.
"""

import math
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.losses import AsymmetricLoss, AsymmetricLossOptimized


# ── helpers ────────────────────────────────────────────────────────────────────
def _make_batch(B=8, C=14, seed=42):
    torch.manual_seed(seed)
    logits  = torch.randn(B, C)
    targets = (torch.rand(B, C) > 0.7).float()
    return logits, targets


# ── AsymmetricLoss (original) ───────────────────────────────────────────────────
class TestAsymmetricLoss:
    def test_output_shape(self):
        logits, targets = _make_batch()
        loss = AsymmetricLoss()(logits, targets)
        assert loss.shape == torch.Size([]), "Loss must be a scalar"

    def test_positive_loss(self):
        logits, targets = _make_batch()
        loss = AsymmetricLoss()(logits, targets)
        assert loss.item() > 0, "Loss must be positive"

    def test_loss_decreases_with_confident_correct_prediction(self):
        """All-positive targets with very large positive logits → loss near 0."""
        logits  = torch.full((4, 4), 10.0)
        targets = torch.ones(4, 4)
        loss = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.0)(logits, targets)
        assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"

    def test_label_smoothing(self):
        logits, targets = _make_batch()
        loss_no_smooth  = AsymmetricLoss(label_smoothing=0.0)(logits, targets)
        loss_smoothed   = AsymmetricLoss(label_smoothing=0.05)(logits, targets)
        assert loss_no_smooth.item() != loss_smoothed.item()


# ── AsymmetricLossOptimized — BUG-4 regression tests ──────────────────────────
class TestAsymmetricLossOptimized:
    def test_output_shape(self):
        logits, targets = _make_batch()
        loss = AsymmetricLossOptimized()(logits, targets)
        assert loss.shape == torch.Size([])

    def test_positive_loss(self):
        logits, targets = _make_batch()
        loss = AsymmetricLossOptimized()(logits, targets)
        assert loss.item() > 0

    def test_matches_base_asl_on_random_inputs(self):
        """
        AsymmetricLossOptimized must produce values close to AsymmetricLoss
        on the same inputs (they differ only in numeric path, not semantics).
        Tolerance is generous (1e-4) to account for float32 rounding.
        """
        logits, targets = _make_batch(seed=0)
        base = AsymmetricLoss(
            gamma_neg=4, gamma_pos=0, clip=0.05,
            disable_torch_grad_focal_loss=True,
        )(logits, targets)
        opt = AsymmetricLossOptimized(
            gamma_neg=4, gamma_pos=0, clip=0.05,
            disable_torch_grad_focal_loss=True,
        )(logits, targets)
        assert abs(base.item() - opt.item()) < 1e-4, (
            f"AsymmetricLoss={base.item():.6f}  "
            f"AsymmetricLossOptimized={opt.item():.6f}  delta too large"
        )

    def test_no_artificial_clamp_on_high_confidence_correct(
        self,
    ):
        """
        BUG-4 regression test.
        With the old code, .clamp(max=-eps) on log_pos capped log values near 0
        and made the loss artificially small for high-confidence correct predictions.
        After the fix, a confident correct prediction (logit=10, target=1) should
        contribute near-zero loss (log(sigmoid(10)) ≈ 0), not a clamped near-zero.
        More importantly: the optimized variant must agree with the base variant
        on extreme logits, where the bug was most visible.
        """
        extreme_logits  = torch.tensor([[10.0, -10.0, 5.0, -5.0]])
        targets         = torch.tensor([[1.0,   0.0,  1.0,  0.0]])

        base = AsymmetricLoss(
            gamma_neg=4, gamma_pos=0, clip=0.0,
            disable_torch_grad_focal_loss=False,
        )(extreme_logits, targets)
        opt = AsymmetricLossOptimized(
            gamma_neg=4, gamma_pos=0, clip=0.0,
            disable_torch_grad_focal_loss=False,
        )(extreme_logits, targets)

        assert abs(base.item() - opt.item()) < 1e-5, (
            f"Extreme-logit mismatch: base={base.item():.8f}  "
            f"opt={opt.item():.8f}  — BUG-4 may have regressed"
        )

    def test_label_smoothing_optimized(self):
        logits, targets = _make_batch()
        loss_no_smooth  = AsymmetricLossOptimized(label_smoothing=0.0)(logits, targets)
        loss_smoothed   = AsymmetricLossOptimized(label_smoothing=0.05)(logits, targets)
        assert loss_no_smooth.item() != loss_smoothed.item()

    def test_gradient_flows(self):
        """Ensure backward pass completes without error."""
        logits  = torch.randn(4, 14, requires_grad=True)
        targets = (torch.rand(4, 14) > 0.7).float()
        loss = AsymmetricLossOptimized()(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
