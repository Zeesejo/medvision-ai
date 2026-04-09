import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

N, C = 16, 14  # batch size, num classes


def make_batch(n=N, c=C):
    torch.manual_seed(0)
    logits = torch.randn(n, c)
    targets = torch.randint(0, 2, (n, c)).float()
    return logits, targets


class TestWeightedBCELoss:

    def test_output_is_scalar(self):
        from losses import WeightedBCELoss
        loss_fn = WeightedBCELoss()
        logits, targets = make_batch()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_loss_positive(self):
        from losses import WeightedBCELoss
        loss_fn = WeightedBCELoss()
        logits, targets = make_batch()
        loss = loss_fn(logits, targets)
        assert loss.item() > 0, "Loss should be positive"

    def test_loss_no_nan(self):
        from losses import WeightedBCELoss
        loss_fn = WeightedBCELoss()
        logits, targets = make_batch()
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss), "Loss must not be NaN"


class TestFocalLoss:

    def test_output_is_scalar(self):
        from losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits, targets = make_batch()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_gamma_zero_equals_bce(self):
        """FocalLoss with gamma=0 should approximate standard BCE."""
        from losses import FocalLoss, WeightedBCELoss
        logits, targets = make_batch()
        focal = FocalLoss(gamma=0.0)(logits, targets).item()
        bce = WeightedBCELoss()(logits, targets).item()
        assert abs(focal - bce) < 0.1, f"FocalLoss(gamma=0) should ~ BCE: {focal:.4f} vs {bce:.4f}"

    def test_loss_no_nan(self):
        from losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits, targets = make_batch()
        assert not torch.isnan(loss_fn(logits, targets))


class TestAsymmetricLoss:

    def test_output_is_scalar(self):
        from losses import AsymmetricLoss
        loss_fn = AsymmetricLoss()
        logits, targets = make_batch()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_loss_positive(self):
        from losses import AsymmetricLoss
        loss_fn = AsymmetricLoss()
        logits, targets = make_batch()
        assert loss_fn(logits, targets).item() > 0

    def test_loss_no_nan(self):
        from losses import AsymmetricLoss
        loss_fn = AsymmetricLoss()
        logits, targets = make_batch()
        assert not torch.isnan(loss_fn(logits, targets))

    def test_gamma_neg_zero_equals_focal(self):
        """ASL with gamma_neg=0, gamma_pos=0 should ~ BCE."""
        from losses import AsymmetricLoss, WeightedBCELoss
        logits, targets = make_batch()
        asl = AsymmetricLoss(gamma_neg=0, gamma_pos=0)(logits, targets).item()
        bce = WeightedBCELoss()(logits, targets).item()
        assert abs(asl - bce) < 0.2, f"ASL(0,0) should ~ BCE: {asl:.4f} vs {bce:.4f}"
