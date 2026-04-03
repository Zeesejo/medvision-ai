"""
Projection and prediction heads for self-supervised learning methods.
Supports SimCLR, BYOL, and MoCo-style heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRProjectionHead(nn.Module):
    """
    2-layer MLP projection head as used in SimCLR.
    Input: encoder features  Output: normalized z vector
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class BYOLPredictionHead(nn.Module):
    """
    BYOL prediction head: maps online z to target z.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 4096, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent) for SimCLR.

    Args:
        temperature: softmax temperature tau
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2 * N, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
        loss = F.cross_entropy(sim, labels)
        return loss
