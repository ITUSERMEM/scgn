"""
encoders.py — CNN Encoders for Temporal and Spectro-Temporal Features
=====================================================================

Corresponds to Sec. III-C of the paper:
  "Local Temporal and Spectro-Temporal Encoding"

Both encoders use InstanceNorm (instead of BatchNorm) because vibration
and acoustic amplitudes drift heavily across operating conditions;
per-sample normalization forces the network to learn shape features.
"""

import torch
import torch.nn as nn


def _adaptive_window_bounds(in_size: int, out_idx: int, out_size: int):
    """Compute the start/end indices for a deterministic adaptive pooling window."""
    start = (out_idx * in_size) // out_size
    end = ((out_idx + 1) * in_size + out_size - 1) // out_size
    return start, max(start + 1, end)


class DeterministicAdaptiveAvgPool1d(nn.Module):
    """Deterministic substitute for nn.AdaptiveAvgPool1d on CUDA."""

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = int(output_size)
        if self.output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise AssertionError(f"Expected (N,C,L), got {tuple(x.shape)}")
        in_size = x.shape[-1]
        bins = []
        for out_idx in range(self.output_size):
            start, end = _adaptive_window_bounds(in_size, out_idx, self.output_size)
            bins.append(x[..., start:end].mean(dim=-1, keepdim=True))
        return torch.cat(bins, dim=-1)


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """Deterministic substitute for nn.AdaptiveAvgPool2d on CUDA."""

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(int(v) for v in output_size)
        if len(self.output_size) != 2 or min(self.output_size) <= 0:
            raise ValueError(
                f"output_size must be a positive int or pair, got {output_size}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise AssertionError(f"Expected (N,C,H,W), got {tuple(x.shape)}")
        in_h, in_w = x.shape[-2], x.shape[-1]
        out_h, out_w = self.output_size
        rows = []
        for out_i in range(out_h):
            hs, he = _adaptive_window_bounds(in_h, out_i, out_h)
            cols = []
            for out_j in range(out_w):
                ws, we = _adaptive_window_bounds(in_w, out_j, out_w)
                cols.append(x[..., hs:he, ws:we].mean(dim=(-2, -1), keepdim=True))
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)


class CNN1DEncoder(nn.Module):
    """1-D temporal encoder.

    Three conv blocks → adaptive-avg-pool → FC projection.
    Output dimensionality is configurable (default d=64 to match the paper).

    Paper: Sec. III-C, Eq. (8)–(9)
    """

    def __init__(self, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 25),
            nn.InstanceNorm1d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 24, 15),
            nn.InstanceNorm1d(24, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(24, 16, 13),
            nn.InstanceNorm1d(16, affine=True),
            nn.ReLU(),
            DeterministicAdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (N, 1, L). Returns: (N, out_dim)."""
        return self.head(self.features(x))


class CNN2DEncoder(nn.Module):
    """2-D spectrogram encoder — STFT magnitude input.

    Three conv blocks → adaptive-avg-pool → FC projection.
    Output dimensionality is configurable (default d=64 to match the paper).

    Paper: Sec. III-C, "Time-Frequency Encoding"
    """

    def __init__(self, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 24, 3),
            nn.InstanceNorm2d(24, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 16, 3),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            DeterministicAdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (N, 1, F, T). Returns: (N, out_dim)."""
        return self.head(self.features(x))
