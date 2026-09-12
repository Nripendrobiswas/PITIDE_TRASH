"""Baselines: seasonal-naive (numpy) and an LSTM forecaster (PyTorch)."""
import numpy as np
import torch
import torch.nn as nn

from .common import add_cdh
from data_pipeline import IDX_DEMAND


def seasonal_naive(xp, horizon):
    """y(t+h) = y(t+h-168): last week's same hour. xp: (M, L, C) scaled features."""
    l = xp.shape[1]
    d = xp[:, :, IDX_DEMAND]
    reps = int(np.ceil(horizon / l))
    return np.tile(d, (1, reps))[:, :horizon]


class LSTMModel(nn.Module):
    def __init__(self, n_hist, n_fut, n_static, l_hist, h_out, stats,
                 hidden=96, layers=2, dropout=0.1):
        super().__init__()
        self.stats = stats
        self.lstm = nn.LSTM(n_hist + 1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fut_proj = nn.Linear((n_fut + 1) * h_out, hidden)
        self.head = nn.Linear(2 * hidden, h_out)

    def forward(self, xp, xf, s):
        xp = add_cdh(xp, self.stats)
        xf = add_cdh(xf, self.stats)
        out, _ = self.lstm(xp)
        h = torch.cat([out[:, -1], torch.relu(self.fut_proj(xf.flatten(1)))], dim=1)
        return self.head(h)
