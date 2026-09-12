"""Physics-Informed Neural Network forecaster: MLP trained with physics losses.

The physics (thermal sensitivity + grid ramp-rate limits, see src/physics.py)
is imposed through the loss; the network is a fully-connected forecast head
over the past window plus static covariates.
"""
import torch
import torch.nn as nn

from .common import ResBlock, add_cdh


class PINNMLP(nn.Module):
    def __init__(self, n_hist, n_fut, n_static, l_hist, h_out, stats,
                 hidden=128, n_res=2, dropout=0.1):
        super().__init__()
        self.stats = stats
        self.n_fut1 = n_fut + 1
        dims = [(n_hist + 1) * l_hist + n_static + self.n_fut1 * h_out] + [hidden] * 3
        self.stem = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.res = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_res)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, h_out)

    def forward(self, xp, xf, s):
        xp = add_cdh(xp, self.stats)
        xf = add_cdh(xf, self.stats)
        x = torch.cat([xp.flatten(1), xf.flatten(1), s], dim=1)
        h = x
        for lin in self.stem:
            h = torch.relu(lin(h))
        h = self.drop(h)
        for blk in self.res:
            h = blk(h)
        return self.head(h)
