import torch
import torch.nn as nn

from data_pipeline import CDH_BASE, IDX_T


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)


def add_cdh(x, stats):
    """Append a standardized cooling-degree-hours channel derived from raw T.

    x[..., IDX_T] is standardized temperature; recover physical T differentiably,
    so autograd gradients w.r.t. T propagate through the CDH channel as well.
    """
    t_raw = x[..., IDX_T] * stats["std"][IDX_T] + stats["mean"][IDX_T]
    cdh = torch.relu(t_raw - CDH_BASE)
    cdh = (cdh - stats["cdh_mean"]) / stats["cdh_std"]
    return torch.cat([x, cdh[..., None]], dim=-1)


def mlp(dims, dropout=0.1):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1])]
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)
