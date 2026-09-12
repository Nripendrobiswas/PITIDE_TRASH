"""TiDE: Time-series Predictive Deep model (Rasul et al., 2023), PyTorch.

Static covariate encoder + DNN encoder/decoder over flattened windows with
residual connections + temporal dynamic regressor on known future covariates.
"""

import torch
import torch.nn as nn

from .common import ResBlock, add_cdh, mlp


class TiDE(nn.Module):
    def __init__(
        self,
        n_hist,
        n_fut,
        n_static,
        l_hist,
        h_out,
        stats,
        hidden=128,
        latent=128,
        n_res=2,
        dropout=0.1,
    ):
        super().__init__()
        self.stats = stats
        self.n_fut1 = n_fut + 1
        self.h_out = h_out

        self.static_enc = mlp([n_static, hidden, hidden], dropout)
        self.ve = nn.Linear(hidden, hidden)
        self.vd = nn.Linear(hidden, hidden)

        self.enc0 = nn.Linear((n_hist + 1) * l_hist, hidden)
        self.enc_res = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_res)])
        self.enc_drop = nn.Dropout(dropout)
        self.latent = nn.Linear(hidden, latent)

        self.dec0 = nn.Linear(latent + self.n_fut1 * h_out, hidden)
        self.dec_res = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_res)])
        self.dec_drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, h_out)

        self.tdr = mlp([self.n_fut1, hidden, 1], dropout)

    def _forward_feats(self, xpf, xff, s):
        sv = self.static_enc(s)

        e = torch.relu(self.enc0(xpf.flatten(1)) + self.ve(sv))
        e = self.enc_drop(e)
        for blk in self.enc_res:
            e = blk(e)
        z = torch.relu(self.latent(e))

        d = torch.relu(self.dec0(torch.cat([z, xff.flatten(1)], dim=1)) + self.vd(sv))
        d = self.dec_drop(d)
        for blk in self.dec_res:
            d = blk(d)
        y = self.out(d)
        return y + self.tdr(xff).squeeze(-1)

    def forward(self, xp, xf, s):
        return self._forward_feats(add_cdh(xp, self.stats), add_cdh(xf, self.stats), s)
