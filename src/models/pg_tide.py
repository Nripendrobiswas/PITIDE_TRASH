"""Physics-Guided TiDE.

TiDE backbone + two physics mechanisms:
1. Structural: a monotone thermal branch  cdh_t * |w_h|  added to the output,
   which by construction cannot decrease demand when temperature rises.
2. Variational: the same physics losses used for the PINN (thermal sensitivity
   of the whole network + grid ramp-rate limits) are added to training loss.
"""

import torch
import torch.nn as nn

from .common import add_cdh
from .tide import TiDE


class PGTiDE(TiDE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_thermal = nn.Parameter(torch.zeros(self.h_out))

    def forward(self, xp, xf, s):
        xpf = add_cdh(xp, self.stats)
        xff = add_cdh(xf, self.stats)
        y = self._forward_feats(xpf, xff, s)
        cdh = xff[..., -1]  # standardized future cooling-degree-hours
        return y + cdh * torch.abs(self.w_thermal)[None, :]
