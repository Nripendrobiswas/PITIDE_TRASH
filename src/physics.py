"""Physics-based penalty terms for electricity demand forecasting.

L_thermal : demand sensitivity to temperature must be non-negative in the hot
            regime (T > comfort threshold) - AC-driven load physics.
L_ramp    : predicted hour-to-hour changes must respect the historical maximum
            ramp rate of the national grid (MW/h generator ramping limit).
"""
import numpy as np  # noqa: F401  (used in thermal_violation_rate)
import torch
import torch.nn.functional as F

from data_pipeline import CDH_BASE, IDX_T, IDX_DEMAND


class PhysicsLoss:
    def __init__(self, lam_thermal, lam_smooth, stats, ramp_scaled):
        self.lam_t = float(lam_thermal)
        self.lam_s = float(lam_smooth)
        self.t_mean, self.t_std = stats["mean"][IDX_T], stats["std"][IDX_T]
        self.ramp_scaled = float(ramp_scaled)

    def thermal(self, model, xp, xf, s):
        """Mean negative d(sum_h yhat)/dT at hot timesteps (autograd through CDH)."""
        xp = xp.detach().clone().requires_grad_(True)
        out = model(xp, xf, s)
        g = torch.autograd.grad(out.sum(), xp, create_graph=True)[0][..., IDX_T]  # (B,L)
        raw_t = xp.detach()[..., IDX_T] * self.t_std + self.t_mean
        hot = raw_t > CDH_BASE
        denom = hot.sum().clamp(min=1)
        return (F.relu(-g) * hot).sum() / denom

    def ramp(self, out, xp):
        first = out[:, 0] - xp[:, -1, IDX_DEMAND]
        rest = out[:, 1:] - out[:, :-1]
        dd = torch.cat([first[:, None], rest], dim=1)
        return F.relu(dd.abs() - self.ramp_scaled).mean()

    def __call__(self, model, xp, xf, s, out=None):
        loss = xp.new_zeros(())
        if self.lam_t > 0:
            loss = loss + self.lam_t * self.thermal(model, xp, xf, s)
        if out is None:
            out = model(xp, xf, s)
        if self.lam_s > 0:
            loss = loss + self.lam_s * self.ramp(out, xp.detach())
        return loss


def thermal_violation_rate(model, xp, xf, s, stats, batch=512, device="cpu"):
    """Fraction of hot input timesteps where the model's dDemand/dT < 0."""
    t_mean, t_std = stats["mean"][IDX_T], stats["std"][IDX_T]
    neg_hits = tot_hits = 0
    was_training = model.training
    model.eval()
    for i in range(0, len(xp), batch):
        xb = torch.as_tensor(np.asarray(xp[i:i + batch])).to(device).clone().requires_grad_(True)
        xfb = torch.as_tensor(np.asarray(xf[i:i + batch])).to(device)
        sb = torch.as_tensor(np.asarray(s[i:i + batch])).to(device)
        out = model(xb, xfb, sb)
        g = torch.autograd.grad(out.sum(), xb)[0][..., IDX_T]
        hot = (xb.detach()[..., IDX_T] * t_std + t_mean) > CDH_BASE
        neg_hits += int(((g < 0) & hot).sum())
        tot_hits += int(hot.sum())
    if was_training:
        model.train()
    return neg_hits / max(tot_hits, 1)
