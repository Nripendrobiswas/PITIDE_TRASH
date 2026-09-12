"""Metrics for electricity demand forecasting."""
import numpy as np


def to_mw(a, dmean, dstd):
    return a * dstd + dmean


def metrics(y_mw, p_mw, peak_mask=None, prefix=""):
    y, p = np.asarray(y_mw, float), np.asarray(p_mw, float)
    e = p - y
    out = {
        f"{prefix}mae": float(np.mean(np.abs(e))),
        f"{prefix}rmse": float(np.sqrt(np.mean(e ** 2))),
        f"{prefix}mape": float(np.mean(np.abs(e) / np.maximum(y, 1.0)) * 100),
        f"{prefix}smape": float(np.mean(2 * np.abs(e) / np.maximum(np.abs(y) + np.abs(p), 1.0)) * 100),
    }
    if peak_mask is not None:
        m = np.asarray(peak_mask, bool)
        out[f"{prefix}mae_peak"] = float(np.mean(np.abs(e[m]))) if m.any() else float("nan")
    return out


def ramp_violation_rate(p_scaled, hist_last_scaled, dstd, ramp_mw):
    """Fraction of predicted hour-to-hour steps exceeding the grid ramp limit (MW)."""
    first = p_scaled[:, 0] - hist_last_scaled
    dd = np.abs(np.concatenate([first[:, None], p_scaled[:, 1:] - p_scaled[:, :-1]], axis=1)) * dstd
    return float((dd > ramp_mw).mean())
