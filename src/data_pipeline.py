"""Data loading, cleaning, feature engineering and zero-copy windowing."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import Config  # noqa: E402

COL_T = "Temperature(2m)"
COL_RH = "Rhumadity(2m)"
COL_SP = "SPressure(kPa)"

# channel order in the history matrix (idx 0 = raw temperature; idx 10 = demand target)
HIST_CHANNELS = [
    "T",
    "RH",
    "SP",
    "h_sin",
    "h_cos",
    "dw_sin",
    "dw_cos",
    "dy_sin",
    "dy_cos",
    "weekend",
    "demand",
]
FUT_CHANNELS = HIST_CHANNELS[:10]  # demand not known in the future
N_HIST, N_FUT = len(HIST_CHANNELS), len(FUT_CHANNELS)
IDX_T, IDX_DEMAND = 0, 10

CDH_BASE = 25.0  # comfort threshold (deg C) used to build cooling-degree hours


def load_frame(csv_path):
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%m/%d/%Y %H:%M")
    df = df.set_index("Datetime").sort_index()

    demand = df["Demand(MW)"].astype(float).copy()
    med = demand.rolling(7, center=True, min_periods=3).median()
    spike = (
        (demand < 0.35 * med) | (demand > 1.6 * med) | (demand > 17500)
    ) & med.notna()
    # isolated V-spikes: big jump immediately reversed at the same point
    d1 = demand.diff()
    up_dn = (d1 > 3000) & (d1.shift(-1) < -3000)
    dn_up = (d1 < -3000) & (d1.shift(-1) > 3000)
    spike |= up_dn.fillna(False) | dn_up.fillna(False)
    n_out = int(spike.sum())
    demand[spike] = np.nan
    demand = demand.interpolate(method="linear", limit_direction="both")
    df["demand_clean"] = demand

    # data-quality evidence for the write-up: Generation column mostly duplicates Demand
    n_dup = int((df["Generation(MW)"] == df["Demand(MW)"]).sum())
    df.attrs["gen_eq_dem_rows"] = n_dup
    df.attrs["outlier_demand_rows"] = n_out
    return df


def build_feature_frame(df, stats=None):
    """Return hourly feature matrix (float32), static covariates and per-series stats."""
    idx = df.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()
    feats = pd.DataFrame(
        {
            "T": df[COL_T].to_numpy(float),
            "RH": df[COL_RH].to_numpy(float),
            "SP": df[COL_SP].to_numpy(float),
            "h_sin": np.sin(2 * np.pi * hour / 24),
            "h_cos": np.cos(2 * np.pi * hour / 24),
            "dw_sin": np.sin(2 * np.pi * dow / 7),
            "dw_cos": np.cos(2 * np.pi * dow / 7),
            "dy_sin": np.sin(2 * np.pi * doy / 365.25),
            "dy_cos": np.cos(2 * np.pi * doy / 365.25),
            "weekend": (dow >= 5).astype(float),
            "demand": df["demand_clean"].to_numpy(float),
        }
    )
    static = pd.DataFrame(
        {
            "pop": df["Population"].to_numpy(float),
            "gdp": df["GDP"].to_numpy(float),
            "year": idx.year.to_numpy(float),
        }
    )

    if stats is None:
        tr = (idx >= pd.Timestamp(df.attrs.get("train_start", "2016-01-01"))) & (
            idx <= pd.Timestamp(df.attrs.get("train_end", "2022-12-31 23:00"))
        )
        mean = feats[tr].mean()
        std = feats[tr].std().replace(0.0, 1.0)
        cdh_tr = np.maximum(feats.loc[tr, "T"] - CDH_BASE, 0.0)
        cdh_mean, cdh_std = float(cdh_tr.mean()), float(cdh_tr.std() or 1.0)
        s_mean, s_std = static[tr].mean(), static[tr].std().replace(0.0, 1.0)
        dd = feats.loc[tr, "demand"].diff().dropna().abs()
        ramp_mw = float(np.quantile(dd, 0.999))
        stats = dict(
            mean=mean.to_numpy(float),
            std=std.to_numpy(float),
            cdh_mean=cdh_mean,
            cdh_std=cdh_std,
            static_mean=s_mean.to_numpy(float),
            static_std=s_std.to_numpy(float),
            ramp_mw=ramp_mw,
        )

    X = ((feats.to_numpy(float) - stats["mean"]) / stats["std"]).astype(np.float32)
    S = ((static.to_numpy(float) - stats["static_mean"]) / stats["static_std"]).astype(
        np.float32
    )
    return X, S, stats


def _range_idx(idx, start, end):
    return int(idx.searchsorted(pd.Timestamp(start))), int(
        idx.searchsorted(pd.Timestamp(end), side="right")
    )


def build_dataset(cfg: Config, horizon: int):
    """Zero-copy sliding windows -> dict of train/val/test tensors-as-arrays + meta."""
    df = load_frame(cfg.csv_path)
    df.attrs["train_start"], df.attrs["train_end"] = cfg.train_start, cfg.train_end
    X, S, stats = build_feature_frame(df)
    idx = df.index
    n = len(X)
    L, H = cfg.l_hist, horizon
    total = L + H
    if n < total + 1:
        raise ValueError("not enough rows")

    # sliding window over the (n, N_HIST) matrix -> (n-total+1, total, N_HIST) view, no copy
    from numpy.lib.stride_tricks import sliding_window_view

    full = sliding_window_view(X, total, axis=0).transpose(
        0, 2, 1
    )  # (M, total, C) zero-copy view
    try:
        full.flags.writeable = True  # safe: we never write
    except ValueError:
        pass
    t_origin = np.arange(n - total + 1)  # origin = first history ts
    ok = t_origin + total - 1 < n

    def split2(start, end, stride):
        i0, i1 = _range_idx(idx, start, end)
        sel = t_origin[i0:i1:stride]
        sel = sel[ok[sel]]
        w = full[sel]  # zero-copy strided view
        return dict(
            xp=w[:, :L],
            xf=w[:, L:, :N_FUT],
            y=w[:, L:, IDX_DEMAND],
            s=S[sel],
            t0=idx[sel],
        )

    tr = split2(cfg.train_start, cfg.train_end, 1)
    va = split2(cfg.val_start, cfg.val_end, max(1, horizon))
    te = split2(cfg.test_start, cfg.test_end, max(1, horizon))

    # ramp limit expressed in scaled-target units
    dmean, dstd = stats["mean"][IDX_DEMAND], stats["std"][IDX_DEMAND]
    ramp_scaled = stats["ramp_mw"] / dstd

    # hourly position of each window's predictions (for peak-hour metrics)
    def peak_positions(t0):
        hrs = (t0.hour.to_numpy()[:, None] + np.arange(H)[None, :]) % 24
        return np.isin(hrs, [18, 19, 20, 21, 22])

    for d in (tr, va, te):
        d["peak"] = peak_positions(d["t0"])
        d["hot"] = (
            d["xp"][:, :, IDX_T] * stats["std"][IDX_T] + stats["mean"][IDX_T]
        ) > cfg.hot_threshold

    return dict(
        train=tr,
        val=va,
        test=te,
        stats=stats,
        horizon=H,
        dmean=float(dmean),
        dstd=float(dstd),
        ramp_scaled=float(ramp_scaled),
    )


if __name__ == "__main__":
    cfg = Config()
    ds = build_dataset(cfg, 24)
    for k in ("train", "val", "test"):
        print(k, {n: ds[k][n].shape for n in ("xp", "xf", "y", "s")})
    print(
        "stats:",
        {
            k: (np.round(v, 3) if isinstance(v, np.ndarray) else round(v, 3))
            for k, v in ds["stats"].items()
        },
    )
