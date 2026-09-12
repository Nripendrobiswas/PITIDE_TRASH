"""Unified training / evaluation harness for all demand-forecasting models.

Usage:
  python src/train.py --model pg_tide --horizon 24 --seed 1
  python src/train.py --all            # full experiment matrix + ablation
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config  # noqa: E402
import data_pipeline as dp  # noqa: E402
from physics import PhysicsLoss, thermal_violation_rate  # noqa: E402
from evaluate import metrics, ramp_violation_rate, to_mw  # noqa: E402
from models import LSTMModel, PGTiDE, PINNMLP, TiDE  # noqa: E402
from models.baselines import seasonal_naive  # noqa: E402

torch.set_num_threads(os.cpu_count() or 4)
_DS_CACHE = {}
PHYSICS_MODELS = ("pg_tide", "pinn")


def resolve_device(pref="auto"):
    if pref in ("cuda", "cpu"):
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dataset(cfg, horizon, quick=False):
    key = (horizon, quick)
    if key not in _DS_CACHE:
        ds = dp.build_dataset(cfg, horizon)
        if quick:
            for part in ("train", "val", "test"):
                m = min(
                    len(ds[part]["y"]), {"train": 6000, "val": 300, "test": 300}[part]
                )
                ds[part] = {k: v[:m] for k, v in ds[part].items()}
        _DS_CACHE[key] = ds
    return _DS_CACHE[key]


def get_model(name, ds, cfg, hp=None):
    hp = hp or {}
    kw = dict(
        n_hist=dp.N_HIST,
        n_fut=dp.N_FUT,
        n_static=3,
        l_hist=cfg.l_hist,
        h_out=ds["horizon"],
        stats=ds["stats"],
        hidden=hp.get("hidden", 64 if name == "lstm" else cfg.hidden),
        latent=hp.get("latent", hp.get("hidden", cfg.latent)),
        n_res=hp.get("n_res_blocks", cfg.n_res_blocks),
        dropout=hp.get("dropout", cfg.dropout),
    )
    if name == "tide":
        return TiDE(**kw)
    if name == "pg_tide":
        return PGTiDE(**kw)
    if name == "pinn":
        kw.pop("latent")
        return PINNMLP(**kw)
    if name == "lstm":
        kw.pop("latent")
        kw.pop("n_res")
        return LSTMModel(**kw, layers=hp.get("layers", 1))
    raise ValueError(name)


def tensors(split):
    return TensorDataset(
        *[torch.from_numpy(np.asarray(split[k])) for k in ("xp", "xf", "y", "s")]
    )


def run_one(
    cfg,
    name,
    horizon,
    seed,
    lam_t=0.0,
    lam_s=0.0,
    quick=False,
    epochs=None,
    save_preds=False,
    verbose=True,
    hp=None,
):
    hp = hp or {}
    t0 = time.time()
    ds = get_dataset(cfg, horizon, quick)
    row = dict(
        model=name,
        horizon=horizon,
        seed=seed,
        lam_thermal=lam_t,
        lam_smooth=lam_s,
        tag="quick" if quick else "full",
    )
    dmean, dstd = ds["dmean"], ds["dstd"]

    def eval_row(split, pred_scaled):
        y_mw = to_mw(split["y"], dmean, dstd)
        p_mw = to_mw(pred_scaled, dmean, dstd)
        return metrics(y_mw, p_mw, split["peak"])

    if name == "seasonal_naive":
        pred = seasonal_naive(ds["test"]["xp"], horizon)
        row.update({f"test_{k}": v for k, v in eval_row(ds["test"], pred).items()})
        row["train_sec"] = 0.0
        if verbose:
            print(
                f"{name:15s} h={horizon:3d} s={seed}           "
                f"MAE={row['test_mae']:7.1f} RMSE={row['test_rmse']:7.1f} "
                f"MAPE={row['test_mape']:5.2f}%"
            )
        return row

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = resolve_device(cfg.device)
    model = get_model(name, ds, cfg, hp).to(dev)
    phys = (
        PhysicsLoss(lam_t, lam_s, ds["stats"], ds["ramp_scaled"])
        if name in PHYSICS_MODELS
        else None
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=hp.get("lr", cfg.lr),
        weight_decay=hp.get("weight_decay", cfg.weight_decay),
    )
    max_ep = epochs or hp.get("epochs") or (6 if quick else cfg.max_epochs)
    if name == "lstm":
        max_ep = min(max_ep, hp.get("epochs", 10))
    batch = hp.get("batch_size") or (1024 if name == "lstm" else cfg.batch_size)
    loader = DataLoader(tensors(ds["train"]), batch_size=batch, shuffle=True)
    v_xp, v_xf, v_s, v_y = [
        torch.from_numpy(np.asarray(ds["val"][k])).to(dev)
        for k in ("xp", "xf", "s", "y")
    ]
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(max_ep):
        model.train()
        for xb, xfb, yb, sb in loader:
            if dev != "cpu":
                xb, xfb, yb, sb = (
                    xb.to(dev, non_blocking=True),
                    xfb.to(dev, non_blocking=True),
                    yb.to(dev, non_blocking=True),
                    sb.to(dev, non_blocking=True),
                )
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb, xfb, sb), yb)
            if phys is not None and (phys.lam_t > 0 or phys.lam_s > 0):
                loss = loss + phys(model, xb, xfb, sb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            vout = []
            for i in range(0, len(v_xp), 1024):
                vout.append(
                    model(v_xp[i : i + 1024], v_xf[i : i + 1024], v_s[i : i + 1024])
                )
            val_mse = F.mse_loss(torch.cat(vout), v_y).item()
        if val_mse < best_val - 1e-5:
            best_val, bad = val_mse, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vp = []
        for i in range(0, len(v_xp), 1024):
            vp.append(model(v_xp[i : i + 1024], v_xf[i : i + 1024], v_s[i : i + 1024]))
        vp = torch.cat(vp).cpu().numpy()
    row["val_mae"] = metrics(
        to_mw(ds["val"]["y"], dmean, dstd), to_mw(vp, dmean, dstd)
    )["mae"]

    with torch.no_grad():
        tp = []
        te = ds["test"]
        t_xp, t_xf, t_s = [
            torch.from_numpy(np.asarray(te[k])).to(dev) for k in ("xp", "xf", "s")
        ]
        for i in range(0, len(t_xp), 1024):
            tp.append(model(t_xp[i : i + 1024], t_xf[i : i + 1024], t_s[i : i + 1024]))
        pred = torch.cat(tp).cpu().numpy()
    row.update({f"test_{k}": v for k, v in eval_row(te, pred).items()})
    row["test_ramp_viol"] = ramp_violation_rate(
        pred, te["xp"][:, -1, dp.IDX_DEMAND], dstd, ds["stats"]["ramp_mw"]
    )
    idx = np.arange(len(te["y"]))[:3000]
    samp = {k: te[k][idx] for k in ("xp", "xf", "s")}
    row["test_negsens_rate"] = thermal_violation_rate(
        model, samp["xp"], samp["xf"], samp["s"], ds["stats"], device=dev
    )
    if save_preds:
        os.makedirs(cfg.results_dir, exist_ok=True)
        np.savez(
            os.path.join(cfg.results_dir, f"preds_{name}_h{horizon}_s{seed}.npz"),
            y=to_mw(te["y"], dmean, dstd),
            p=to_mw(pred, dmean, dstd),
            t=te["t0"].values.astype("int64"),
        )
    row["train_sec"] = round(time.time() - t0, 1)
    row["val_mse_scaled"] = round(best_val, 5)
    if verbose:
        print(
            f"{name:15s} h={horizon:3d} s={seed} lt={lam_t} ls={lam_s} "
            f"val_MAE={row.get('val_mae', 0):7.1f} MAE={row.get('test_mae', 0):7.1f} "
            f"RMSE={row.get('test_rmse', 0):7.1f} "
            f"MAPE={row.get('test_mape', 0):5.2f}% negsens={row.get('test_negsens_rate', 0):.3f} "
            f"[{row['train_sec']}s]",
            flush=True,
        )
    return row


FIELDNAMES = [
    "model",
    "horizon",
    "seed",
    "lam_thermal",
    "lam_smooth",
    "tag",
    "train_sec",
    "val_mse_scaled",
    "val_mae",
    "test_mae",
    "test_rmse",
    "test_mae_peak",
    "test_mape",
    "test_smape",
    "test_negsens_rate",
    "test_ramp_viol",
]


def append_csv(path, rows):
    import csv

    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})


def done_keys(path):
    import csv

    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                keys.add(
                    (
                        r["model"],
                        int(float(r["horizon"])),
                        int(float(r["seed"])),
                        float(r.get("lam_thermal", 0) or 0),
                        float(r.get("lam_smooth", 0) or 0),
                        r.get("tag", ""),
                    )
                )
            except (KeyError, ValueError):
                continue
    return keys


BEST_PARAMS_PATH = os.path.join(Config().results_dir, "best_params.json")


def load_best_params():
    import json

    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            return json.load(f)
    return {}


def tuned_for(bp, name, horizon):
    entry = {k: v for k, v in bp.get(name, {}).items() if not v.get("quick")}
    if not entry:
        return {}, None, None
    key = (
        str(horizon)
        if str(horizon) in entry
        else min(entry, key=lambda k: abs(int(k) - horizon))
    )
    e = entry[key]
    return e.get("hp", {}), e.get("lam_thermal"), e.get("lam_smooth")


def run_all(cfg, quick=False, use_tuned=False):
    os.makedirs(cfg.results_dir, exist_ok=True)
    out = os.path.join(cfg.results_dir, "metrics.csv")
    done = done_keys(out)
    bp = load_best_params() if use_tuned else {}
    if use_tuned and not bp:
        print(
            "warning: --use-tuned set but results/best_params.json missing; using defaults"
        )
    models = ["seasonal_naive", "lstm", "tide", "pinn", "pg_tide"]
    rows = []
    for h in cfg.horizons:
        for name in models:
            seeds = (1,) if name == "seasonal_naive" else cfg.seeds
            for seed in seeds:
                hp = {}
                lt = cfg.lam_thermal if name in PHYSICS_MODELS else 0.0
                ls = cfg.lam_smooth if name in PHYSICS_MODELS else 0.0
                if name in PHYSICS_MODELS or name == "tide" or name == "lstm":
                    hp, t_lt, t_ls = tuned_for(bp, name, h)
                    if t_lt is not None:
                        lt = t_lt
                    if t_ls is not None:
                        ls = t_ls
                tag = "quick" if quick else ("tuned" if (use_tuned and hp) else "full")
                if (name, h, seed, lt, ls, tag) in done:
                    print(f"skip {name} h={h} s={seed} (already in metrics.csv)")
                    continue
                r = run_one(
                    cfg,
                    name,
                    h,
                    seed,
                    lt,
                    ls,
                    quick,
                    save_preds=(name in ("tide", "pg_tide") and h == 72),
                    hp=hp,
                )
                r["tag"] = tag
                rows.append(r)
                append_csv(out, [r])
    combos = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)]
    for h in (24, 168):
        ab_hp, _, _ = tuned_for(bp, "pg_tide", h)
        for lt, ls in combos:
            if ("pg_tide", h, 1, float(lt), float(ls), "ablation") in done:
                print(f"skip ablation pg_tide h={h} lam=({lt},{ls})")
                continue
            r = run_one(cfg, "pg_tide", h, 1, float(lt), float(ls), quick, hp=ab_hp)
            r["tag"] = "ablation"
            rows.append(r)
            append_csv(out, [r])
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.results_dir, "metrics_run.csv"), index=False)
    if len(df):
        print(
            df[df.tag != "ablation"]
            .pivot_table(index="model", columns="horizon", values="test_mae")
            .round(1)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", choices=["seasonal_naive", "lstm", "tide", "pg_tide", "pinn"]
    )
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lam-t", type=float, default=None)
    ap.add_argument("--lam-s", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--use-tuned",
        action="store_true",
        help="use hyperparameters/lambdas from results/best_params.json (see src/tune.py)",
    )
    a = ap.parse_args()
    if not a.all and not a.model:
        ap.error("--model required unless --all is set")
    cfg = Config()
    if a.all:
        run_all(cfg, quick=a.quick, use_tuned=a.use_tuned)
    else:
        hp = {}
        lt = cfg.lam_thermal if a.lam_t is None else a.lam_t
        ls = cfg.lam_smooth if a.lam_s is None else a.lam_s
        if a.use_tuned:
            hp, t_lt, t_ls = tuned_for(load_best_params(), a.model, a.horizon)
            if a.lam_t is None and t_lt is not None:
                lt = t_lt
            if a.lam_s is None and t_ls is not None:
                ls = t_ls
            if not hp:
                print("warning: no tuned entry for this model; using defaults")
        r = run_one(
            cfg,
            a.model,
            a.horizon,
            a.seed,
            lt if a.model in PHYSICS_MODELS else 0.0,
            ls if a.model in PHYSICS_MODELS else 0.0,
            a.quick,
            a.epochs,
            hp=hp,
        )
        os.makedirs(cfg.results_dir, exist_ok=True)
        append_csv(os.path.join(cfg.results_dir, "metrics.csv"), [r])


if __name__ == "__main__":
    main()
