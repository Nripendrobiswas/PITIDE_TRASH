"""Hyperparameter tuning for the demand-forecasting models.

Searches architecture/optimizer settings (and the physics loss weights for
PINN / PG-TiDE) with Optuna TPE when installed, otherwise random search.
The objective is validation MAE (2023) only - the 2024 test split is never
used for selection.

Usage:
  python src/tune.py --model pg_tide --horizon 24 --trials 15
  python src/tune.py --all --horizon 24 --trials 10
Best settings are merged into results/best_params.json (consumed by
`python src/train.py --all --use-tuned`), full trial logs go to
results/tuning_<model>_h<H>.csv.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config  # noqa: E402
import train as T  # noqa: E402

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class RandomTrial:
    """Minimal optuna-compatible suggest API for the random-search fallback."""

    def __init__(self, rng, number):
        self.rng = rng
        self.number = number
        self.params = {}

    def suggest_categorical(self, name, choices):
        v = choices[int(self.rng.integers(len(choices)))]
        self.params[name] = v
        return v

    def suggest_float(self, name, low, high, log=False, step=None):
        if log:
            v = float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        else:
            v = float(self.rng.uniform(low, high))
        self.params[name] = v
        return v

    def suggest_int(self, name, low, high, step=1):
        v = int(self.rng.integers((high - low) // step + 1)) * step + low
        self.params[name] = v
        return v


class RandomSearch:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def optimize(self, n_trials, func, show_progress_bar=False):
        for i in range(n_trials):
            func(RandomTrial(self.rng, i))

    def best_trial(self):
        return None


def search_space(trial, name, tune_epochs):
    hp = {}
    if name == "lstm":
        hp["hidden"] = trial.suggest_categorical("hidden", [32, 64, 96])
        hp["layers"] = trial.suggest_int("layers", 1, 2)
    else:
        hp["hidden"] = trial.suggest_categorical("hidden", [64, 128, 256])
    if name in ("tide", "pg_tide", "pinn"):
        hp["n_res_blocks"] = trial.suggest_int("n_res_blocks", 1, 3)
    hp["latent"] = trial.suggest_categorical("latent", [64, 128, 256])
    hp["dropout"] = round(trial.suggest_float("dropout", 0.0, 0.3), 3)
    hp["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
    hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hp["epochs"] = tune_epochs
    lam_t = lam_s = None
    if name in T.PHYSICS_MODELS:
        lam_t = round(trial.suggest_float("lam_thermal", 0.05, 5.0, log=True), 4)
        lam_s = round(trial.suggest_float("lam_smooth", 0.05, 5.0, log=True), 4)
    return hp, lam_t, lam_s


def tune(cfg, name, horizon, n_trials=10, seed=1, quick=False, epochs=8):
    if name == "seasonal_naive":
        print("seasonal_naive has no hyperparameters - nothing to tune")
        return None
    log_rows = []

    def objective(trial):
        hp, lt, ls = search_space(trial, name, epochs)
        t0 = time.time()
        r = T.run_one(cfg, name, horizon, seed, lam_t=lt or 0.0, lam_s=ls or 0.0,
                      quick=quick, verbose=False, hp=hp)
        row = {"trial": trial.number, "val_mae": round(r["val_mae"], 2),
               "test_mae": round(r["test_mae"], 2), "sec": round(time.time() - t0, 1),
               "lam_thermal": lt, "lam_smooth": ls}
        row.update({f"hp_{k}": v for k, v in hp.items()})
        log_rows.append(row)
        print(f"[{name} h={horizon}] trial {trial.number:3d}  val_MAE={row['val_mae']:8.1f}  "
              f"test_MAE={row['test_mae']:8.1f}  lam=({lt}, {ls})  "
              f"lr={hp['lr']:.2e} hid={hp['hidden']} bs={hp['batch_size']} "
              f"[{row['sec']}s]", flush=True)
        return r["val_mae"]

    if HAS_OPTUNA:
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=1234))
    else:
        print("optuna not installed -> random search fallback")
        study = RandomSearch(seed=1234)
    study.optimize(n_trials, objective) if isinstance(study, RandomSearch) \
        else study.optimize(objective, n_trials=n_trials)

    best = (min(log_rows, key=lambda r: r["val_mae"]) if log_rows else None)
    if best is None:
        return None

    os.makedirs(cfg.results_dir, exist_ok=True)
    import csv
    tpath = os.path.join(cfg.results_dir, f"tuning_{name}_h{horizon}.csv")
    keys = list(log_rows[0])
    with open(tpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(log_rows)

    bp_path = T.BEST_PARAMS_PATH
    bp = json.load(open(bp_path)) if os.path.exists(bp_path) else {}
    hp = {k[3:]: v for k, v in best.items() if k.startswith("hp_")}
    hp.pop("epochs", None)  # final runs use full budget, not the tuning cap
    bp.setdefault(name, {})[str(horizon)] = {
        "val_mae": best["val_mae"], "test_mae": best["test_mae"],
        "lam_thermal": best["lam_thermal"], "lam_smooth": best["lam_smooth"],
        "hp": hp, "quick": quick, "n_trials": n_trials, "optuna": HAS_OPTUNA}
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(bp_path, "w") as f:
        json.dump(bp, f, indent=2)
    print(f"best {name} h={horizon}: val_MAE={best['val_mae']}  "
          f"lam=({best['lam_thermal']}, {best['lam_smooth']})  -> {bp_path}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tide", "pinn", "pg_tide"])
    ap.add_argument("--all", action="store_true", help="tune all trainable models")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=8,
                    help="epoch cap during tuning (final runs use config default)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="subsample data (for testing the tuner itself)")
    a = ap.parse_args()
    cfg = Config()
    names = ["tide", "pg_tide", "pinn", "lstm"] if a.all else [a.model]
    if not names[0]:
        ap.error("choose --model or --all")
    for n in names:
        tune(cfg, n, a.horizon, a.trials, a.seed, a.quick, a.epochs)


if __name__ == "__main__":
    main()
