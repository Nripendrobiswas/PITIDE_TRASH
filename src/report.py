"""Compile results/metrics.csv into paper-ready tables and figures.

Run:  python src/report.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config  # noqa: E402

RES = Config().results_dir
FIGS = os.path.join(RES, "figures")


def main():
    cfg = Config()
    os.makedirs(FIGS, exist_ok=True)
    df = pd.read_csv(os.path.join(cfg.results_dir, "metrics.csv"))
    df = df[df.tag.isin(["full", "tuned"])].drop_duplicates(
        subset=["model", "horizon", "seed"], keep="last"
    )
    # prefer tuned-configuration rows over default-config rows per (model, horizon)
    tuned = set(map(tuple, df[df.tag == "tuned"][["model", "horizon"]].values))
    df = df[
        ~(
            df.apply(
                lambda r: (r["model"], r["horizon"]) in tuned and r["tag"] == "full",
                axis=1,
            )
        )
    ]
    num = df.select_dtypes("number")
    df = pd.concat([df.drop(columns=num.columns), num.round(4)], axis=1)

    agg = (
        df.groupby(["model", "horizon"])
        .agg(
            mae=("test_mae", "mean"),
            mae_sd=("test_mae", "std"),
            rmse=("test_rmse", "mean"),
            mape=("test_mape", "mean"),
            smape=("test_smape", "mean"),
            mae_peak=("test_mae_peak", "mean"),
            negsens=("test_negsens_rate", "mean"),
            ramp_viol=("test_ramp_viol", "mean"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    order = ["seasonal_naive", "lstm", "tide", "pinn", "pg_tide"]
    agg["model"] = pd.Categorical(agg["model"], order, ordered=True)
    agg = agg.sort_values(["horizon", "model"])
    agg.to_csv(os.path.join(RES, "table_main.csv"), index=False)
    print("=== Main results (mean over seeds, 2024 test) ===")
    print(agg.to_string(index=False))

    ab = pd.read_csv(os.path.join(RES, "metrics.csv"))
    ab = ab[ab.tag == "ablation"]
    if len(ab):
        ab = ab.sort_values(["horizon", "lam_thermal", "lam_smooth"])
        ab.to_csv(os.path.join(RES, "table_ablation.csv"), index=False)
        print("\n=== PG-TiDE lambda ablation (seed 1) ===")
        print(
            ab[
                [
                    "horizon",
                    "lam_thermal",
                    "lam_smooth",
                    "test_mae",
                    "test_rmse",
                    "test_negsens_rate",
                    "test_ramp_viol",
                ]
            ].to_string(index=False)
        )

    # figure: accuracy vs physics compliance scatter
    sub = agg.dropna(subset=["negsens", "mae"])
    sub = sub[sub.model != "seasonal_naive"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = {"lstm": "o", "tide": "s", "pinn": "^", "pg_tide": "D"}
    colors = {
        "lstm": "tab:gray",
        "tide": "tab:blue",
        "pinn": "tab:red",
        "pg_tide": "tab:green",
    }
    for h, gg in sub.groupby("horizon"):
        for _, r in gg.iterrows():
            ax.scatter(
                r.negsens * 100,
                r.mae,
                marker=markers[r.model],
                s=60,
                c=colors[r.model],
                edgecolors="k",
                label=f"{r.model} h={h}" if h == sub.horizon.min() else None,
            )
    ax.set_xlabel("negative dDemand/dT rate on hot hours (%)")
    ax.set_ylabel("test MAE (MW)")
    ax.set_title("Accuracy vs physics compliance (lower-left is better)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_accuracy_vs_physics.png"), dpi=130)
    plt.close(fig)

    if len(ab):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for h, gg in ab.groupby("horizon"):
            gg = gg.sort_values(["lam_thermal", "lam_smooth"])
            ax.plot(range(len(gg)), gg.test_mae, "o-", label=f"h={h}")
        ax.set_xticks(range(len(ab[ab.horizon == ab.horizon.min()])))
        ax.set_xticklabels(
            [
                f"({a},{b})"
                for a, b in ab[ab.horizon == ab.horizon.min()][
                    ["lam_thermal", "lam_smooth"]
                ].values
            ],
            fontsize=8,
        )
        ax.set_xlabel(r"$(\lambda_{thermal}, \lambda_{ramp})$")
        ax.set_ylabel("test MAE (MW)")
        ax.set_title("PG-TiDE physics-weight ablation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig6_lambda_ablation.png"), dpi=130)
        plt.close(fig)
    print(f"\ntables -> results/table_main.csv, figures -> {FIGS}/")


if __name__ == "__main__":
    main()
