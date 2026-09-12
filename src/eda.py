"""Exploratory data analysis + data-quality evidence for the write-up.

Saves figures to results/figures/ and a text report to results/data_quality.txt.
Run:  python src/eda.py
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
from data_pipeline import (CDH_BASE, COL_SP, COL_RH, COL_T,  # noqa: E402
                           load_frame)

FIGS = os.path.join(Config().results_dir, "figures")


def raw_demand_frame(cfg):
    df = pd.read_csv(cfg.csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%m/%d/%Y %H:%M")
    return df.set_index("Datetime").sort_index()


def main():
    os.makedirs(FIGS, exist_ok=True)
    cfg = Config()
    raw = raw_demand_frame(cfg)
    df = load_frame(cfg.csv_path)
    d, traw = df["demand_clean"], raw["Demand(MW)"]

    lines = []
    lines.append("Data quality report - BangladeshData_2016_2024.csv")
    lines.append("=" * 55)
    lines.append(f"rows: {len(raw)}, hourly, {raw.index.min()} .. {raw.index.max()}")
    lines.append(f"missing values: {int(raw.isna().sum().sum())}, duplicated timestamps: {int(raw.index.duplicated().sum())}")
    full = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    lines.append(f"missing hours (gaps): {len(full) - len(raw)}")
    lines.append(f"Generation(MW) identical to Demand(MW): {df.attrs['gen_eq_dem_rows']} rows "
                 f"({df.attrs['gen_eq_dem_rows'] / len(raw) * 100:.1f}%) -> column unusable for a "
                 f"power-balance constraint; modeling uses Demand(MW) only.")
    lines.append(f"demand spike rows cleaned (isolated glitches / implausible values): {df.attrs['outlier_demand_rows']}")
    top = traw.diff().abs().sort_values(ascending=False).head(5)
    lines.append("top raw demand jumps: " + "; ".join(f"{t} = {v:.0f} MW" for t, v in top.items()))
    lines.append(f"temperature: mean {raw[COL_T].mean():.1f} C, max {raw[COL_T].max():.1f} C; "
                 f"hours with T > {CDH_BASE:.0f} C (hot regime): {(raw[COL_T] > CDH_BASE).mean() * 100:.1f}%")
    lines.append(f"annual peak demand (GW): " + ", ".join(
        f"{y.year}: {v / 1000:.2f}" for y, v in d.resample('YE').max().items()))
    lines.append(f"annual min demand (GW): " + ", ".join(
        f"{y.year}: {v / 1000:.2f}" for y, v in d.resample('YE').min().items()))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(cfg.results_dir, "data_quality.txt"), "w") as f:
        f.write(report + "\n")

    # fig1: full series + one glitchy week before/after cleaning
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    ax[0].plot(d.index, d / 1000, lw=0.3)
    ax[0].set_ylabel("Demand (GW)")
    ax[0].set_title("Hourly demand, Bangladesh 2016-2024 (cleaned)")
    t0 = pd.Timestamp("2024-05-01")
    w_raw = traw.loc[t0 - pd.Timedelta("3D"):t0 + pd.Timedelta("3D")]
    w_cln = d.loc[t0 - pd.Timedelta("3D"):t0 + pd.Timedelta("3D")]
    ax[1].plot(w_raw.index, w_raw / 1000, "o-", ms=2, label="raw")
    ax[1].plot(w_cln.index, w_cln / 1000, "-", lw=1.5, label="cleaned")
    ax[1].legend()
    ax[1].set_title("Isolated sensor glitches around 2024-05-01 06:00")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_demand_series.png"), dpi=130)
    plt.close(fig)

    # fig2: demand vs temperature
    s = df[[COL_T, "demand_clean"]].sample(20000, random_state=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(s[COL_T], s["demand_clean"] / 1000, s=2, alpha=0.25, c=s.index.month, cmap="twilight")
    binm = pd.DataFrame({"T": df[COL_T], "d": df["demand_clean"]}).groupby(
        pd.cut(df[COL_T], 40), observed=True)["d"].mean().reset_index(name="d")
    cen = [i.mid for i in binm.iloc[:, 0]]
    ax.plot(cen, binm["d"] / 1000, "k-", lw=2, label="binned mean")
    ax.axvline(CDH_BASE, color="r", ls="--", label=f"T = {CDH_BASE:.0f} C (comfort)")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Demand (GW)")
    ax.legend()
    ax.set_title("Thermal driver of electricity demand")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_demand_vs_temperature.png"), dpi=130)
    plt.close(fig)

    # fig3: month x hour heatmap
    piv = pd.DataFrame({"h": df.index.hour, "m": df.index.month, "d": d}).pivot_table(
        index="m", columns="h", values="d")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(piv / 1000, aspect="auto", cmap="viridis")
    ax.set_xticks(range(0, 24, 2))
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Month")
    fig.colorbar(im, label="Mean demand (GW)")
    ax.set_title("Demand seasonal pattern")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_seasonality.png"), dpi=130)
    plt.close(fig)

    # fig4: generation duplication evidence
    per_year = ((raw["Generation(MW)"] == raw["Demand(MW)"]).groupby(raw.index.year)).mean() * 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    per_year.plot.bar(ax=ax, color="firebrick")
    ax.set_ylabel("% hours")
    ax.set_xlabel("Year")
    ax.set_title("Generation(MW) is an exact copy of Demand(MW) -> excluded")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_gen_duplication.png"), dpi=130)
    plt.close(fig)
    print(f"\nfigures written to {FIGS}/")


if __name__ == "__main__":
    main()
