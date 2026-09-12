# Physics-Guided TiDE (PG-TiDE) for Electricity Demand Forecasting

Research codebase for hourly electricity demand forecasting of the Bangladesh
national grid (2016–2024) integrating **Physics-Informed Neural Networks
(PINN)** concepts into deep sequence forecasting, with a novel **Physics-Guided
TiDE** model as the main contribution. Built with PyTorch; runs on CPU and
CUDA (auto-detected), designed to be cloned and executed on Kaggle GPU.

```
git clone https://github.com/Nripendrobiswas/PITIDE_TRASH.git
```

## Dataset

`BangladeshData_2016_2024.csv` — 78,912 rows, complete hourly grid
(2016-01-01 00:00 → 2024-12-31 23:00, no gaps/duplicates/NaNs).

| column | role |
|---|---|
| `Datetime` | timestamp |
| `Temperature(2m)`, `Rhumadity(2m)`, `SPressure(kPa)` | known-future weather covariates |
| `Population`, `GDP` | static (yearly) covariates |
| `Demand(MW)` | **forecast target** |
| `Generation(MW)` | **excluded** — identical to `Demand(MW)` in 71.0% of rows |

`BangladeshData_2016_2024_GEMINI.csv` is a reduced variant (weather + demand
only) kept for reference; the pipeline reads the full file.

### Data quality (evidence in `results/data_quality.txt`, figures in `results/figures/`)

- 74 isolated sensor glitches (e.g. a single hour at 143 MW inside a 13.5 GW
  series) removed with median-ratio + jump-reversal detection and interpolation.
- Peak demand grows 9.0 GW (2016) → 17.2 GW (2024).
- 64.6% of all hours are above 25 °C (hot regime) — demand is strongly
  thermal-driven, which motivates the physics constraints below.

## Problem setting

Multi-horizon pure forecasting: given the last `L = 168` h of demand and
weather plus calendar/static covariates, predict `H ∈ {24, 48, 72, 168}` h of
demand. Chronological split — train 2016–2022, validation 2023 (model
selection / early stopping), test 2024 (single final evaluation, rolling
non-overlapping windows). Future weather is treated as known (perfect-weather-
forecast setting, standard for short-term load forecasting benchmarks).

## Physics formulation

Aggregate demand has no governing PDE, so physics is imposed through
**soft constraints** (differentiable penalties, `src/physics.py`):

1. **L_thermal** — the grid load must not decrease when it gets hotter in the
   hot regime: `∂Demand/∂T ≥ 0` for `T > 25 °C`. Computed with PyTorch autograd
   w.r.t. the temperature input; cooling-degree-hours `CDH = relu(T − 25)` is
   built *inside* every model forward (`src/models/common.py`), so gradients
   propagate through the CDH path as well.
2. **L_ramp** — hour-to-hour changes must respect the national grid's historical
   ramp-rate limit `R_max` (99.9th percentile of |ΔDemand| on train, ≈ 2.7 GW/h):
   `relu(|Δŷ| − R_max)`.

Total training loss:

```
L = MSE(ŷ, y) + λ_thermal · L_thermal + λ_ramp · L_ramp
```

## Models (`src/models/`)

| name | description |
|---|---|
| `seasonal_naive` | y(t+h) = y(t+h−168) |
| `lstm` | 1–2-layer LSTM baseline with future-covariate projection |
| `tide` | TiDE (Rasul et al., 2023): static covariate encoder, flattened-MLP encoder/decoder with residual blocks, temporal dynamic regressor |
| `pinn` | MLP forecaster trained with the physics losses (PINN-style variational constraint) |
| `pg_tide` | **PG-TiDE (novel)**: TiDE backbone + (a) structural monotone thermal branch `CDH_future · \|w_h\|` that cannot decrease with temperature, and (b) both physics losses in training |

`λ = 0` reduces PG-TiDE to its structural branch only, enabling clean
architecture-vs-loss ablations.

## Repository layout

```
├── BangladeshData_2016_2024.csv      # dataset
├── requirements.txt
├── notebooks/01_eda.ipynb
├── src/
│   ├── config.py                     # all defaults (paths, horizons, λ, training)
│   ├── data_pipeline.py              # cleaning, features, zero-copy windowing, splits
│   ├── physics.py                    # L_thermal (autograd), L_ramp, violation diagnostics
│   ├── models/{common,tide,pg_tide,pnn,baselines}.py
│   ├── train.py                      # unified trainer / rolling test evaluation / matrix runner
│   ├── tune.py                       # Optuna TPE (or random-search) hyperparameter tuning
│   ├── evaluate.py                   # MAE/RMSE/MAPE/sMAPE/peak-MAE, ramp-violation rate
│   ├── eda.py                        # data-quality report + figures
│   └── report.py                     # paper tables + figures from metrics.csv
└── results/                          # generated: metrics.csv, tables, figures, best_params.json
                                      # (untracked; pilot numbers are quoted below)
```

## Setup

```bash
pip install -r requirements.txt
# CPU-only local torch:  pip install torch --index-url https://download.pytorch.org/whl/cpu
# Kaggle: torch with CUDA is preinstalled; device=auto picks it up.
```

GPU support: tested on Kaggle T4; any CUDA GPU works, including P100
(compute capability 6.0, within the sm_50+ range of prebuilt PyTorch CUDA
wheels). The LSTM sensitivity diagnostic runs with cuDNN disabled, so the
"cudnn RNN backward in eval mode" limitation does not apply. Note that
per-seed numbers vary slightly between different GPU models — keep the same
accelerator for the final paper tables.

## Running experiments

```bash
python src/eda.py                                             # report + figures
python src/tune.py --model pg_tide --horizon 24 --trials 20   # per-model tuning (val MAE only)
python src/tune.py --all --horizon 24 --trials 15

python src/train.py --model pg_tide --horizon 24 --seed 1     # single run
python src/train.py --all                                     # full matrix + λ ablation
python src/train.py --all --use-tuned                         # matrix with tuned configs
python src/report.py                                          # tables + figures
```

Notes:
- `--all` is **resumable**: completed `(model, horizon, seed, λ, tag)` combinations
  already present in `results/metrics.csv` are skipped.
- Tuning selection uses **validation MAE (2023) only**; the 2024 test split is
  evaluated once per run and never used for selection.
- Results are appended to `results/metrics.csv` (`--use-tuned` runs are tagged
  `tuned` and preferred by `report.py`); tuned settings are merged into
  `results/best_params.json`.
- Physics weights are search dimensions for `pinn`/`pg_tide`
  (`λ_thermal, λ_ramp ∈ [0.05, 5]`, log).

## Results (Kaggle GPU, tuned configs + `--use-tuned`, 2024 test, mean of 3 seeds)

Test MAE (MW) by horizon — physics models are bolded, tuned λ per model/horizon
in `results/best_params.json`:

| model | h=24 | h=48 | h=72 | h=168 | neg. dD/dT (hot h) |
|---|---|---|---|---|---|
| Seasonal-Naive | 1318–1324 | 1321 | 1324 | 1324 | — |
| LSTM | 741 | 848 | 900 | 1037 | 69–79% |
| TiDE | 703 | 822 | 933 | 950 | 66–89% |
| **PINN** | **531** | **623** | **723** | **908** | 3.7–6.3% |
| **PG-TiDE** | 593 | 717 | 784 | 933 | **1.9–4.8%** |

h=24 detail: PINN MAPE 4.87% / RMSE 774 vs TiDE 6.27% / 936; peak-hour MAE
604 vs 785.

Main findings:

- **PG-TiDE beats vanilla TiDE at every horizon** (−15.7% MAE at 24 h,
  −12.7% at 48 h, −16.0% at 72 h, −1.8% at 168 h); at h=72 physics guidance
  lifts TiDE from below LSTM to well above it.
- **PINN achieves the best accuracy at all horizons** with the lowest
  seed-variance (MAE sd 18–55 vs 35–108 MW), i.e. physics regularization
  also stabilizes generalization.
- Physics terms cut unphysical negative temperature response ~15–45×
  (66–89% → 2–6% of hot hours).
- **λ ablation** (seed 1, `table_ablation.csv`): λ_thermal drives both
  compliance (negsens 0.60 → 0.027 already at λ=1) and the accuracy/stability
  effect; λ_ramp is nearly inert on accuracy because grid ramp violations are
  rare by construction (limit = p99.9 of historical ΔDemand) — it acts as a
  safety prior, which is the honest framing.
- Tuned λ (≈ 0.2/0.7) beats default λ=1 on accuracy; note the accuracy↔
  compliance trade-off: if compliance is the headline, tune with multi-
  objective (val MAE + negsens) instead.

## Reproducing this study end-to-end (Kaggle GPU notebook)

```python
!git clone https://github.com/Nripendrobiswas/PITIDE_TRASH.git
%cd PITIDE_TRASH
!pip install -q optuna
!python src/tune.py --all --horizon 24 --trials 15
!python src/train.py --all --use-tuned
!python src/report.py
```

## Limitations / honest caveats

- "Physics" here is soft-constraint priors (thermal monotonicity, ramp limits),
  not a conservation law — the common formulation in PGNN load-forecasting
  literature; a power-balance term was impossible because the supplied
  `Generation(MW)` column duplicates `Demand(MW)`.
- Weather in the forecast horizon is observed ground truth (no forecast error
  modeled).
- A single country/year grid; external validity beyond Bangladesh's hot,
  AC-dominated load profile is untested.

## Reference

TiDE: Rasul et al., *"TiDE: Time-series Predictive Deep Model for Forecasting
with Exogenous Inputs"*, 2023 (arXiv:2304.08424).
