import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    csv_path: str = os.path.join(ROOT, "BangladeshData_2016_2024.csv")
    results_dir: str = os.path.join(ROOT, "results")

    l_hist: int = 168
    horizons: tuple = (24, 48, 72, 168)

    hidden: int = 128
    latent: int = 128
    n_res_blocks: int = 2
    dropout: float = 0.1

    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 18
    patience: int = 4
    grad_clip: float = 1.0

    seeds: tuple = (1, 2, 3)

    lam_thermal: float = 1.0
    lam_smooth: float = 1.0
    hot_threshold: float = 25.0
    ramp_quantile: float = 0.999

    train_start: str = "2016-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-12-31"

    device: str = "auto"  # auto -> cuda if available (Kaggle), else cpu
