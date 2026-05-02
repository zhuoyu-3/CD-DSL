import csv
from pathlib import Path
from typing import Dict, List

from .config import CDDSLConfig


METRICS_FIELDNAMES: List[str] = [
    "round",
    "mean_test_acc",
    "consensus_test_acc",
    "cross_acc",
    "neighbor_eval_loss",
    "train_loss",
    "cross_loss",
    "mean_push_sum_mass",
    "mean_self_weight",
    "lr",
    "c0",
    "elapsed_sec",
]


def initialize_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()


def append_metrics_csv(path: Path, row: Dict[str, float]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDNAMES)
        writer.writerow(row)


def heterogeneity_summary(cfg: CDDSLConfig) -> Dict[str, object]:
    if cfg.data_style == "iid":
        return {
            "data_style": cfg.data_style,
            "heterogeneity_evaluation": "iid",
            "concentration_parameter": float("inf"),
        }
    return {
        "data_style": cfg.data_style,
        "heterogeneity_evaluation": "dirichlet_alpha",
        "concentration_parameter": float(cfg.dirichlet_alpha),
    }
