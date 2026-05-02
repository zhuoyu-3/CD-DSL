from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np
import torch

from .config import CDDSLConfig
from .metrics import heterogeneity_summary
from .state import StateDict, clone_state

if TYPE_CHECKING:
    from .client import CDDSLClient


def checkpoint_metadata(
    cfg: CDDSLConfig,
    classes: Sequence[str],
    A: np.ndarray,
    base_P: np.ndarray,
    P: np.ndarray,
    masses: Sequence[float],
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    return {
        "config": asdict(cfg),
        "classes": list(classes),
        "adjacency": torch.as_tensor(A, dtype=torch.float64),
        "base_mixing": torch.as_tensor(base_P, dtype=torch.float64),
        "last_mixing": torch.as_tensor(P, dtype=torch.float64),
        "push_sum_masses": torch.as_tensor(masses, dtype=torch.float64),
        "heterogeneity": heterogeneity_summary(cfg),
        "history": copy.deepcopy(history),
        "metrics": metrics,
    }


def save_consensus_checkpoint(
    path: Path,
    model_state: StateDict,
    metadata: Dict[str, object],
    model_kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "model_kind": model_kind,
            "model_state_dict": model_state,
        },
        path,
    )


def save_client_checkpoint(
    path: Path,
    clients: Sequence[CDDSLClient],
    metadata: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "model_kind": "client_population",
            "client_state_dicts": [
                clone_state(client.model, torch.device("cpu")) for client in clients
            ],
            "local_best_state_dicts": [
                {key: value.detach().clone().cpu() for key, value in client.local_best_state.items()}
                for client in clients
            ],
            "neighbor_best_state_dicts": [
                {key: value.detach().clone().cpu() for key, value in client.neighbor_best_state.items()}
                for client in clients
            ],
        },
        path,
    )
