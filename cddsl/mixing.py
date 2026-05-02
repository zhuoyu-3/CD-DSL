from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import CDDSLConfig
from .evaluation import evaluate_state_loss
from .state import StateDict, average_states
from .topology import incoming_neighbors, neighbor_support

if TYPE_CHECKING:
    from .client import CDDSLClient


def softmax_from_losses(losses: Sequence[float], temperature: float) -> np.ndarray:
    tau = max(float(temperature), 1e-8)
    values = -np.array(losses, dtype=np.float64) / tau
    values -= np.max(values)
    weights = np.exp(values)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(losses), dtype=np.float64) / len(losses)
    return weights / total


def clamp_probability(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def warmup_mixing_stats(P: np.ndarray) -> Dict[str, float]:
    return {
        "mean_neighbor_eval_loss": float("nan"),
        "mean_self_weight": float(np.mean(np.diag(P))),
    }


def push_sum_consensus(
    states: Sequence[StateDict],
    masses: Sequence[float],
    P: np.ndarray,
    parameter_keys: Set[str],
) -> Tuple[List[StateDict], List[float]]:
    next_states: List[StateDict] = []
    next_masses: List[float] = []
    clients = len(states)

    for dst in range(clients):
        srcs = incoming_neighbors(P, dst)
        weighted_masses = [P[src, dst] * masses[src] for src in srcs]
        mass = float(sum(weighted_masses))
        if mass <= 0:
            raise RuntimeError(f"Non-positive push-sum mass at node {dst}.")
        next_states.append(
            average_states(
                [states[src] for src in srcs],
                weighted_masses,
                parameter_keys=parameter_keys,
                buffer_reference=states[dst],
            )
        )
        next_masses.append(mass)
    return next_states, next_masses


def optimize_mixing_by_local_evaluation(
    base_model: nn.Module,
    current_states: Sequence[StateDict],
    clients: Sequence[CDDSLClient],
    A: np.ndarray,
    base_P: np.ndarray,
    cfg: CDDSLConfig,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if cfg.mixing_objective == "uniform":
        return base_P.copy(), {
            "mean_neighbor_eval_loss": float("nan"),
            "mean_self_weight": float(np.mean(np.diag(base_P))),
        }

    scratch_model = copy.deepcopy(base_model).to(device)
    P_eval = np.zeros_like(base_P, dtype=np.float64)
    mean_losses: List[float] = []

    for dst in range(A.shape[0]):
        srcs = neighbor_support(A, dst)
        losses = [
            evaluate_state_loss(
                scratch_model,
                current_states[src],
                clients[dst].cross_loader,
                device,
            )
            for src in srcs
        ]
        weights = softmax_from_losses(losses, cfg.mixing_temperature)

        if cfg.mixing_min_self_weight > 0 and dst in srcs:
            self_pos = srcs.index(dst)
            min_self = clamp_probability(cfg.mixing_min_self_weight)
            weights *= 1.0 - min_self
            weights[self_pos] += min_self

        for src, weight in zip(srcs, weights):
            P_eval[src, dst] = float(weight)
        mean_losses.append(float(np.dot(weights, np.array(losses, dtype=np.float64))))

    blend = clamp_probability(cfg.mixing_blend)
    P = blend * P_eval + (1.0 - blend) * base_P
    P /= P.sum(axis=0, keepdims=True)
    return P, {
        "mean_neighbor_eval_loss": float(np.mean(mean_losses)),
        "mean_self_weight": float(np.mean(np.diag(P))),
    }
