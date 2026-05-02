from typing import Dict, Optional, Sequence, Set

import torch
import torch.nn as nn


StateDict = Dict[str, torch.Tensor]


def model_parameter_keys(model: nn.Module) -> Set[str]:
    return {name for name, _ in model.named_parameters()}


def clone_state(model: nn.Module, device: torch.device = torch.device("cpu")) -> StateDict:
    return {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}


def set_state(model: nn.Module, state: StateDict) -> None:
    model.load_state_dict(state, strict=True)


def average_states(
    states: Sequence[StateDict],
    weights: Sequence[float],
    parameter_keys: Optional[Set[str]] = None,
    buffer_reference: Optional[StateDict] = None,
) -> StateDict:
    if len(states) != len(weights):
        raise ValueError("states and weights must have the same length.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive sum.")

    out: StateDict = {}
    for key, tensor in states[0].items():
        if parameter_keys is not None and key not in parameter_keys:
            source = buffer_reference if buffer_reference is not None else states[0]
            out[key] = source[key].clone()
        elif torch.is_floating_point(tensor):
            buff = torch.zeros_like(tensor)
            for state, weight in zip(states, weights):
                buff.add_(state[key], alpha=float(weight) / total)
            out[key] = buff
        else:
            out[key] = tensor.clone()
    return out
