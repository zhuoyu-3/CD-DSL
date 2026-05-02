from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Iterable, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .state import StateDict, average_states, set_state

if TYPE_CHECKING:
    from .client import CDDSLClient


def evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += float(criterion(outputs, labels).item())
            pred = outputs.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        return float("inf"), 0.0
    return total_loss / total, correct / total


def evaluate_state_loss(
    model: nn.Module,
    state: StateDict,
    loader: DataLoader,
    device: torch.device,
) -> float:
    set_state(model, state)
    loss, _acc = evaluate_loss_accuracy(model, loader, device)
    return loss


@torch.no_grad()
def recompute_batch_norm_stats(
    model: nn.Module,
    loaders: Iterable[DataLoader],
    device: torch.device,
    max_batches_per_loader: int,
) -> None:
    bn_layers = [
        module for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_layers:
        return

    saved_momenta = {}
    for layer in bn_layers:
        layer.reset_running_stats()
        saved_momenta[layer] = layer.momentum
        layer.momentum = None

    model.train()
    for loader in loaders:
        for batch_id, (inputs, _labels) in enumerate(loader):
            if max_batches_per_loader > 0 and batch_id >= max_batches_per_loader:
                break
            model(inputs.to(device))

    model.eval()
    for layer, momentum in saved_momenta.items():
        layer.momentum = momentum


def evaluate_decentralized_population(
    clients: Sequence[CDDSLClient],
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    accs = []
    losses = []
    for client in clients:
        loss, acc = evaluate_loss_accuracy(client.model, testloader, device)
        losses.append(loss)
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


def build_consensus_model(
    base_model: nn.Module,
    clients: Sequence[CDDSLClient],
    device: torch.device,
    parameter_keys: Set[str],
    bn_calibration_batches: int,
) -> nn.Module:
    states = [client.current_state() for client in clients]
    state = average_states(
        states,
        [1.0] * len(states),
        parameter_keys=parameter_keys,
        buffer_reference=states[0],
    )
    model = copy.deepcopy(base_model).to(device)
    set_state(model, state)
    recompute_batch_norm_stats(
        model,
        (client.train_loader for client in clients),
        device,
        bn_calibration_batches,
    )
    return model


def evaluate_consensus_model(
    base_model: nn.Module,
    clients: Sequence[CDDSLClient],
    testloader: DataLoader,
    device: torch.device,
    parameter_keys: Set[str],
    bn_calibration_batches: int,
) -> Tuple[float, float]:
    model = build_consensus_model(
        base_model,
        clients,
        device,
        parameter_keys,
        bn_calibration_batches,
    )
    return evaluate_loss_accuracy(model, testloader, device)
