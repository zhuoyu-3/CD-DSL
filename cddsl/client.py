import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import CDDSLConfig
from .evaluation import evaluate_loss_accuracy, evaluate_state_loss
from .state import StateDict, clone_state, set_state


def train_local(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    cfg: CDDSLConfig,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0
    for _ in range(cfg.local_epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            if cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            total_steps += 1
    return total_loss / max(1, total_steps)


def round_learning_rate(cfg: CDDSLConfig, round_id: int) -> float:
    if cfg.lr_schedule == "cosine":
        if cfg.rounds <= 1:
            return cfg.lr
        progress = round_id / max(1, cfg.rounds - 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine)
    if cfg.lr_decay_every <= 0:
        return max(cfg.lr, cfg.lr_min)
    decay_steps = round_id // cfg.lr_decay_every
    return max(cfg.lr * (cfg.lr_gamma ** decay_steps), cfg.lr_min)


class CDDSLClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        cross_loader: DataLoader,
        cfg: CDDSLConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.cross_loader = cross_loader
        self.cfg = cfg
        self.device = device

        state = clone_state(self.model, device)
        self.local_best_state = copy.deepcopy(state)
        self.local_best_loss = float("inf")
        self.local_best_acc = 0.0
        self.neighbor_best_state = copy.deepcopy(state)
        self.neighbor_best_loss = float("inf")
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov and cfg.momentum > 0,
        )
        self.velocity: StateDict = {}
        for key, value in state.items():
            if torch.is_floating_point(value):
                self.velocity[key] = torch.zeros_like(value)

    def current_state(self) -> StateDict:
        return clone_state(self.model, self.device)

    def set_learning_rate(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update_local_best(self) -> Tuple[float, float]:
        loss, acc = evaluate_loss_accuracy(self.model, self.cross_loader, self.device)
        if loss < self.local_best_loss:
            self.local_best_loss = loss
            self.local_best_acc = acc
            self.local_best_state = self.current_state()
        return loss, acc

    def update_neighbor_best(self, candidate_state: StateDict) -> float:
        current_state = self.current_state()
        loss = evaluate_state_loss(self.model, candidate_state, self.cross_loader, self.device)
        set_state(self.model, current_state)
        if loss < self.neighbor_best_loss:
            self.neighbor_best_loss = loss
            self.neighbor_best_state = copy.deepcopy(candidate_state)
        return loss

    def swarm_step(
        self,
        consensus_state: StateDict,
        neighbor_best_state: StateDict,
        c0: float,
        swarm_enabled: bool,
    ) -> None:
        current = self.current_state()
        next_state: StateDict = {}

        if swarm_enabled:
            r1 = float(torch.rand((), device=self.device).item())
            r2 = float(torch.rand((), device=self.device).item())

        for key, value in current.items():
            if key not in self.velocity:
                next_state[key] = value.clone()
                continue
            consensus_delta = consensus_state[key] - value
            blended = value + self.cfg.consensus_step_size * consensus_delta
            if not swarm_enabled:
                next_state[key] = blended
                continue

            self.velocity[key] = (
                c0 * self.velocity[key]
                + self.cfg.c1 * r1 * (self.local_best_state[key] - blended)
                + self.cfg.c2 * r2 * (neighbor_best_state[key] - blended)
            )
            next_state[key] = blended + self.cfg.velocity_step_size * self.velocity[key]
        set_state(self.model, next_state)

    def decentralized_round(
        self,
        consensus_state: StateDict,
        neighbor_best_state: StateDict,
        c0: float,
        lr: float,
        swarm_enabled: bool,
    ) -> Tuple[float, float, float]:
        self.set_learning_rate(lr)
        self.swarm_step(consensus_state, neighbor_best_state, c0, swarm_enabled)
        train_loss = train_local(
            self.model,
            self.train_loader,
            self.optimizer,
            self.criterion,
            self.cfg,
            self.device,
        )
        val_loss, val_acc = self.update_local_best()
        return train_loss, val_loss, val_acc
