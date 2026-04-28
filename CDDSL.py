# -*- coding: utf-8 -*-


import argparse
import csv
import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models


StateDict = Dict[str, torch.Tensor]


@dataclass
class CDDSLConfig:
    dataset: str = "CIFAR10"
    data_root: str = "./data"
    batch_size: int = 64
    clients: int = 50
    rounds: int = 100
    local_epochs: int = 4
    lr: float = 0.01
    lr_gamma: float = 0.5
    lr_decay_every: int = 10
    lr_min: float = 1e-4
    lr_schedule: str = "cosine"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 5.0
    data_style: str = "non_iid"
    split_type: str = "diri"
    dirichlet_alpha: float = 5.0
    min_classes: int = 2
    samples_per_client: int = 512
    cross_split: float = 0.1
    cross_samples_per_client: int = 128
    topology: str = "directed_ring"
    extra_edges: int = 1
    graph_degree: int = 2
    connection_rate: float = 0.1
    topology_max_retries: int = 256
    mixing_objective: str = "eval"
    mixing_temperature: float = 0.9
    mixing_blend: float = 0.5
    mixing_min_self_weight: float = 0.2
    mixing_warmup_rounds: int = 3
    c0: float = 0.1
    c1: float = 0.4
    c2: float = 0.4
    c0_decay: float = 0.98
    swarm_warmup_rounds: int = 5
    consensus_step_size: float = 0.5
    velocity_step_size: float = 0.1
    bn_calibration_batches: int = 2
    norm_kind: str = "group"
    group_norm_groups: int = 8
    save_checkpoint: bool = False
    checkpoint_dir: str = "./checkpoints"
    checkpoint_prefix: str = "cddsl"
    save_client_checkpoints: bool = False
    save_metrics: bool = False
    metrics_dir: str = "./results"
    metrics_prefix: str = "cddsl_metrics"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 1
    num_workers: int = 0
    download: bool = True


class MNISTNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def resolve_group_count(num_channels: int, requested_groups: int) -> int:
    groups = max(1, min(int(requested_groups), int(num_channels)))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def build_norm2d(num_channels: int, norm_kind: str, group_norm_groups: int) -> nn.Module:
    norm_kind = norm_kind.lower()
    if norm_kind == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_kind == "group":
        return nn.GroupNorm(resolve_group_count(num_channels, group_norm_groups), num_channels)
    raise ValueError(f"Unsupported norm kind: {norm_kind}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(cfg: CDDSLConfig) -> Tuple[Dataset, Dataset, Sequence[str]]:
    name = cfg.dataset.upper()
    if name == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainset = torchvision.datasets.MNIST(
            root=cfg.data_root, train=True, download=cfg.download, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=cfg.data_root, train=False, download=cfg.download, transform=transform
        )
        classes = tuple(str(i) for i in range(10))
    elif name == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=cfg.data_root, train=True, download=cfg.download, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=cfg.data_root, train=False, download=cfg.download, transform=eval_transform
        )
        classes = (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    return trainset, testset, classes


def build_model(cfg: CDDSLConfig) -> nn.Module:
    if cfg.dataset.upper() == "MNIST":
        return MNISTNet()
    if cfg.norm_kind.lower() == "group":
        net = models.resnet18(
            weights=None,
            norm_layer=lambda channels: nn.GroupNorm(
                resolve_group_count(channels, cfg.group_norm_groups),
                channels,
            ),
        )
    else:
        net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, 10)
    return net


def train_eval_dataset(trainset: Dataset, cfg: CDDSLConfig) -> Dataset:
    evalset = copy.copy(trainset)
    if cfg.dataset.upper() == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        evalset.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return evalset


def dataset_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    labels = []
    for _, label in dataset:
        labels.append(int(label))
    return np.array(labels)


def split_indices_iid(total_size: int, clients: int, samples_per_client: int) -> Dict[int, np.ndarray]:
    required = clients * samples_per_client
    if required > total_size:
        raise ValueError(
            f"Need {required} samples for clients, but dataset has {total_size}."
        )
    shuffled = np.random.permutation(total_size)[:required]
    return {
        k: shuffled[k * samples_per_client : (k + 1) * samples_per_client]
        for k in range(clients)
    }


def client_split_sizes(cfg: CDDSLConfig) -> Tuple[int, int, int]:
    train_size = int(cfg.samples_per_client)
    cross_size = (
        int(cfg.cross_samples_per_client)
        if cfg.cross_samples_per_client > 0
        else max(1, int(cfg.samples_per_client * cfg.cross_split))
    )
    return train_size, cross_size, train_size + cross_size


def split_indices_dirichlet(
    targets: np.ndarray,
    clients: int,
    samples_per_client: int,
    num_classes: int,
    alpha: float,
    min_classes: int,
) -> Dict[int, np.ndarray]:
    """Standard class-wise Dirichlet split without client-order bias."""

    required = clients * samples_per_client
    if required > len(targets):
        raise ValueError(
            f"Need {required} samples for clients, but dataset has {len(targets)}."
        )

    alpha_vec = np.full(clients, float(alpha), dtype=np.float64)
    for _attempt in range(200):
        idx_batch: List[List[int]] = [[] for _ in range(clients)]

        for cls in range(num_classes):
            idx_c = np.where(targets == cls)[0]
            idx_c = np.random.permutation(idx_c)
            proportions = np.random.dirichlet(alpha_vec)
            split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            class_splits = np.split(idx_c, split_points)
            idx_batch = [
                client_indices + split.tolist()
                for client_indices, split in zip(idx_batch, class_splits)
            ]

        oversized = [client_id for client_id, indices in enumerate(idx_batch) if len(indices) > samples_per_client]
        undersized = [client_id for client_id, indices in enumerate(idx_batch) if len(indices) < samples_per_client]

        while oversized and undersized:
            donor = max(oversized, key=lambda client_id: len(idx_batch[client_id]))
            receiver = min(undersized, key=lambda client_id: len(idx_batch[client_id]))
            donor_extra = len(idx_batch[donor]) - samples_per_client
            receiver_need = samples_per_client - len(idx_batch[receiver])
            move_count = min(donor_extra, receiver_need)

            donor_indices = np.random.permutation(idx_batch[donor]).tolist()
            moved = donor_indices[:move_count]
            kept = donor_indices[move_count:]
            idx_batch[donor] = kept
            idx_batch[receiver].extend(moved)

            oversized = [
                client_id for client_id, indices in enumerate(idx_batch)
                if len(indices) > samples_per_client
            ]
            undersized = [
                client_id for client_id, indices in enumerate(idx_batch)
                if len(indices) < samples_per_client
            ]

        split_map: Dict[int, np.ndarray] = {}
        success = True
        for client_id, indices in enumerate(idx_batch):
            if len(indices) < samples_per_client:
                success = False
                break
            chosen = np.random.permutation(indices)[:samples_per_client]
            if np.unique(targets[chosen]).size < min_classes:
                success = False
                break
            split_map[client_id] = chosen

        if success:
            return split_map

    raise RuntimeError(
        "Dirichlet split failed. Try larger alpha, fewer clients, or fewer "
        "samples per client."
    )


def make_client_loaders(
    trainset: Dataset,
    cfg: CDDSLConfig,
    num_classes: int,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    evalset = train_eval_dataset(trainset, cfg)
    train_size, cross_size, total_samples_per_client = client_split_sizes(cfg)
    if cfg.data_style == "iid":
        split_map = split_indices_iid(len(trainset), cfg.clients, total_samples_per_client)
    elif cfg.split_type == "diri":
        split_map = split_indices_dirichlet(
            dataset_targets(trainset),
            cfg.clients,
            total_samples_per_client,
            num_classes,
            cfg.dirichlet_alpha,
            cfg.min_classes,
        )
    else:
        raise ValueError(f"Unsupported split: data_style={cfg.data_style}, split_type={cfg.split_type}")

    train_loaders: List[DataLoader] = []
    cross_loaders: List[DataLoader] = []
    rng = np.random.default_rng(cfg.seed)
    for client_id in range(cfg.clients):
        shuffled_indices = rng.permutation(split_map[client_id]).tolist()
        train_subset = Subset(trainset, shuffled_indices[:train_size])
        cross_subset = Subset(
            evalset,
            shuffled_indices[train_size : train_size + cross_size],
        )
        train_loaders.append(
            DataLoader(
                train_subset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
            )
        )
        cross_loaders.append(
            DataLoader(
                cross_subset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
            )
        )
    return train_loaders, cross_loaders


def graph_neighbors(A: np.ndarray, node: int) -> List[int]:
    return [dst for dst in range(A.shape[1]) if node != dst and A[node, dst] > 0]


def reverse_graph_neighbors(A: np.ndarray, node: int) -> List[int]:
    return [src for src in range(A.shape[0]) if node != src and A[src, node] > 0]


def reachable_nodes(A: np.ndarray, start: int, reverse: bool = False) -> Set[int]:
    visit = reverse_graph_neighbors if reverse else graph_neighbors
    seen: Set[int] = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in visit(A, node):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def is_strongly_connected(A: np.ndarray) -> bool:
    clients = A.shape[0]
    if clients <= 1:
        return True
    return (
        len(reachable_nodes(A, 0, reverse=False)) == clients
        and len(reachable_nodes(A, 0, reverse=True)) == clients
    )


def minimum_connection_rate(clients: int) -> float:
    if clients <= 1:
        return 0.0
    # Conservative heuristic above the directed Erdős-Rényi connectivity
    # threshold to avoid spending many retries on nearly disconnected graphs.
    return float(min(1.0, max(1.0 / (clients - 1), 1.5 * np.log(clients) / clients)))


def build_adjacency(
    clients: int,
    topology: str,
    extra_edges: int,
    graph_degree: int,
    connection_rate: float,
    topology_max_retries: int,
) -> np.ndarray:
    """Return A[src, dst] = 1 if src can send to dst.

    The diagonal is initialized to 1 so each client always keeps a self-loop
    for its own historical experience.
    """

    topology = topology.lower()
    max_retries = max(1, int(topology_max_retries))
    random_topology = topology in {"random_degree", "random_rate"}
    effective_rate = float(min(1.0, max(0.0, connection_rate)))
    if topology == "random_rate":
        effective_rate = max(effective_rate, minimum_connection_rate(clients))

    for _attempt in range(max_retries):
        A = np.eye(clients, dtype=np.float64)
        if topology == "directed_ring":
            for src in range(clients):
                A[src, (src + 1) % clients] = 1.0
                for hop in range(2, extra_edges + 2):
                    A[src, (src + hop) % clients] = 1.0
        elif topology == "bidirectional_ring":
            for src in range(clients):
                for hop in range(1, extra_edges + 2):
                    A[src, (src + hop) % clients] = 1.0
                    A[src, (src - hop) % clients] = 1.0
        elif topology == "fully_connected":
            A[:] = 1.0
        elif topology == "random_degree":
            degree = max(0, min(int(graph_degree), clients - 1))
            for src in range(clients):
                chosen = set()
                if degree > 0:
                    ring_dst = (src + 1) % clients
                    chosen.add(ring_dst)
                    A[src, ring_dst] = 1.0

                remaining = degree - len(chosen)
                if remaining <= 0:
                    continue

                candidates = [
                    dst for dst in range(clients)
                    if dst != src and dst not in chosen
                ]
                sampled = np.random.choice(candidates, size=remaining, replace=False)
                for dst in sampled:
                    A[src, int(dst)] = 1.0
        elif topology == "random_rate":
            for src in range(clients):
                for dst in range(clients):
                    if dst == src:
                        continue
                    if np.random.rand() < effective_rate:
                        A[src, dst] = 1.0
        else:
            raise ValueError(f"Unsupported topology: {topology}")

        if is_strongly_connected(A):
            return A
        if not random_topology:
            break

    if topology == "random_rate":
        raise RuntimeError(
            "Failed to sample a strongly connected random_rate graph after "
            f"{max_retries} retries. Increase --connection-rate above "
            f"{minimum_connection_rate(clients):.4f} or raise --topology-max-retries."
        )
    if topology == "random_degree":
        raise RuntimeError(
            "Failed to sample a strongly connected random_degree graph after "
            f"{max_retries} retries. Increase --graph-degree or raise "
            "--topology-max-retries."
        )
    raise RuntimeError(f"Topology {topology} is not strongly connected.")


def column_stochastic_mixing(A: np.ndarray) -> np.ndarray:
    """Column-stochastic P[src, dst] for push-sum receive aggregation.

    The raw mixing matrix keeps P[i, i] = 1 before normalization, matching the
    CD-DSL self-experience term. The returned matrix is normalized by columns
    for push-sum consensus on directed graphs.
    """

    P = A.astype(np.float64).copy()
    np.fill_diagonal(P, 1.0)
    col_sums = P.sum(axis=0, keepdims=True)
    if np.any(col_sums == 0):
        raise ValueError("Each node must have at least one in-neighbor.")
    return P / col_sums


def incoming_neighbors(P: np.ndarray, node: int) -> List[int]:
    return [src for src in range(P.shape[0]) if P[src, node] > 0]


def neighbor_support(A: np.ndarray, node: int) -> List[int]:
    return [src for src in range(A.shape[0]) if A[src, node] > 0]


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


def push_sum_consensus(
    states: Sequence[StateDict],
    masses: Sequence[float],
    P: np.ndarray,
    parameter_keys: Set[str],
) -> Tuple[List[StateDict], List[float]]:
    """One push-sum receive step: (W P) / (z P), z <- z P."""

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


def softmax_from_losses(losses: Sequence[float], temperature: float) -> np.ndarray:
    tau = max(float(temperature), 1e-8)
    values = -np.array(losses, dtype=np.float64) / tau
    values -= np.max(values)
    weights = np.exp(values)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(losses), dtype=np.float64) / len(losses)
    return weights / total


def optimize_mixing_by_local_evaluation(
    base_model: nn.Module,
    current_states: Sequence[StateDict],
    clients: Sequence["CDDSLClient"],
    A: np.ndarray,
    base_P: np.ndarray,
    cfg: CDDSLConfig,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Optimize P by local evaluation over one-hop neighbors.

    For each receiver i, only neighbors j with A[j, i] = 1 are considered,
    including j = i from the self-loop. The column P[:, i] is the solution of

        min_p sum_j p[j, i] * f_i(w_j) + tau * sum_j p[j, i] log p[j, i]

    subject to p[j, i] >= 0, sum_j p[j, i] = 1, and graph support. This keeps
    the push-sum matrix column-stochastic while optimizing aggregation toward
    models that perform well on receiver i's local validation dataset.
    """

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


def clamp_probability(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


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

        for key, value in current.items():
            if key not in self.velocity:
                next_state[key] = value.clone()
                continue
            consensus_delta = consensus_state[key] - value
            blended = value + self.cfg.consensus_step_size * consensus_delta
            if not swarm_enabled:
                next_state[key] = blended
                continue

            r1 = torch.rand_like(value)
            r2 = torch.rand_like(value)
            self.velocity[key] = (
                c0 * self.velocity[key]
                + self.cfg.c1 * r1 * (self.local_best_state[key] - value)
                + self.cfg.c2 * r2 * (neighbor_best_state[key] - value)
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


def warmup_mixing_stats(P: np.ndarray) -> Dict[str, float]:
    return {
        "mean_neighbor_eval_loss": float("nan"),
        "mean_self_weight": float(np.mean(np.diag(P))),
    }


METRICS_FIELDNAMES = [
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


def run_cd_dsl(cfg: CDDSLConfig) -> Dict[str, List[float]]:
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    trainset, testset, classes = load_dataset(cfg)
    testloader = DataLoader(
        testset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    train_loaders, cross_loaders = make_client_loaders(
        trainset, cfg, num_classes=len(classes)
    )

    A = build_adjacency(
        cfg.clients,
        cfg.topology,
        cfg.extra_edges,
        cfg.graph_degree,
        cfg.connection_rate,
        cfg.topology_max_retries,
    )
    base_P = column_stochastic_mixing(A)
    P = base_P.copy()
    heterogeneity = heterogeneity_summary(cfg)

    base_model = build_model(cfg)
    parameter_keys = model_parameter_keys(base_model)
    clients = [
        CDDSLClient(
            client_id=k,
            model=copy.deepcopy(base_model),
            train_loader=train_loaders[k],
            cross_loader=cross_loaders[k],
            cfg=cfg,
            device=device,
        )
        for k in range(cfg.clients)
    ]

    for client in clients:
        client.update_local_best()
        client.update_neighbor_best(client.current_state())

    masses = [1.0 for _ in range(cfg.clients)]
    history = {
        "mean_client_test_acc": [],
        "consensus_test_acc": [],
        "mean_cross_acc": [],
        "mean_cross_loss": [],
        "mean_train_loss": [],
        "mean_push_sum_mass": [],
        "mean_neighbor_eval_loss": [],
        "mean_self_weight": [],
    }
    checkpoint_enabled = cfg.save_checkpoint or cfg.save_client_checkpoints
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_run_id = (
        f"{cfg.checkpoint_prefix}_{cfg.dataset.lower()}_"
        f"{run_timestamp}_seed{cfg.seed}"
        if checkpoint_enabled
        else ""
    )
    metrics_path: Optional[Path] = None
    if cfg.save_metrics:
        metrics_run_id = (
            f"{cfg.metrics_prefix}_{cfg.dataset.lower()}_"
            f"{run_timestamp}_seed{cfg.seed}"
        )
        metrics_path = Path(cfg.metrics_dir) / f"{metrics_run_id}.csv"
        initialize_metrics_csv(metrics_path)
    best_consensus_acc = float("-inf")
    best_consensus_state: Optional[StateDict] = None
    final_consensus_state: Optional[StateDict] = None
    best_checkpoint_metrics: Dict[str, float] = {}
    final_checkpoint_metrics: Dict[str, float] = {}

    print("\n====================== CD-DSL Settings ======================")
    print(f"dataset={cfg.dataset}, clients={cfg.clients}, rounds={cfg.rounds}")
    print(f"device={device}")
    print(
        f"data_style={cfg.data_style}, split={cfg.split_type}, "
        f"heterogeneity_eval={heterogeneity['heterogeneity_evaluation']}, "
        f"concentration={heterogeneity['concentration_parameter']}"
    )
    print(
        f"train_samples_per_client={cfg.samples_per_client}, "
        f"cross_samples_per_client={client_split_sizes(cfg)[1]}"
    )
    print(
        f"topology={cfg.topology}, extra_edges={cfg.extra_edges}, "
        f"graph_degree={cfg.graph_degree}, connection_rate={cfg.connection_rate}, "
        f"topology_max_retries={cfg.topology_max_retries}"
    )
    print(
        f"mixing_objective={cfg.mixing_objective}, "
        f"temperature={cfg.mixing_temperature}, blend={cfg.mixing_blend}, "
        f"min_self={cfg.mixing_min_self_weight}"
    )
    print(
        f"lr={cfg.lr}, schedule={cfg.lr_schedule}, lr_gamma={cfg.lr_gamma}, "
        f"lr_decay_every={cfg.lr_decay_every}, lr_min={cfg.lr_min}"
    )
    print(
        f"momentum={cfg.momentum}, weight_decay={cfg.weight_decay}, "
        f"nesterov={cfg.nesterov}, label_smoothing={cfg.label_smoothing}, "
        f"grad_clip={cfg.gradient_clip_norm}"
    )
    print(
        f"norm_kind={cfg.norm_kind}, group_norm_groups={cfg.group_norm_groups}, "
        f"model=resnet18"
    )
    print(
        f"mixing_warmup_rounds={cfg.mixing_warmup_rounds}, "
        f"swarm_warmup_rounds={cfg.swarm_warmup_rounds}"
    )
    print(
        f"c0={cfg.c0}, c1={cfg.c1}, c2={cfg.c2}, c0_decay={cfg.c0_decay}, "
        f"consensus_step_size={cfg.consensus_step_size}, "
        f"velocity_step_size={cfg.velocity_step_size}"
    )
    if checkpoint_enabled:
        print(
            f"checkpoint_dir={cfg.checkpoint_dir}, "
            f"checkpoint_run_id={checkpoint_run_id}, "
            f"save_consensus={cfg.save_checkpoint}, "
            f"save_clients={cfg.save_client_checkpoints}"
        )
    if metrics_path is not None:
        print(f"metrics_csv={metrics_path}")
    print("=============================================================\n")

    start = time.time()
    for round_id in range(cfg.rounds):
        c0 = cfg.c0 * (cfg.c0_decay ** round_id)
        lr = round_learning_rate(cfg, round_id)
        current_states = [client.current_state() for client in clients]
        if round_id < cfg.mixing_warmup_rounds:
            P = base_P.copy()
            mixing_stats = warmup_mixing_stats(P)
        else:
            P, mixing_stats = optimize_mixing_by_local_evaluation(
                base_model,
                current_states,
                clients,
                A,
                base_P,
                cfg,
                device,
            )
        consensus_states, masses = push_sum_consensus(
            current_states,
            masses,
            P,
            parameter_keys,
        )

        for client_id, client in enumerate(clients):
            client.update_neighbor_best(consensus_states[client_id])

        train_losses = []
        cross_losses = []
        cross_accs = []
        swarm_enabled = round_id >= cfg.swarm_warmup_rounds
        for client_id, client in enumerate(clients):
            train_loss, cross_loss, cross_acc = client.decentralized_round(
                consensus_states[client_id],
                client.neighbor_best_state,
                c0,
                lr,
                swarm_enabled,
            )
            train_losses.append(train_loss)
            cross_losses.append(cross_loss)
            cross_accs.append(cross_acc)

        _mean_test_loss, mean_test_acc = evaluate_decentralized_population(
            clients, testloader, device
        )
        consensus_model = build_consensus_model(
            base_model,
            clients,
            device,
            parameter_keys,
            cfg.bn_calibration_batches,
        )
        _consensus_loss, consensus_acc = evaluate_loss_accuracy(
            consensus_model,
            testloader,
            device,
        )

        if cfg.save_checkpoint:
            current_checkpoint_metrics = {
                "round": float(round_id + 1),
                "consensus_test_loss": float(_consensus_loss),
                "consensus_test_acc": float(consensus_acc),
                "mean_client_test_loss": float(_mean_test_loss),
                "mean_client_test_acc": float(mean_test_acc),
                "mean_cross_acc": float(np.mean(cross_accs)),
                "mean_cross_loss": float(np.mean(cross_losses)),
            }
            if consensus_acc > best_consensus_acc:
                best_consensus_acc = consensus_acc
                best_consensus_state = clone_state(consensus_model, torch.device("cpu"))
                best_checkpoint_metrics = dict(current_checkpoint_metrics)
            if round_id == cfg.rounds - 1:
                final_consensus_state = clone_state(consensus_model, torch.device("cpu"))
                final_checkpoint_metrics = dict(current_checkpoint_metrics)

        history["mean_client_test_acc"].append(mean_test_acc)
        history["consensus_test_acc"].append(consensus_acc)
        history["mean_cross_acc"].append(float(np.mean(cross_accs)))
        history["mean_cross_loss"].append(float(np.mean(cross_losses)))
        history["mean_train_loss"].append(float(np.mean(train_losses)))
        history["mean_push_sum_mass"].append(float(np.mean(masses)))
        history["mean_neighbor_eval_loss"].append(mixing_stats["mean_neighbor_eval_loss"])
        history["mean_self_weight"].append(mixing_stats["mean_self_weight"])

        if metrics_path is not None:
            append_metrics_csv(
                metrics_path,
                {
                    "round": float(round_id + 1),
                    "mean_test_acc": float(mean_test_acc),
                    "consensus_test_acc": float(consensus_acc),
                    "cross_acc": float(np.mean(cross_accs)),
                    "neighbor_eval_loss": float(mixing_stats["mean_neighbor_eval_loss"]),
                    "train_loss": float(np.mean(train_losses)),
                    "cross_loss": float(np.mean(cross_losses)),
                    "mean_push_sum_mass": float(np.mean(masses)),
                    "mean_self_weight": float(mixing_stats["mean_self_weight"]),
                    "lr": float(lr),
                    "c0": float(c0),
                    "elapsed_sec": float(time.time() - start),
                },
            )

        print(
            f"round {round_id + 1:03d}/{cfg.rounds} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"cross_acc={np.mean(cross_accs):.4f} | "
            f"mean_test_acc={mean_test_acc:.4f} | "
            f"consensus_test_acc={consensus_acc:.4f} | "
            f"neighbor_eval_loss={mixing_stats['mean_neighbor_eval_loss']:.4f} | "
            f"self_w={mixing_stats['mean_self_weight']:.3f} | "
            f"lr={lr:.5f} | c0={c0:.4f}"
        )

    print(f"\nCD-DSL finished in {time.time() - start:.2f} seconds.")
    print("consensus_test_accuracy =", history["consensus_test_acc"])
    print("mean_client_test_accuracy =", history["mean_client_test_acc"])
    if metrics_path is not None:
        print(f"saved metrics csv: {metrics_path}")

    if cfg.save_checkpoint:
        checkpoint_dir = Path(cfg.checkpoint_dir)
        if final_consensus_state is None:
            consensus_model = build_consensus_model(
                base_model,
                clients,
                device,
                parameter_keys,
                cfg.bn_calibration_batches,
            )
            final_consensus_state = clone_state(consensus_model, torch.device("cpu"))
            final_checkpoint_metrics = {
                "round": float(cfg.rounds),
                "consensus_test_acc": float(history["consensus_test_acc"][-1]),
                "mean_client_test_acc": float(history["mean_client_test_acc"][-1]),
            }
        if best_consensus_state is None:
            best_consensus_state = final_consensus_state
            best_checkpoint_metrics = dict(final_checkpoint_metrics)

        best_path = checkpoint_dir / f"{checkpoint_run_id}_best_consensus.pt"
        final_path = checkpoint_dir / f"{checkpoint_run_id}_final_consensus.pt"
        save_consensus_checkpoint(
            best_path,
            best_consensus_state,
            checkpoint_metadata(
                cfg,
                classes,
                A,
                base_P,
                P,
                masses,
                history,
                best_checkpoint_metrics,
            ),
            model_kind="best_consensus",
        )
        save_consensus_checkpoint(
            final_path,
            final_consensus_state,
            checkpoint_metadata(
                cfg,
                classes,
                A,
                base_P,
                P,
                masses,
                history,
                final_checkpoint_metrics,
            ),
            model_kind="final_consensus",
        )
        print(f"saved best consensus checkpoint: {best_path}")
        print(f"saved final consensus checkpoint: {final_path}")

    if cfg.save_client_checkpoints:
        checkpoint_dir = Path(cfg.checkpoint_dir)
        client_path = checkpoint_dir / f"{checkpoint_run_id}_clients.pt"
        save_client_checkpoint(
            client_path,
            clients,
            checkpoint_metadata(
                cfg,
                classes,
                A,
                base_P,
                P,
                masses,
                history,
                final_checkpoint_metrics,
            ),
        )
        print(f"saved client population checkpoint: {client_path}")
    return history


def parse_args() -> CDDSLConfig:
    parser = argparse.ArgumentParser(description="Consensus-based decentralized DSL")
    parser.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clients", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "step"])
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-decay-every", type=int, default=10)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--disable-nesterov", action="store_true")
    parser.add_argument("--data-style", default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--split-type", default="diri", choices=["diri"])
    parser.add_argument("--dirichlet-alpha", type=float, default=5.0)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--samples-per-client", type=int, default=512)
    parser.add_argument("--cross-split", type=float, default=0.1)
    parser.add_argument("--cross-samples-per-client", type=int, default=128)
    parser.add_argument(
        "--topology",
        default="directed_ring",
        choices=[
            "directed_ring",
            "bidirectional_ring",
            "fully_connected",
            "random_degree",
            "random_rate",
        ],
    )
    parser.add_argument("--extra-edges", type=int, default=1)
    parser.add_argument(
        "--graph-degree",
        type=int,
        default=2,
        help="Outgoing random neighbor degree per client, excluding self-loop; used by topology=random_degree.",
    )
    parser.add_argument(
        "--connection-rate",
        type=float,
        default=0.1,
        help="Directed off-diagonal edge rate for topology=random_rate. The effective rate is clamped upward if needed and the graph is resampled until strongly connected.",
    )
    parser.add_argument(
        "--topology-max-retries",
        type=int,
        default=256,
        help="Maximum retries when sampling random topologies until a strongly connected graph is found.",
    )
    parser.add_argument("--mixing-objective", default="eval", choices=["eval", "uniform"])
    parser.add_argument("--mixing-temperature", type=float, default=0.9)
    parser.add_argument("--mixing-blend", type=float, default=0.5)
    parser.add_argument("--mixing-min-self-weight", type=float, default=0.2)
    parser.add_argument("--mixing-warmup-rounds", type=int, default=3)
    parser.add_argument("--c0", type=float, default=0.1)
    parser.add_argument("--c1", type=float, default=0.4)
    parser.add_argument("--c2", type=float, default=0.4)
    parser.add_argument("--c0-decay", type=float, default=0.98)
    parser.add_argument("--swarm-warmup-rounds", type=int, default=5)
    parser.add_argument("--consensus-step-size", type=float, default=0.5)
    parser.add_argument("--velocity-step-size", type=float, default=0.1)
    parser.add_argument("--bn-calibration-batches", type=int, default=2)
    parser.add_argument("--norm-kind", default="group", choices=["batch", "group"])
    parser.add_argument("--group-norm-groups", type=int, default=8)
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save best and final consensus model checkpoints.",
    )
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--checkpoint-prefix", default="cddsl")
    parser.add_argument(
        "--save-client-checkpoints",
        action="store_true",
        help="Also save all final client models. This can be large for many clients.",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save per-round accuracy/loss metrics to CSV.",
    )
    parser.add_argument("--metrics-dir", default="./results")
    parser.add_argument("--metrics-prefix", default="cddsl_metrics")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    cfg = CDDSLConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_gamma=args.lr_gamma,
        lr_decay_every=args.lr_decay_every,
        lr_min=args.lr_min,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=not args.disable_nesterov,
        label_smoothing=args.label_smoothing,
        gradient_clip_norm=args.gradient_clip_norm,
        data_style=args.data_style,
        split_type=args.split_type,
        dirichlet_alpha=args.dirichlet_alpha,
        min_classes=args.min_classes,
        samples_per_client=args.samples_per_client,
        cross_split=args.cross_split,
        cross_samples_per_client=args.cross_samples_per_client,
        topology=args.topology,
        extra_edges=args.extra_edges,
        graph_degree=args.graph_degree,
        connection_rate=args.connection_rate,
        topology_max_retries=args.topology_max_retries,
        mixing_objective=args.mixing_objective,
        mixing_temperature=args.mixing_temperature,
        mixing_blend=args.mixing_blend,
        mixing_min_self_weight=args.mixing_min_self_weight,
        mixing_warmup_rounds=args.mixing_warmup_rounds,
        c0=args.c0,
        c1=args.c1,
        c2=args.c2,
        c0_decay=args.c0_decay,
        swarm_warmup_rounds=args.swarm_warmup_rounds,
        consensus_step_size=args.consensus_step_size,
        velocity_step_size=args.velocity_step_size,
        bn_calibration_batches=args.bn_calibration_batches,
        norm_kind=args.norm_kind,
        group_norm_groups=args.group_norm_groups,
        save_checkpoint=args.save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        save_client_checkpoints=args.save_client_checkpoints,
        save_metrics=args.save_metrics,
        metrics_dir=args.metrics_dir,
        metrics_prefix=args.metrics_prefix,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        download=not args.no_download,
    )
    return cfg


if __name__ == "__main__":
    config = parse_args()
    run_cd_dsl(config)
