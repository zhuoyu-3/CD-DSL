import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from .config import CDDSLConfig


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
        train_steps: List[object] = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if cfg.random_erasing_prob > 0.0:
            train_steps.append(
                transforms.RandomErasing(
                    p=float(cfg.random_erasing_prob),
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value=0.0,
                )
            )
        train_transform = transforms.Compose(train_steps)
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


def split_indices_dirichlet_groups(
    targets: np.ndarray,
    counts: Sequence[int],
    alphas: Sequence[float],
    samples_per_client: int,
    num_classes: int,
    min_classes: int,
    rng_seed: int,
) -> Dict[int, np.ndarray]:
    if len(counts) != len(alphas):
        raise ValueError(
            f"heterogeneity_counts (len={len(counts)}) and heterogeneity_alphas "
            f"(len={len(alphas)}) must have the same length."
        )
    total_clients = int(sum(counts))
    total_required = total_clients * samples_per_client
    if total_required > len(targets):
        raise ValueError(
            f"Need {total_required} samples for {total_clients} clients (sum of counts), "
            f"but dataset has {len(targets)}."
        )

    rng = np.random.default_rng(rng_seed)
    shuffled = rng.permutation(len(targets))[:total_required]

    split_map: Dict[int, np.ndarray] = {}
    cursor = 0
    next_client_id = 0
    for n_g, alpha_g in zip(counts, alphas):
        n_g = int(n_g)
        chunk_size = n_g * samples_per_client
        chunk_indices = shuffled[cursor : cursor + chunk_size]
        cursor += chunk_size

        chunk_targets = targets[chunk_indices]
        sub_split = split_indices_dirichlet(
            chunk_targets,
            n_g,
            samples_per_client,
            num_classes,
            float(alpha_g),
            min_classes,
        )
        for sub_id in range(n_g):
            local_idx = sub_split[sub_id]
            split_map[next_client_id] = chunk_indices[local_idx]
            next_client_id += 1

    return split_map


def split_test_indices_for_cross(
    test_size: int, clients: int, cross_size: int, seed: int
) -> Dict[int, np.ndarray]:
    required = clients * cross_size
    if required > test_size:
        raise ValueError(
            f"Need {required} testset samples for cross loaders but testset has {test_size}."
        )
    rng = np.random.default_rng(seed + 1)
    pool = rng.permutation(test_size)[:required]
    return {
        k: pool[k * cross_size : (k + 1) * cross_size]
        for k in range(clients)
    }


def make_client_loaders(
    trainset: Dataset,
    testset: Dataset,
    cfg: CDDSLConfig,
    num_classes: int,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    train_size, cross_size, _ = client_split_sizes(cfg)
    if cfg.data_style == "iid":
        train_split = split_indices_iid(len(trainset), cfg.clients, train_size)
        cross_split = split_test_indices_for_cross(
            len(testset), cfg.clients, cross_size, cfg.seed
        )
    elif cfg.split_type == "diri":
        train_split = split_indices_dirichlet(
            dataset_targets(trainset),
            cfg.clients,
            train_size,
            num_classes,
            cfg.dirichlet_alpha,
            cfg.min_classes,
        )
        cross_split = split_test_indices_for_cross(
            len(testset), cfg.clients, cross_size, cfg.seed
        )
    elif cfg.split_type == "diri_groups":
        if int(sum(cfg.heterogeneity_counts)) != cfg.clients:
            raise ValueError(
                f"sum(heterogeneity_counts)={sum(cfg.heterogeneity_counts)} "
                f"must equal clients={cfg.clients}."
            )
        train_split = split_indices_dirichlet_groups(
            dataset_targets(trainset),
            cfg.heterogeneity_counts,
            cfg.heterogeneity_alphas,
            train_size,
            num_classes,
            cfg.min_classes,
            rng_seed=cfg.seed + 7,
        )
        cross_split = split_indices_dirichlet_groups(
            dataset_targets(testset),
            cfg.heterogeneity_counts,
            cfg.heterogeneity_alphas,
            cross_size,
            num_classes,
            cfg.min_classes,
            rng_seed=cfg.seed + 11,
        )
    else:
        raise ValueError(f"Unsupported split: data_style={cfg.data_style}, split_type={cfg.split_type}")

    pin_memory = str(cfg.device).startswith("cuda")
    train_loaders: List[DataLoader] = []
    cross_loaders: List[DataLoader] = []
    rng = np.random.default_rng(cfg.seed)
    for client_id in range(cfg.clients):
        train_indices = rng.permutation(train_split[client_id]).tolist()
        train_subset = Subset(trainset, train_indices)
        cross_subset = Subset(testset, cross_split[client_id].tolist())
        train_loaders.append(
            DataLoader(
                train_subset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=pin_memory,
            )
        )
        cross_loaders.append(
            DataLoader(
                cross_subset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=pin_memory,
            )
        )
    return train_loaders, cross_loaders
