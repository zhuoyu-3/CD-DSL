from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class CDDSLConfig:
    dataset: str = "CIFAR10"
    data_root: str = "./data"
    batch_size: int = 32
    clients: int = 50
    rounds: int = 100
    local_epochs: int = 6
    lr: float = 0.01
    lr_gamma: float = 0.5
    lr_decay_every: int = 10
    lr_min: float = 1e-5
    lr_schedule: str = "step"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 5.0
    data_style: str = "non_iid"
    split_type: str = "diri"
    dirichlet_alpha: float = 5.0
    heterogeneity_counts: Tuple[int, ...] = (20, 15, 10, 5)
    heterogeneity_alphas: Tuple[float, ...] = (0.1, 0.5, 1.0, 10.0)
    min_classes: int = 2
    samples_per_client: int = 1000
    cross_split: float = 0.1
    cross_samples_per_client: int = 200
    topology: str = "random_rate"
    extra_edges: int = 1
    graph_degree: int = 2
    connection_rate: float = 0.7
    topology_max_retries: int = 256
    mixing_objective: str = "eval"
    mixing_temperature: float = 0.5
    mixing_blend: float = 0.5
    mixing_min_self_weight: float = 0.0
    mixing_warmup_rounds: int = 3
    c0: float = 0.05
    c1: float = 0.15
    c2: float = 0.15
    c0_decay: float = 0.95
    swarm_warmup_rounds: int = 5
    consensus_step_size: float = 1.0
    velocity_step_size: float = 0.05
    bn_calibration_batches: int = 2
    norm_kind: str = "group"
    group_norm_groups: int = 8
    random_erasing_prob: float = 0.25
    save_checkpoint: bool = False
    checkpoint_dir: str = "./checkpoints"
    checkpoint_prefix: str = "cddsl"
    save_client_checkpoints: bool = False
    save_metrics: bool = False
    metrics_dir: str = "./results"
    metrics_prefix: str = "cddsl_metrics"
    eval_population_every: int = 5
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 1
    num_workers: int = 0
    download: bool = True
