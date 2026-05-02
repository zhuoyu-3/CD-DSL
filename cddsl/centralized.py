import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import data as _data
from .checkpoints import (
    checkpoint_metadata,
    save_consensus_checkpoint,
)
from .client import round_learning_rate
from .config import CDDSLConfig
from .evaluation import evaluate_loss_accuracy
from .metrics import (
    append_metrics_csv,
    initialize_metrics_csv,
)
from .model import build_model
from .state import StateDict, clone_state
from .utils import seed_everything


def run_centralized(cfg: CDDSLConfig) -> Dict[str, List[float]]:
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    trainset, testset, classes = _data.load_dataset(cfg)
    pin_memory = str(cfg.device).startswith("cuda")
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov and cfg.momentum > 0,
    )

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

    checkpoint_enabled = cfg.save_checkpoint
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
    best_test_acc = float("-inf")
    best_state: Optional[StateDict] = None
    final_state: Optional[StateDict] = None
    best_checkpoint_metrics: Dict[str, float] = {}
    final_checkpoint_metrics: Dict[str, float] = {}

    print("\n=================== Centralized Settings ===================")
    print(f"dataset={cfg.dataset}, train_samples={len(trainset)}, test_samples={len(testset)}")
    print(f"device={device}")
    print(f"epochs={cfg.rounds}, batch_size={cfg.batch_size}")
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
        f"model=resnet18, random_erasing_prob={cfg.random_erasing_prob}"
    )
    print(
        "CENTRALIZED: single model trained on the full training set, "
        "no clients, no decentralization."
    )
    if checkpoint_enabled:
        print(
            f"checkpoint_dir={cfg.checkpoint_dir}, "
            f"checkpoint_run_id={checkpoint_run_id}"
        )
    if metrics_path is not None:
        print(f"metrics_csv={metrics_path}")
    print("============================================================\n")

    start = time.time()
    for round_id in range(cfg.rounds):
        lr_round = round_learning_rate(cfg, round_id)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_round

        model.train()
        epoch_loss = 0.0
        n_steps = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            if cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        train_loss = epoch_loss / max(1, n_steps)

        test_loss, test_acc = evaluate_loss_accuracy(model, testloader, device)

        if cfg.save_checkpoint:
            current_metrics = {
                "round": float(round_id + 1),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "train_loss": float(train_loss),
            }
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = clone_state(model, torch.device("cpu"))
                best_checkpoint_metrics = dict(current_metrics)
            if round_id == cfg.rounds - 1:
                final_state = clone_state(model, torch.device("cpu"))
                final_checkpoint_metrics = dict(current_metrics)

        history["mean_client_test_acc"].append(float(test_acc))
        history["consensus_test_acc"].append(float(test_acc))
        history["mean_cross_acc"].append(float("nan"))
        history["mean_cross_loss"].append(float("nan"))
        history["mean_train_loss"].append(float(train_loss))
        history["mean_push_sum_mass"].append(float("nan"))
        history["mean_neighbor_eval_loss"].append(float("nan"))
        history["mean_self_weight"].append(float("nan"))

        if metrics_path is not None:
            append_metrics_csv(
                metrics_path,
                {
                    "round": float(round_id + 1),
                    "mean_test_acc": float(test_acc),
                    "consensus_test_acc": float(test_acc),
                    "cross_acc": float("nan"),
                    "neighbor_eval_loss": float("nan"),
                    "train_loss": float(train_loss),
                    "cross_loss": float("nan"),
                    "mean_push_sum_mass": float("nan"),
                    "mean_self_weight": float("nan"),
                    "lr": float(lr_round),
                    "c0": float("nan"),
                    "elapsed_sec": float(time.time() - start),
                },
            )

        print(
            f"epoch {round_id + 1:03d}/{cfg.rounds} | "
            f"train_loss={train_loss:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"lr={lr_round:.5f}"
        )

    print(f"\nCentralized finished in {time.time() - start:.2f} seconds.")
    print("test_accuracy =", history["consensus_test_acc"])
    if metrics_path is not None:
        print(f"saved metrics csv: {metrics_path}")

    if cfg.save_checkpoint:
        checkpoint_dir = Path(cfg.checkpoint_dir)
        if final_state is None:
            final_state = clone_state(model, torch.device("cpu"))
            final_checkpoint_metrics = {
                "round": float(cfg.rounds),
                "test_acc": float(history["consensus_test_acc"][-1]),
            }
        if best_state is None:
            best_state = final_state
            best_checkpoint_metrics = dict(final_checkpoint_metrics)

        empty_A = np.zeros((1, 1), dtype=np.float64)
        empty_P = np.zeros((1, 1), dtype=np.float64)
        empty_masses = np.array([1.0], dtype=np.float64)

        best_path = checkpoint_dir / f"{checkpoint_run_id}_best.pt"
        final_path = checkpoint_dir / f"{checkpoint_run_id}_final.pt"
        save_consensus_checkpoint(
            best_path,
            best_state,
            checkpoint_metadata(
                cfg, classes, empty_A, empty_P, empty_P, empty_masses,
                history, best_checkpoint_metrics,
            ),
            model_kind="centralized_best",
        )
        save_consensus_checkpoint(
            final_path,
            final_state,
            checkpoint_metadata(
                cfg, classes, empty_A, empty_P, empty_P, empty_masses,
                history, final_checkpoint_metrics,
            ),
            model_kind="centralized_final",
        )
        print(f"saved best checkpoint: {best_path}")
        print(f"saved final checkpoint: {final_path}")

    return history
