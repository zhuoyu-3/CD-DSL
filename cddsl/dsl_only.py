import copy
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as _data
from .checkpoints import (
    checkpoint_metadata,
    save_client_checkpoint,
    save_consensus_checkpoint,
)
from .client import CDDSLClient, round_learning_rate, train_local
from .config import CDDSLConfig
from .data import client_split_sizes, make_client_loaders
from .evaluation import (
    build_consensus_model,
    evaluate_decentralized_population,
    evaluate_loss_accuracy,
    evaluate_state_loss,
)
from .metrics import (
    append_metrics_csv,
    heterogeneity_summary,
    initialize_metrics_csv,
)
from .model import build_model
from .state import StateDict, clone_state, model_parameter_keys, set_state
from .topology import build_adjacency, column_stochastic_mixing, neighbor_support
from .utils import seed_everything


def _pso_velocity_step(
    client: CDDSLClient,
    neighbor_best_state: StateDict,
    c0: float,
    swarm_enabled: bool,
) -> None:
    current = client.current_state()
    next_state: StateDict = {}
    if swarm_enabled:
        r1 = float(torch.rand((), device=client.device).item())
        r2 = float(torch.rand((), device=client.device).item())
    for key, value in current.items():
        if key not in client.velocity:
            next_state[key] = value.clone()
            continue
        if not swarm_enabled:
            next_state[key] = value.clone()
            continue
        client.velocity[key] = (
            c0 * client.velocity[key]
            + client.cfg.c1 * r1 * (client.local_best_state[key] - value)
            + client.cfg.c2 * r2 * (neighbor_best_state[key] - value)
        )
        next_state[key] = value + client.cfg.velocity_step_size * client.velocity[key]
    set_state(client.model, next_state)


def run_dsl_only(cfg: CDDSLConfig) -> Dict[str, List[float]]:
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    trainset, testset, classes = _data.load_dataset(cfg)
    testloader = DataLoader(
        testset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    train_loaders, cross_loaders = make_client_loaders(
        trainset, testset, cfg, num_classes=len(classes)
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
    last_mean_test_loss = float("nan")
    last_mean_test_acc = float("nan")

    print("\n=================== DSL-Only Settings ===================")
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
        f"topology={cfg.topology}, connection_rate={cfg.connection_rate}, "
        f"extra_edges={cfg.extra_edges} (used only for neighbor lookup)"
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
        f"c0={cfg.c0}, c1={cfg.c1}, c2={cfg.c2}, c0_decay={cfg.c0_decay}, "
        f"velocity_step_size={cfg.velocity_step_size}, "
        f"swarm_warmup_rounds={cfg.swarm_warmup_rounds}"
    )
    print(
        f"norm_kind={cfg.norm_kind}, group_norm_groups={cfg.group_norm_groups}, "
        f"model=resnet18, random_erasing_prob={cfg.random_erasing_prob}"
    )
    print(
        "PURE DSL: argmin-loss best-neighbor selection per round, "
        "NO push-sum, NO model averaging across clients. "
        f"eval_population_every={cfg.eval_population_every}"
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
    print("=========================================================\n")

    scratch_model = copy.deepcopy(base_model).to(device)

    start = time.time()
    for round_id in range(cfg.rounds):
        c0 = cfg.c0 * (cfg.c0_decay ** round_id)
        lr_round = round_learning_rate(cfg, round_id)
        swarm_enabled = round_id >= cfg.swarm_warmup_rounds

        current_states = [client.current_state() for client in clients]

        neighbor_eval_losses: List[float] = []
        for receiver_idx, receiver in enumerate(clients):
            srcs = neighbor_support(A, receiver_idx)
            losses = [
                evaluate_state_loss(
                    scratch_model,
                    current_states[src],
                    receiver.cross_loader,
                    device,
                )
                for src in srcs
            ]
            best_pos = int(np.argmin(losses))
            best_state = current_states[srcs[best_pos]]
            receiver.update_neighbor_best(best_state)
            neighbor_eval_losses.append(float(losses[best_pos]))

        train_losses = []
        cross_losses = []
        cross_accs = []
        for client in clients:
            client.set_learning_rate(lr_round)
            _pso_velocity_step(
                client,
                client.neighbor_best_state,
                c0,
                swarm_enabled,
            )
            train_loss = train_local(
                client.model,
                client.train_loader,
                client.optimizer,
                client.criterion,
                cfg,
                device,
            )
            cross_loss, cross_acc = client.update_local_best()
            train_losses.append(train_loss)
            cross_losses.append(cross_loss)
            cross_accs.append(cross_acc)

        eval_pop_every = max(1, int(cfg.eval_population_every))
        should_eval_pop = (
            (round_id + 1) % eval_pop_every == 0
            or round_id == cfg.rounds - 1
            or round_id == 0
        )
        if should_eval_pop:
            _mean_test_loss, mean_test_acc = evaluate_decentralized_population(
                clients, testloader, device
            )
            last_mean_test_loss = _mean_test_loss
            last_mean_test_acc = mean_test_acc
        else:
            _mean_test_loss = last_mean_test_loss
            mean_test_acc = last_mean_test_acc

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
        history["mean_push_sum_mass"].append(float("nan"))
        history["mean_neighbor_eval_loss"].append(float(np.mean(neighbor_eval_losses)))
        history["mean_self_weight"].append(float("nan"))

        if metrics_path is not None:
            append_metrics_csv(
                metrics_path,
                {
                    "round": float(round_id + 1),
                    "mean_test_acc": float(mean_test_acc),
                    "consensus_test_acc": float(consensus_acc),
                    "cross_acc": float(np.mean(cross_accs)),
                    "neighbor_eval_loss": float(np.mean(neighbor_eval_losses)),
                    "train_loss": float(np.mean(train_losses)),
                    "cross_loss": float(np.mean(cross_losses)),
                    "mean_push_sum_mass": float("nan"),
                    "mean_self_weight": float("nan"),
                    "lr": float(lr_round),
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
            f"neighbor_loss={np.mean(neighbor_eval_losses):.4f} | "
            f"lr={lr_round:.5f} | c0={c0:.4f}"
        )

    print(f"\nDSL-only finished in {time.time() - start:.2f} seconds.")
    print("consensus_test_accuracy =", history["consensus_test_acc"])
    print("mean_client_test_accuracy =", history["mean_client_test_acc"])
    if metrics_path is not None:
        print(f"saved metrics csv: {metrics_path}")

    masses = [1.0] * cfg.clients
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
                base_P,
                masses,
                history,
                best_checkpoint_metrics,
            ),
            model_kind="dsl_only_best_consensus",
        )
        save_consensus_checkpoint(
            final_path,
            final_consensus_state,
            checkpoint_metadata(
                cfg,
                classes,
                A,
                base_P,
                base_P,
                masses,
                history,
                final_checkpoint_metrics,
            ),
            model_kind="dsl_only_final_consensus",
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
                base_P,
                masses,
                history,
                final_checkpoint_metrics,
            ),
        )
        print(f"saved client population checkpoint: {client_path}")
    return history
