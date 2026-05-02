import argparse

import torch

from cddsl import *  # noqa: F401,F403
from cddsl import CDDSLConfig, run_cd_dsl


def parse_args() -> CDDSLConfig:
    parser = argparse.ArgumentParser(description="Consensus-based decentralized DSL")
    parser.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clients", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-schedule", default="step", choices=["cosine", "step"])
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-decay-every", type=int, default=10)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--disable-nesterov", action="store_true")
    parser.add_argument("--data-style", default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--split-type", default="diri", choices=["diri", "diri_groups"])
    parser.add_argument("--dirichlet-alpha", type=float, default=5.0)
    parser.add_argument(
        "--heterogeneity-counts",
        type=int,
        nargs="+",
        default=[20, 15, 10, 5],
        help="Per-group client counts for split-type=diri_groups. Sum must equal --clients.",
    )
    parser.add_argument(
        "--heterogeneity-alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 10.0],
        help="Per-group Dirichlet alpha for split-type=diri_groups. Length must match --heterogeneity-counts.",
    )
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--samples-per-client", type=int, default=1000)
    parser.add_argument("--cross-split", type=float, default=0.1)
    parser.add_argument("--cross-samples-per-client", type=int, default=200)
    parser.add_argument(
        "--topology",
        default="random_rate",
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
        default=0.7,
        help="Directed off-diagonal edge rate for topology=random_rate. The effective rate is clamped upward if needed and the graph is resampled until strongly connected.",
    )
    parser.add_argument(
        "--topology-max-retries",
        type=int,
        default=256,
        help="Maximum retries when sampling random topologies until a strongly connected graph is found.",
    )
    parser.add_argument("--mixing-objective", default="eval", choices=["eval", "uniform"])
    parser.add_argument("--mixing-temperature", type=float, default=0.5)
    parser.add_argument("--mixing-blend", type=float, default=0.5)
    parser.add_argument("--mixing-min-self-weight", type=float, default=0.0)
    parser.add_argument("--mixing-warmup-rounds", type=int, default=3)
    parser.add_argument("--c0", type=float, default=0.05)
    parser.add_argument("--c1", type=float, default=0.15)
    parser.add_argument("--c2", type=float, default=0.15)
    parser.add_argument("--c0-decay", type=float, default=0.95)
    parser.add_argument("--swarm-warmup-rounds", type=int, default=5)
    parser.add_argument("--consensus-step-size", type=float, default=1.0)
    parser.add_argument("--velocity-step-size", type=float, default=0.05)
    parser.add_argument("--bn-calibration-batches", type=int, default=2)
    parser.add_argument("--norm-kind", default="group", choices=["batch", "group"])
    parser.add_argument("--group-norm-groups", type=int, default=8)
    parser.add_argument(
        "--random-erasing-prob",
        type=float,
        default=0.25,
        help="Probability of RandomErasing applied to CIFAR-10 training images. 0 disables it. 0.25 is a common CIFAR setting.",
    )
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
    parser.add_argument(
        "--eval-population-every",
        type=int,
        default=5,
        help="Recompute the mean per-client test accuracy only every N rounds. Reuses the previous value in between to save evaluation time on the full testset.",
    )
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
        heterogeneity_counts=tuple(args.heterogeneity_counts),
        heterogeneity_alphas=tuple(args.heterogeneity_alphas),
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
        random_erasing_prob=args.random_erasing_prob,
        save_checkpoint=args.save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        save_client_checkpoints=args.save_client_checkpoints,
        save_metrics=args.save_metrics,
        metrics_dir=args.metrics_dir,
        metrics_prefix=args.metrics_prefix,
        eval_population_every=args.eval_population_every,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        download=not args.no_download,
    )
    return cfg


if __name__ == "__main__":
    config = parse_args()
    run_cd_dsl(config)
