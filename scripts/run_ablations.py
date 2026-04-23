"""Reward × featurizer ablation grid.

For each (reward_variant, featurizer_variant, train_seed) cell, train a
MaskablePPO policy and evaluate it on held-out seeds across the canonical
regimes. Emits a single tidy CSV and a heatmap per metric.

Dry-run mode (``--dry-run``) swaps training for a RandomScheduler so the
grid can be exercised fast for tests and pipeline smoke checks.

Usage:
    python scripts/run_ablations.py --timesteps 200000 \\
        --train-seeds 0 1 2 --eval-seeds 1000 1001 1002 \\
        --out-dir artifacts/ablations

    python scripts/run_ablations.py --dry-run --out-dir /tmp/dry
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from cluster_scheduler.env import RewardConfig
from cluster_scheduler.evaluation import RegimeSpec, default_regimes, sweep
from cluster_scheduler.featurizers import BasicFeaturizer, Featurizer, RichFeaturizer
from cluster_scheduler.report import plot_ablation_heatmap, write_results_csv
from cluster_scheduler.scheduler import RandomScheduler, Scheduler
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


#: Reward variants stressed by the ablation.
DEFAULT_REWARD_VARIANTS: dict[str, RewardConfig] = {
    "sparse": RewardConfig(mode="sparse", completion_bonus=1.0),
    "dense": RewardConfig(mode="dense"),
    "dense_backlog": RewardConfig(mode="dense", backlog_penalty_weight=0.1),
}

#: Featurizer variants. Values are factories so each cell gets a fresh one.
DEFAULT_FEATURIZER_VARIANTS: dict[str, Callable[[], Featurizer]] = {
    "basic": lambda: BasicFeaturizer(),
    "rich": lambda: RichFeaturizer(top_k=4),
}

#: Training regime. Eval uses ``default_regimes()`` so we measure generalization.
DEFAULT_TRAIN_SIM = SimulatorConfig(num_machines=10, cpu_per_machine=8, memory_per_machine=16)
DEFAULT_TRAIN_WORKLOAD = WorkloadConfig(num_jobs=100, arrival_rate=2.0)

HEATMAP_METRICS = ("avg_waiting_time", "p95_waiting_time", "utilization")


def _train_rl_scheduler(
    reward_cfg: RewardConfig,
    featurizer: Featurizer,
    train_seed: int,
    timesteps: int,
    regime: RegimeSpec,
) -> Scheduler:
    """Train a MaskablePPO policy and wrap it as an RLScheduler for one eval regime."""
    from cluster_scheduler.scheduler import RLScheduler
    from cluster_scheduler.train import make_env, train_maskable_ppo

    factory = make_env(
        sim_config=DEFAULT_TRAIN_SIM,
        workload_config=DEFAULT_TRAIN_WORKLOAD,
        reward_config=reward_cfg,
        featurizer=featurizer,
        seed=train_seed,
    )
    model = train_maskable_ppo(factory, total_timesteps=timesteps, seed=train_seed)
    return RLScheduler(
        model=model,
        sim_config=regime.sim_config,
        queue_capacity=regime.workload_config.num_jobs,
        featurizer=featurizer,
    )


def run_ablations(
    *,
    reward_variants: dict[str, RewardConfig] = DEFAULT_REWARD_VARIANTS,
    featurizer_variants: dict[str, Callable[[], Featurizer]] = DEFAULT_FEATURIZER_VARIANTS,
    train_seeds: Sequence[int] = (0, 1, 2),
    eval_seeds: Sequence[int] = (1000, 1001, 1002),
    regimes: Sequence[RegimeSpec] | None = None,
    timesteps: int = 200_000,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the full ablation grid and return a tidy DataFrame.

    Columns: ``reward``, ``featurizer``, ``train_seed``, plus the
    per-(scheduler, regime, seed) columns emitted by :func:`sweep`.

    In dry-run mode, training is skipped and a ``RandomScheduler`` is used
    in each cell — useful for wiring smoke tests without paying for PPO
    training time.
    """
    regimes_seq = list(regimes) if regimes is not None else default_regimes()
    frames: list[pd.DataFrame] = []

    for reward_name, reward_cfg in reward_variants.items():
        for feat_name, feat_factory in featurizer_variants.items():
            for train_seed in train_seeds:
                for regime in regimes_seq:
                    # Each (train_seed, regime) combination gets its own
                    # freshly-constructed featurizer, so featurizer state
                    # (if any) doesn't leak across cells.
                    featurizer = feat_factory()
                    if dry_run:
                        scheduler: Scheduler = RandomScheduler(seed=train_seed)
                    else:
                        scheduler = _train_rl_scheduler(
                            reward_cfg=reward_cfg,
                            featurizer=featurizer,
                            train_seed=train_seed,
                            timesteps=timesteps,
                            regime=regime,
                        )
                    df = sweep({"rl": scheduler}, [regime], eval_seeds)
                    df["reward"] = reward_name
                    df["featurizer"] = feat_name
                    df["train_seed"] = int(train_seed)
                    frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_artifacts(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(df, out_dir / "ablations_raw.csv")
    for metric in HEATMAP_METRICS:
        plot_ablation_heatmap(
            df, metric=metric, out_path=out_dir / f"heatmap_{metric}.png"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    p.add_argument("--out-dir", type=str, default="artifacts/ablations")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip PPO training; use RandomScheduler in every cell.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    df = run_ablations(
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        timesteps=args.timesteps,
        dry_run=args.dry_run,
    )
    write_artifacts(df, Path(args.out_dir))


if __name__ == "__main__":
    main()
