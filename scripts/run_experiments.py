"""End-to-end evaluation: run baselines (+ optional trained RL policy) across
the canonical regime presets, write tidy CSVs, and render comparison plots.

Usage:
    python scripts/run_experiments.py --seeds 1000 1001 1002
    python scripts/run_experiments.py --model artifacts/ppo.zip \\
        --seeds 1000 1001 1002 --out-dir artifacts/eval_run

Each (regime, seed) workload is generated once and shared across all
schedulers, preserving the fair-comparison contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cluster_scheduler.evaluation import RegimeSpec, default_regimes, summarize, sweep
from cluster_scheduler.report import plot_regime_bars, write_results_csv
from cluster_scheduler.scheduler import (
    BestFitScheduler,
    FirstFitScheduler,
    ShortestJobFirstScheduler,
)


PLOT_METRICS = (
    "avg_waiting_time",
    "p95_waiting_time",
    "avg_completion_time",
    "utilization",
)


def _baseline_schedulers() -> dict:
    return {
        "first_fit": FirstFitScheduler(),
        "best_fit": BestFitScheduler(),
        "sjf": ShortestJobFirstScheduler(),
    }


def _load_model(model_path: str):
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(model_path)


def _build_rl_scheduler(model, regime: RegimeSpec):
    """Build an RLScheduler bound to a specific regime's cluster shape.

    RLScheduler needs ``sim_config`` for obs normalization and
    ``queue_capacity`` for the queue-length feature, so it must be rebuilt
    per regime.
    """
    from cluster_scheduler.scheduler import RLScheduler

    return RLScheduler(
        model=model,
        sim_config=regime.sim_config,
        queue_capacity=regime.workload_config.num_jobs,
    )


def _compatible_regimes(regimes: list[RegimeSpec], model) -> tuple[list[RegimeSpec], list[RegimeSpec]]:
    """Split regimes into (compatible, skipped) given the model's action space.

    A Discrete(n)-action policy cannot be applied to clusters with a different
    number of machines — both the mask length and the obs shape would mismatch.
    """
    n_actions = int(model.action_space.n)
    compatible, skipped = [], []
    for r in regimes:
        if r.sim_config.num_machines == n_actions:
            compatible.append(r)
        else:
            skipped.append(r)
    return compatible, skipped


def run(
    regimes: list[RegimeSpec],
    seeds: list[int],
    model_path: str | None,
    out_dir: Path,
) -> pd.DataFrame:
    model = _load_model(model_path) if model_path else None
    if model is not None:
        rl_regimes, skipped = _compatible_regimes(regimes, model)
        if skipped:
            names = ", ".join(r.name for r in skipped)
            print(
                f"[run_experiments] Skipping RL on regimes with mismatched "
                f"num_machines (trained={model.action_space.n}): {names}"
            )
    else:
        rl_regimes, skipped = [], []

    frames = []
    for regime in regimes:
        schedulers = _baseline_schedulers()
        if model is not None and regime in rl_regimes:
            schedulers["rl"] = _build_rl_scheduler(model, regime)
        frames.append(sweep(schedulers, [regime], seeds))
    df = pd.concat(frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(df, out_dir / "results_raw.csv")
    write_results_csv(summarize(df), out_dir / "results_summary.csv")

    for metric in PLOT_METRICS:
        plot_regime_bars(
            df, metric=metric, out_path=out_dir / f"regime_bars_{metric}.png"
        )

    return df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=str, default=None,
                   help="Path to a MaskablePPO .zip. If omitted, only baselines run.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    p.add_argument("--out-dir", type=str, default="artifacts/eval_run")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        regimes=default_regimes(),
        seeds=list(args.seeds),
        model_path=args.model,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
