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
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cluster_scheduler.evaluation import RegimeSpec, default_regimes, summarize, sweep
from cluster_scheduler.featurizers import BasicFeaturizer, Featurizer
from cluster_scheduler.report import plot_regime_bars, write_results_csv
from cluster_scheduler.scheduler import (
    BestFitScheduler,
    FirstFitScheduler,
    ShortestJobFirstScheduler,
)
from cluster_scheduler.train import build_featurizer


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


def _load_sidecar(model_path: str) -> dict[str, Any]:
    """Return the sidecar metadata for ``model_path`` or ``{}`` if absent."""
    p = Path(model_path)
    sidecar = p.with_suffix(p.suffix + ".meta.json") if p.suffix else p.with_suffix(".meta.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[run_experiments] Warning: could not read sidecar {sidecar}: {e}")
    return {}


def _featurizer_from_meta(meta: dict[str, Any]) -> Featurizer:
    spec = meta.get("featurizer")
    if not spec:
        return BasicFeaturizer()
    return build_featurizer(spec)


def _load_model(model_path: str) -> tuple[Any, dict[str, Any]]:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path)
    return model, _load_sidecar(model_path)


def _build_rl_scheduler(model, meta: dict[str, Any], regime: RegimeSpec):
    """Build an RLScheduler bound to a regime, using the featurizer recorded
    in the sidecar (falls back to BasicFeaturizer for pre-sidecar models)."""
    from cluster_scheduler.scheduler import RLScheduler

    featurizer = _featurizer_from_meta(meta)
    return RLScheduler(
        model=model,
        sim_config=regime.sim_config,
        queue_capacity=regime.workload_config.num_jobs,
        featurizer=featurizer,
    )


def _models_by_num_machines(model_paths: list[str]) -> dict[int, tuple[Any, dict[str, Any]]]:
    """Load each .zip (+ sidecar) and key by the policy's action_space.n.

    A Discrete(n)-action policy can only be applied to clusters with exactly
    n machines; this lets the runner dispatch the right model per regime.
    """
    table: dict[int, tuple[Any, dict[str, Any]]] = {}
    for p in model_paths:
        model, meta = _load_model(p)
        n = int(model.action_space.n)
        if n in table:
            print(f"[run_experiments] Warning: overriding policy for num_machines={n} with {p}")
        feat = meta.get("featurizer", {}).get("name", "basic") if meta else "basic (no sidecar)"
        print(f"[run_experiments]   {p} → num_machines={n}, featurizer={feat}")
        table[n] = (model, meta)
    return table


def run(
    regimes: list[RegimeSpec],
    seeds: list[int],
    model_paths: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    models = _models_by_num_machines(model_paths) if model_paths else {}
    if models:
        print(f"[run_experiments] Loaded RL policies for num_machines in {sorted(models)}")

    skipped: list[str] = []
    frames = []
    for regime in regimes:
        schedulers = _baseline_schedulers()
        n = regime.sim_config.num_machines
        if n in models:
            model, meta = models[n]
            schedulers["rl"] = _build_rl_scheduler(model, meta, regime)
        elif models:
            skipped.append(f"{regime.name} (num_machines={n})")
        frames.append(sweep(schedulers, [regime], seeds))

    if skipped:
        print("[run_experiments] No matching RL policy for regimes: " + ", ".join(skipped))

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
    p.add_argument("--model", type=str, nargs="+", default=[],
                   help="Paths to one or more MaskablePPO .zip files. Each is "
                        "dispatched to regimes whose num_machines matches its "
                        "action_space.n. If omitted, only baselines run.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    p.add_argument("--out-dir", type=str, default="artifacts/eval_run")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(
        regimes=default_regimes(),
        seeds=list(args.seeds),
        model_paths=list(args.model),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
