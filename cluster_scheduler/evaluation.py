"""Multi-regime, multi-seed evaluation harness.

Runs a collection of schedulers (heuristic or learned) across a fixed set
of workload/cluster regimes and returns tidy per-(scheduler, regime, seed)
metrics. A companion ``summarize`` reduces that into mean/std/CI per
(scheduler, regime, metric) for plotting and tables.

Intended use: lock a canonical set of held-out regimes at training time
via :func:`default_regimes` (or a custom list) so the learned policy is
evaluated under distributional shifts, not memorization of the train
regime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cluster_scheduler.metrics import compute_metrics, mean_ci
from cluster_scheduler.scheduler import Scheduler
from cluster_scheduler.simulator import Simulator, SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig, WorkloadGenerator


#: Metric fields lifted from ``SchedulerMetrics`` into DataFrame columns.
METRIC_FIELDS: tuple[str, ...] = (
    "avg_completion_time",
    "avg_waiting_time",
    "p95_completion_time",
    "p99_completion_time",
    "p95_waiting_time",
    "p99_waiting_time",
    "utilization",
    "total_jobs",
)


@dataclass
class RegimeSpec:
    """A named (workload, cluster) evaluation regime.

    Attributes:
        name: Identifier used as the regime label in output DataFrames.
        workload_config: Workload generator configuration.
        sim_config: Cluster configuration.
    """

    name: str
    workload_config: WorkloadConfig
    sim_config: SimulatorConfig


def sweep(
    schedulers: dict[str, Scheduler],
    regimes: Sequence[RegimeSpec],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Run every (scheduler, regime, seed) combination.

    The workload is generated *once per (regime, seed)* and reused across
    all schedulers in that cell, so every scheduler sees exactly the same
    job trace — the fair-comparison contract the project's report promises.

    Args:
        schedulers: ``{name: Scheduler}`` mapping.
        regimes: Evaluation regimes.
        seeds: RNG seeds (one row per seed per cell).

    Returns:
        A tidy DataFrame with columns ``scheduler``, ``regime``, ``seed``,
        and one column per entry in :data:`METRIC_FIELDS`.
    """
    rows: list[dict[str, object]] = []
    for regime in regimes:
        for seed in seeds:
            jobs = WorkloadGenerator(regime.workload_config, seed=seed).generate()
            for name, scheduler in schedulers.items():
                completed = Simulator(regime.sim_config).run(jobs, scheduler)
                metrics = compute_metrics(
                    completed,
                    num_machines=regime.sim_config.num_machines,
                    cpu_per_machine=regime.sim_config.cpu_per_machine,
                    memory_per_machine=regime.sim_config.memory_per_machine,
                )
                row: dict[str, object] = {
                    "scheduler": name,
                    "regime": regime.name,
                    "seed": int(seed),
                }
                for f in METRIC_FIELDS:
                    row[f] = getattr(metrics, f)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Reduce sweep output to per-(scheduler, regime, metric) mean/std/CI.

    Args:
        df: Output of :func:`sweep`.
        alpha: CI significance level, passed to :func:`mean_ci`.

    Returns:
        Long-form DataFrame with columns ``scheduler``, ``regime``,
        ``metric``, ``mean``, ``std``, ``ci_lo``, ``ci_hi``, ``n``.
    """
    records: list[dict[str, object]] = []
    for (sched, regime), group in df.groupby(["scheduler", "regime"], sort=False):
        for metric in METRIC_FIELDS:
            values = group[metric].to_numpy(dtype=float)
            mean, ci_lo, ci_hi = mean_ci(values, alpha=alpha)
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            records.append(
                {
                    "scheduler": sched,
                    "regime": regime,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n": int(values.size),
                }
            )
    return pd.DataFrame(records)


def default_regimes() -> list[RegimeSpec]:
    """Canonical held-out regimes for testing generalization.

    Each regime is tuned to produce non-trivial queueing (~60-100% CPU
    utilization) so schedulers actually differentiate. Arrival rates are
    calibrated against the default demand distribution (cpu_range (1,4),
    duration_range (1,10) → ~13.75 CPU-sec per job) and cluster capacity.
    """
    return [
        RegimeSpec(
            name="light_poisson",
            workload_config=WorkloadConfig(num_jobs=200, arrival_rate=5.0),
            sim_config=SimulatorConfig(num_machines=10),
        ),
        RegimeSpec(
            name="heavy_poisson",
            workload_config=WorkloadConfig(num_jobs=200, arrival_rate=8.0),
            sim_config=SimulatorConfig(num_machines=10),
        ),
        RegimeSpec(
            name="bursty",
            workload_config=WorkloadConfig(
                num_jobs=200,
                arrival_rate=4.0,
                burst_enabled=True,
                burst_arrival_rate=12.0,
                burst_probability=0.15,
                burst_length=8,
            ),
            sim_config=SimulatorConfig(num_machines=10),
        ),
        RegimeSpec(
            name="large_jobs",
            workload_config=WorkloadConfig(
                num_jobs=150,
                arrival_rate=3.0,
                cpu_range=(3.0, 6.0),
                memory_range=(4.0, 12.0),
            ),
            sim_config=SimulatorConfig(num_machines=10),
        ),
        RegimeSpec(
            name="small_cluster",
            workload_config=WorkloadConfig(num_jobs=150, arrival_rate=2.0),
            sim_config=SimulatorConfig(num_machines=4),
        ),
        RegimeSpec(
            name="wide_cluster",
            workload_config=WorkloadConfig(num_jobs=400, arrival_rate=6.0),
            sim_config=SimulatorConfig(num_machines=20),
        ),
    ]
