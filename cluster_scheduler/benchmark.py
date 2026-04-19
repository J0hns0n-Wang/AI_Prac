"""Benchmark runner for comparing schedulers across multiple seeds."""

from dataclasses import dataclass

import numpy as np

from cluster_scheduler.metrics import SchedulerMetrics, compute_metrics
from cluster_scheduler.scheduler import Scheduler
from cluster_scheduler.simulator import Simulator, SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig, WorkloadGenerator


@dataclass
class BenchmarkResult:
    """Aggregated results for one scheduler across multiple seeds."""

    scheduler_name: str
    per_seed_metrics: list[SchedulerMetrics]

    @property
    def num_seeds(self) -> int:
        return len(self.per_seed_metrics)

    def _gather(self, field: str) -> np.ndarray:
        return np.array([getattr(m, field) for m in self.per_seed_metrics])

    def mean(self, field: str) -> float:
        return float(np.mean(self._gather(field)))

    def std(self, field: str) -> float:
        return float(np.std(self._gather(field)))

    def summary(self) -> dict[str, str]:
        """Return a dict of 'mean ± std' strings for key metrics."""
        fields = [
            "avg_completion_time", "avg_waiting_time",
            "p95_completion_time", "p99_completion_time",
            "utilization",
        ]
        return {
            f: f"{self.mean(f):.3f} ± {self.std(f):.3f}" for f in fields
        }


def run_benchmark(
    schedulers: dict[str, Scheduler],
    workload_config: WorkloadConfig,
    sim_config: SimulatorConfig,
    seeds: list[int],
) -> list[BenchmarkResult]:
    """Run all schedulers on the same workloads and collect metrics.

    Args:
        schedulers: Dict mapping scheduler name to Scheduler instance.
        workload_config: Configuration for workload generation.
        sim_config: Configuration for the simulator.
        seeds: List of random seeds (each seed = one trial).

    Returns:
        List of BenchmarkResult, one per scheduler.
    """
    results = []

    for name, scheduler in schedulers.items():
        seed_metrics = []
        for seed in seeds:
            jobs = WorkloadGenerator(workload_config, seed=seed).generate()
            sim = Simulator(sim_config)
            completed = sim.run(jobs, scheduler)
            metrics = compute_metrics(
                completed,
                num_machines=sim_config.num_machines,
                cpu_per_machine=sim_config.cpu_per_machine,
                memory_per_machine=sim_config.memory_per_machine,
            )
            seed_metrics.append(metrics)
        results.append(BenchmarkResult(scheduler_name=name, per_seed_metrics=seed_metrics))

    return results


def print_benchmark(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison table."""
    fields = [
        ("Avg Completion", "avg_completion_time"),
        ("Avg Wait", "avg_waiting_time"),
        ("p95 Completion", "p95_completion_time"),
        ("p99 Completion", "p99_completion_time"),
        ("Utilization", "utilization"),
    ]

    # Header
    header = f"{'Scheduler':<20}"
    for label, _ in fields:
        header += f"  {label:<22}"
    print(header)
    print("-" * len(header))

    # Rows
    for r in results:
        row = f"{r.scheduler_name:<20}"
        for _, field in fields:
            row += f"  {r.mean(field):>8.3f} ± {r.std(field):<8.3f}  "
        print(row)
