"""Metrics computation for evaluating scheduler performance."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cluster_scheduler.models import Job


# Two-sided normal critical values for common alpha levels. Using a normal
# approximation keeps mean_ci dependency-free (no scipy). With small N this
# slightly understates the CI vs. a t-distribution, which is acceptable for
# the project's scale of ~5-20 seeds.
_Z_BY_ALPHA: dict[float, float] = {
    0.01: 2.576,
    0.05: 1.96,
    0.10: 1.645,
}


def mean_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """Return ``(mean, ci_lo, ci_hi)`` for a 1-D array of values.

    Uses a normal-approximation CI on the standard error of the mean.
    Returns ``(mean, mean, mean)`` for length-0 or length-1 inputs.

    Args:
        values: Sample values.
        alpha: Two-sided significance level (e.g. 0.05 for a 95% CI).
            Supported: 0.01, 0.05, 0.10.

    Raises:
        KeyError: If ``alpha`` is not one of the supported values.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    z = _Z_BY_ALPHA[alpha]
    return mean, mean - z * sem, mean + z * sem


@dataclass
class SchedulerMetrics:
    """Performance metrics for a single simulation run.

    Attributes:
        avg_completion_time: Mean turnaround time (arrival to finish).
        avg_waiting_time: Mean time jobs spent in the queue.
        p95_completion_time: 95th percentile turnaround time.
        p99_completion_time: 99th percentile turnaround time.
        p95_waiting_time: 95th percentile waiting time.
        p99_waiting_time: 99th percentile waiting time.
        utilization: Fraction of total machine-time that was used by jobs.
        total_jobs: Number of jobs in this run.
    """

    avg_completion_time: float
    avg_waiting_time: float
    p95_completion_time: float
    p99_completion_time: float
    p95_waiting_time: float
    p99_waiting_time: float
    utilization: float
    total_jobs: int


def compute_metrics(
    completed_jobs: list[Job],
    num_machines: int,
    cpu_per_machine: float,
    memory_per_machine: float,
) -> SchedulerMetrics:
    """Compute performance metrics from a list of completed jobs.

    Args:
        completed_jobs: Jobs with start_time and completion_time set.
        num_machines: Number of machines in the cluster.
        cpu_per_machine: CPU capacity per machine.
        memory_per_machine: Memory capacity per machine.

    Returns:
        SchedulerMetrics with all fields populated.
    """
    if not completed_jobs:
        return SchedulerMetrics(
            avg_completion_time=0.0, avg_waiting_time=0.0,
            p95_completion_time=0.0, p99_completion_time=0.0,
            p95_waiting_time=0.0, p99_waiting_time=0.0,
            utilization=0.0, total_jobs=0,
        )

    turnaround_times = np.array([j.turnaround_time for j in completed_jobs])
    waiting_times = np.array([j.waiting_time for j in completed_jobs])

    # Utilization: total CPU-time used by jobs / total CPU-time available
    # Available = num_machines * cpu_per_machine * simulation_duration
    sim_start = min(j.arrival_time for j in completed_jobs)
    sim_end = max(j.completion_time for j in completed_jobs)
    sim_duration = sim_end - sim_start

    if sim_duration > 0:
        total_cpu_used = sum(j.cpu * j.duration for j in completed_jobs)
        total_cpu_available = num_machines * cpu_per_machine * sim_duration
        utilization = total_cpu_used / total_cpu_available
    else:
        utilization = 0.0

    return SchedulerMetrics(
        avg_completion_time=float(np.mean(turnaround_times)),
        avg_waiting_time=float(np.mean(waiting_times)),
        p95_completion_time=float(np.percentile(turnaround_times, 95)),
        p99_completion_time=float(np.percentile(turnaround_times, 99)),
        p95_waiting_time=float(np.percentile(waiting_times, 95)),
        p99_waiting_time=float(np.percentile(waiting_times, 99)),
        utilization=utilization,
        total_jobs=len(completed_jobs),
    )
