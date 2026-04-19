"""Tests for metrics computation and benchmarking."""

import pytest
import numpy as np

from cluster_scheduler.models import Job
from cluster_scheduler.metrics import compute_metrics, SchedulerMetrics
from cluster_scheduler.benchmark import run_benchmark, BenchmarkResult
from cluster_scheduler.scheduler import FirstFitScheduler, BestFitScheduler
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


def make_completed_job(job_id, cpu, memory, duration, arrival, start):
    """Create a job with all timing fields set."""
    return Job(
        job_id=job_id, cpu=cpu, memory=memory, duration=duration,
        arrival_time=arrival, start_time=start,
        completion_time=start + duration,
    )


class TestComputeMetrics:
    def test_basic_metrics(self):
        jobs = [
            make_completed_job(0, cpu=1, memory=2, duration=4, arrival=0, start=0),
            make_completed_job(1, cpu=1, memory=2, duration=6, arrival=0, start=0),
        ]
        m = compute_metrics(jobs, num_machines=2, cpu_per_machine=4, memory_per_machine=8)
        # turnaround: job0=4, job1=6 → avg=5
        assert m.avg_completion_time == pytest.approx(5.0)
        # both started immediately → avg wait = 0
        assert m.avg_waiting_time == pytest.approx(0.0)
        assert m.total_jobs == 2

    def test_waiting_time(self):
        jobs = [
            make_completed_job(0, cpu=1, memory=1, duration=5, arrival=0, start=0),
            make_completed_job(1, cpu=1, memory=1, duration=5, arrival=0, start=5),  # waited 5
        ]
        m = compute_metrics(jobs, num_machines=1, cpu_per_machine=4, memory_per_machine=8)
        assert m.avg_waiting_time == pytest.approx(2.5)  # (0 + 5) / 2

    def test_percentiles(self):
        # 100 jobs, all with turnaround = their index
        jobs = [
            make_completed_job(i, cpu=1, memory=1, duration=1,
                               arrival=0, start=i)
            for i in range(100)
        ]
        m = compute_metrics(jobs, num_machines=1, cpu_per_machine=4, memory_per_machine=8)
        # turnaround[i] = start + duration - arrival = i + 1
        # p95 should be around 96
        assert m.p95_completion_time == pytest.approx(96.0, abs=1.0)
        assert m.p99_completion_time == pytest.approx(100.0, abs=1.0)

    def test_utilization(self):
        # 1 machine, 4 CPU, sim runs from t=0 to t=10
        # Job uses 2 CPU for 10 time units → cpu_used = 20
        # Available = 1 * 4 * 10 = 40 → utilization = 0.5
        jobs = [make_completed_job(0, cpu=2, memory=1, duration=10, arrival=0, start=0)]
        m = compute_metrics(jobs, num_machines=1, cpu_per_machine=4, memory_per_machine=8)
        assert m.utilization == pytest.approx(0.5)

    def test_empty_jobs(self):
        m = compute_metrics([], num_machines=1, cpu_per_machine=4, memory_per_machine=8)
        assert m.total_jobs == 0
        assert m.avg_completion_time == 0.0


class TestBenchmark:
    def test_benchmark_runs(self):
        schedulers = {
            "FirstFit": FirstFitScheduler(),
            "BestFit": BestFitScheduler(),
        }
        wl_cfg = WorkloadConfig(num_jobs=20, arrival_rate=2.0,
                                cpu_range=(1, 2), memory_range=(1, 3),
                                duration_range=(2, 5))
        sim_cfg = SimulatorConfig(num_machines=3, cpu_per_machine=4, memory_per_machine=8)
        results = run_benchmark(schedulers, wl_cfg, sim_cfg, seeds=[1, 2, 3])

        assert len(results) == 2
        for r in results:
            assert r.num_seeds == 3
            assert r.mean("avg_completion_time") > 0
            assert r.std("avg_completion_time") >= 0

    def test_same_seeds_same_results(self):
        """Running benchmark twice with same seeds should give identical metrics."""
        schedulers = {"FirstFit": FirstFitScheduler()}
        wl_cfg = WorkloadConfig(num_jobs=15)
        sim_cfg = SimulatorConfig(num_machines=3)

        r1 = run_benchmark(schedulers, wl_cfg, sim_cfg, seeds=[42])
        r2 = run_benchmark(schedulers, wl_cfg, sim_cfg, seeds=[42])

        assert r1[0].mean("avg_completion_time") == r2[0].mean("avg_completion_time")
