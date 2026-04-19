"""Tests for heuristic schedulers."""

import pytest
from cluster_scheduler.models import Job, Machine
from cluster_scheduler.scheduler import (
    FirstFitScheduler,
    BestFitScheduler,
    ShortestJobFirstScheduler,
)
from cluster_scheduler.simulator import Simulator, SimulatorConfig


def make_job(job_id, cpu, memory, duration, arrival_time=0):
    return Job(job_id=job_id, cpu=cpu, memory=memory,
               duration=duration, arrival_time=arrival_time)


def make_machines(specs):
    """specs: list of (cpu_capacity, memory_capacity)"""
    return [Machine(machine_id=i, cpu_capacity=c, memory_capacity=m)
            for i, (c, m) in enumerate(specs)]


# ── FirstFit ───────────────────────────────────────────────────

class TestFirstFit:
    def test_picks_first_eligible(self):
        machines = make_machines([(4, 8), (4, 8), (4, 8)])
        job = make_job(0, cpu=2, memory=3, duration=5)
        result = FirstFitScheduler().schedule([job], machines)
        assert result == (0, 0)  # first job, first machine

    def test_skips_full_machine(self):
        machines = make_machines([(2, 4), (4, 8)])
        # Fill first machine
        machines[0].place_job(make_job(99, cpu=2, memory=4, duration=10), current_time=0)
        job = make_job(0, cpu=1, memory=2, duration=5)
        result = FirstFitScheduler().schedule([job], machines)
        assert result == (0, 1)  # first job, second machine

    def test_returns_none_when_nothing_fits(self):
        machines = make_machines([(2, 4)])
        job = make_job(0, cpu=5, memory=10, duration=5)
        result = FirstFitScheduler().schedule([job], machines)
        assert result is None


# ── BestFit ────────────────────────────────────────────────────

class TestBestFit:
    def test_picks_tightest_fit(self):
        machines = make_machines([(8, 16), (3, 6), (4, 8)])
        job = make_job(0, cpu=2, memory=4, duration=5)
        result = BestFitScheduler().schedule([job], machines)
        # Machine 1 (3cpu, 6mem) has tightest fit: remaining = (3-2)+(6-4) = 3
        # Machine 2 (4cpu, 8mem): remaining = (4-2)+(8-4) = 6
        # Machine 0 (8cpu, 16mem): remaining = (8-2)+(16-4) = 18
        assert result == (0, 1)

    def test_returns_none_when_nothing_fits(self):
        machines = make_machines([(2, 4)])
        job = make_job(0, cpu=5, memory=10, duration=5)
        result = BestFitScheduler().schedule([job], machines)
        assert result is None


# ── ShortestJobFirst ───────────────────────────────────────────

class TestSJF:
    def test_picks_shortest_job(self):
        machines = make_machines([(4, 8)])
        queue = [
            make_job(0, cpu=1, memory=1, duration=10),  # long
            make_job(1, cpu=1, memory=1, duration=2),   # short
            make_job(2, cpu=1, memory=1, duration=5),   # medium
        ]
        result = ShortestJobFirstScheduler().schedule(queue, machines)
        assert result[0] == 1  # picks job index 1 (duration=2)

    def test_skips_unfittable_short_job(self):
        machines = make_machines([(2, 4)])
        queue = [
            make_job(0, cpu=1, memory=1, duration=10),  # long, fits
            make_job(1, cpu=5, memory=10, duration=1),  # shortest but too big
        ]
        result = ShortestJobFirstScheduler().schedule(queue, machines)
        assert result[0] == 0  # falls back to the job that fits


# ── End-to-end through simulator ───────────────────────────────

class TestSchedulersInSimulator:
    def _run_workload(self, scheduler):
        from cluster_scheduler.workload import WorkloadConfig, WorkloadGenerator
        cfg = WorkloadConfig(num_jobs=30, arrival_rate=2.0,
                             cpu_range=(1, 3), memory_range=(1, 4),
                             duration_range=(2, 8))
        jobs = WorkloadGenerator(cfg, seed=42).generate()
        sim = Simulator(SimulatorConfig(num_machines=5, cpu_per_machine=8, memory_per_machine=16))
        return sim.run(jobs, scheduler)

    def test_first_fit_completes_all(self):
        completed = self._run_workload(FirstFitScheduler())
        assert len(completed) == 30

    def test_best_fit_completes_all(self):
        completed = self._run_workload(BestFitScheduler())
        assert len(completed) == 30

    def test_sjf_completes_all(self):
        completed = self._run_workload(ShortestJobFirstScheduler())
        assert len(completed) == 30
