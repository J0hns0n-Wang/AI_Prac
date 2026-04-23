"""Tests for heuristic schedulers."""

import numpy as np
import pytest

from cluster_scheduler.benchmark import run_benchmark
from cluster_scheduler.featurizers import BasicFeaturizer
from cluster_scheduler.models import Job, Machine
from cluster_scheduler.scheduler import (
    BestFitScheduler,
    FirstFitScheduler,
    RLScheduler,
    ShortestJobFirstScheduler,
)
from cluster_scheduler.simulator import Simulator, SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


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


# ── RLScheduler (adapter for trained policies) ─────────────────

class _StubModel:
    """Deterministic stub that mimics the MaskablePPO.predict signature.

    The ``choose`` callable receives ``(obs, mask)`` and returns the action;
    defaults to the first valid action.
    """

    def __init__(self, choose=None):
        self._choose = choose or (lambda obs, mask: int(np.where(mask)[0][0]))
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def predict(self, obs, action_masks=None, deterministic=True):
        mask = np.asarray(action_masks, dtype=bool)
        self.calls.append((np.asarray(obs, dtype=np.float32), mask))
        action = self._choose(obs, mask)
        return np.array(action, dtype=np.int64), None


class TestRLScheduler:
    def _sim_config(self, num_machines=3):
        return SimulatorConfig(num_machines=num_machines, cpu_per_machine=8, memory_per_machine=16)

    def test_returns_first_fittable_job_and_model_choice(self):
        machines = make_machines([(8, 16), (8, 16)])
        # Job 0 doesn't fit (too big), job 1 fits.
        queue = [
            make_job(0, cpu=20, memory=20, duration=5),
            make_job(1, cpu=2, memory=4, duration=5),
        ]
        # Stub always picks machine 1.
        model = _StubModel(choose=lambda obs, mask: 1)
        rl = RLScheduler(
            model=model,
            sim_config=self._sim_config(num_machines=2),
            queue_capacity=10,
            featurizer=BasicFeaturizer(),
        )
        result = rl.schedule(queue, machines)
        assert result == (1, 1)
        # Mask must have only the fittable machines for job 1 set True.
        _, mask = model.calls[-1]
        assert mask.tolist() == [True, True]

    def test_mask_reflects_can_fit(self):
        # Two machines; fill machine 0 so job cannot fit there.
        machines = make_machines([(8, 16), (8, 16)])
        machines[0].place_job(make_job(99, cpu=8, memory=0, duration=10), current_time=0)
        queue = [make_job(0, cpu=2, memory=4, duration=5)]
        model = _StubModel()
        rl = RLScheduler(
            model=model,
            sim_config=self._sim_config(num_machines=2),
            queue_capacity=10,
        )
        result = rl.schedule(queue, machines)
        assert result == (0, 1)
        _, mask = model.calls[-1]
        assert mask.tolist() == [False, True]

    def test_returns_none_when_nothing_fits(self):
        machines = make_machines([(2, 2)])
        queue = [make_job(0, cpu=10, memory=10, duration=5)]
        rl = RLScheduler(
            model=_StubModel(),
            sim_config=self._sim_config(num_machines=1),
            queue_capacity=10,
        )
        assert rl.schedule(queue, machines) is None

    def test_falls_back_when_model_violates_mask(self):
        # Machine 0 is full; model is buggy and proposes action 0.
        machines = make_machines([(8, 16), (8, 16)])
        machines[0].place_job(make_job(99, cpu=8, memory=0, duration=10), current_time=0)
        queue = [make_job(0, cpu=2, memory=4, duration=5)]
        bad_model = _StubModel(choose=lambda obs, mask: 0)
        rl = RLScheduler(
            model=bad_model,
            sim_config=self._sim_config(num_machines=2),
            queue_capacity=10,
        )
        # Fallback should pick the first *valid* machine (machine 1).
        assert rl.schedule(queue, machines) == (0, 1)

    def test_runs_inside_simulator(self):
        """End-to-end: Simulator.run with an RLScheduler stub completes the workload."""
        sim_cfg = self._sim_config(num_machines=5)
        wl = WorkloadConfig(
            num_jobs=20, arrival_rate=2.0,
            cpu_range=(1, 3), memory_range=(1, 4), duration_range=(2, 6),
        )
        from cluster_scheduler.workload import WorkloadGenerator
        jobs = WorkloadGenerator(wl, seed=7).generate()
        rl = RLScheduler(
            model=_StubModel(),
            sim_config=sim_cfg,
            queue_capacity=wl.num_jobs,
        )
        completed = Simulator(sim_cfg).run(jobs, rl)
        assert len(completed) == 20

    def test_runs_inside_run_benchmark(self):
        """run_benchmark accepts RLScheduler alongside a heuristic."""
        sim_cfg = self._sim_config(num_machines=4)
        wl = WorkloadConfig(
            num_jobs=15, arrival_rate=2.0,
            cpu_range=(1, 3), memory_range=(1, 4), duration_range=(2, 6),
        )
        results = run_benchmark(
            schedulers={
                "first_fit": FirstFitScheduler(),
                "stub_rl": RLScheduler(
                    model=_StubModel(),
                    sim_config=sim_cfg,
                    queue_capacity=wl.num_jobs,
                ),
            },
            workload_config=wl,
            sim_config=sim_cfg,
            seeds=[0, 1],
        )
        names = {r.scheduler_name for r in results}
        assert names == {"first_fit", "stub_rl"}
        for r in results:
            assert r.num_seeds == 2
            # Every seed produced metrics for a full workload.
            assert all(m.total_jobs == wl.num_jobs for m in r.per_seed_metrics)
