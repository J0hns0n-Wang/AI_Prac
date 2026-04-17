"""Tests for the WorkloadGenerator."""

import numpy as np
import pytest

from cluster_scheduler.workload import WorkloadConfig, WorkloadGenerator


class TestWorkloadBasics:
    def test_generates_correct_number_of_jobs(self):
        gen = WorkloadGenerator(WorkloadConfig(num_jobs=50), seed=42)
        jobs = gen.generate()
        assert len(jobs) == 50

    def test_job_ids_are_sequential(self):
        gen = WorkloadGenerator(WorkloadConfig(num_jobs=20), seed=42)
        jobs = gen.generate()
        assert [j.job_id for j in jobs] == list(range(20))

    def test_arrival_times_are_monotonic(self):
        gen = WorkloadGenerator(WorkloadConfig(num_jobs=100), seed=42)
        jobs = gen.generate()
        for i in range(1, len(jobs)):
            assert jobs[i].arrival_time >= jobs[i - 1].arrival_time

    def test_first_job_arrives_at_zero(self):
        gen = WorkloadGenerator(WorkloadConfig(num_jobs=10), seed=42)
        jobs = gen.generate()
        assert jobs[0].arrival_time == 0.0


class TestResourceRanges:
    def test_cpu_within_range(self):
        cfg = WorkloadConfig(num_jobs=200, cpu_range=(2.0, 6.0))
        gen = WorkloadGenerator(cfg, seed=42)
        jobs = gen.generate()
        for j in jobs:
            assert 2.0 <= j.cpu <= 6.0

    def test_memory_within_range(self):
        cfg = WorkloadConfig(num_jobs=200, memory_range=(1.0, 4.0))
        gen = WorkloadGenerator(cfg, seed=42)
        jobs = gen.generate()
        for j in jobs:
            assert 1.0 <= j.memory <= 4.0

    def test_duration_within_range(self):
        cfg = WorkloadConfig(num_jobs=200, duration_range=(3.0, 7.0))
        gen = WorkloadGenerator(cfg, seed=42)
        jobs = gen.generate()
        for j in jobs:
            assert 3.0 <= j.duration <= 7.0


class TestReproducibility:
    def test_same_seed_same_jobs(self):
        cfg = WorkloadConfig(num_jobs=50)
        jobs_a = WorkloadGenerator(cfg, seed=123).generate()
        jobs_b = WorkloadGenerator(cfg, seed=123).generate()
        for a, b in zip(jobs_a, jobs_b):
            assert a.arrival_time == b.arrival_time
            assert a.cpu == b.cpu
            assert a.memory == b.memory
            assert a.duration == b.duration

    def test_different_seed_different_jobs(self):
        cfg = WorkloadConfig(num_jobs=50)
        jobs_a = WorkloadGenerator(cfg, seed=1).generate()
        jobs_b = WorkloadGenerator(cfg, seed=2).generate()
        # At least some jobs should differ
        diffs = sum(1 for a, b in zip(jobs_a, jobs_b) if a.cpu != b.cpu)
        assert diffs > 0


class TestBurstMode:
    def test_burst_produces_tighter_arrivals(self):
        cfg_normal = WorkloadConfig(num_jobs=200, arrival_rate=1.0, burst_enabled=False)
        cfg_burst = WorkloadConfig(
            num_jobs=200, arrival_rate=1.0,
            burst_enabled=True, burst_arrival_rate=10.0,
            burst_probability=0.3, burst_length=10,
        )
        jobs_normal = WorkloadGenerator(cfg_normal, seed=42).generate()
        jobs_burst = WorkloadGenerator(cfg_burst, seed=42).generate()

        # Burst workload should have a shorter total time span
        # because burst periods pack jobs more tightly
        span_normal = jobs_normal[-1].arrival_time - jobs_normal[0].arrival_time
        span_burst = jobs_burst[-1].arrival_time - jobs_burst[0].arrival_time
        assert span_burst < span_normal

    def test_burst_still_monotonic(self):
        cfg = WorkloadConfig(
            num_jobs=100, burst_enabled=True,
            burst_arrival_rate=10.0, burst_probability=0.5, burst_length=5,
        )
        gen = WorkloadGenerator(cfg, seed=42)
        jobs = gen.generate()
        for i in range(1, len(jobs)):
            assert jobs[i].arrival_time >= jobs[i - 1].arrival_time
