"""Tests for the Simulator."""

import pytest
from cluster_scheduler.models import Job, Machine
from cluster_scheduler.scheduler import RandomScheduler
from cluster_scheduler.simulator import Simulator, SimulatorConfig


def make_job(job_id, cpu, memory, duration, arrival_time):
    return Job(job_id=job_id, cpu=cpu, memory=memory,
               duration=duration, arrival_time=arrival_time)


class TestSimulatorBasics:
    def test_all_jobs_complete(self):
        """Every job should eventually finish."""
        sim = Simulator(SimulatorConfig(num_machines=3, cpu_per_machine=4, memory_per_machine=8))
        jobs = [make_job(i, cpu=1, memory=2, duration=5, arrival_time=i) for i in range(10)]
        completed = sim.run(jobs, RandomScheduler(seed=42))
        assert len(completed) == 10

    def test_completion_times_are_set(self):
        sim = Simulator(SimulatorConfig(num_machines=2, cpu_per_machine=4, memory_per_machine=8))
        jobs = [make_job(0, cpu=1, memory=2, duration=5, arrival_time=0)]
        completed = sim.run(jobs, RandomScheduler(seed=42))
        assert completed[0].start_time == 0
        assert completed[0].completion_time == 5

    def test_jobs_placed_immediately_when_room(self):
        """With plenty of capacity, no job should wait."""
        sim = Simulator(SimulatorConfig(num_machines=5, cpu_per_machine=8, memory_per_machine=16))
        jobs = [make_job(i, cpu=1, memory=1, duration=3, arrival_time=i) for i in range(5)]
        completed = sim.run(jobs, RandomScheduler(seed=42))
        for job in completed:
            assert job.waiting_time == 0.0


class TestQueueing:
    def test_jobs_queue_when_full(self):
        """Jobs should wait when no machine can fit them."""
        # 1 machine, capacity for 1 job at a time
        sim = Simulator(SimulatorConfig(num_machines=1, cpu_per_machine=2, memory_per_machine=4))
        jobs = [
            make_job(0, cpu=2, memory=4, duration=10, arrival_time=0),  # fills machine
            make_job(1, cpu=2, memory=4, duration=5, arrival_time=1),   # must wait
        ]
        completed = sim.run(jobs, RandomScheduler(seed=42))
        assert len(completed) == 2
        # Second job can't start until first finishes at t=10
        job1 = next(j for j in completed if j.job_id == 1)
        assert job1.start_time >= 10
        assert job1.waiting_time > 0

    def test_queued_jobs_placed_after_completion(self):
        """Once resources free up, queued jobs should be placed."""
        sim = Simulator(SimulatorConfig(num_machines=1, cpu_per_machine=4, memory_per_machine=8))
        jobs = [
            make_job(0, cpu=4, memory=8, duration=5, arrival_time=0),
            make_job(1, cpu=2, memory=4, duration=3, arrival_time=1),
        ]
        completed = sim.run(jobs, RandomScheduler(seed=42))
        job1 = next(j for j in completed if j.job_id == 1)
        # Job 1 should start at t=5 when job 0 finishes
        assert job1.start_time == 5
        assert job1.completion_time == 8


class TestTimeAdvancement:
    def test_time_advances_correctly(self):
        sim = Simulator(SimulatorConfig(num_machines=2, cpu_per_machine=4, memory_per_machine=8))
        jobs = [
            make_job(0, cpu=2, memory=4, duration=10, arrival_time=0),
            make_job(1, cpu=2, memory=4, duration=5, arrival_time=3),
        ]
        completed = sim.run(jobs, RandomScheduler(seed=42))
        j0 = next(j for j in completed if j.job_id == 0)
        j1 = next(j for j in completed if j.job_id == 1)
        assert j0.start_time == 0
        assert j0.completion_time == 10
        assert j1.start_time == 3
        assert j1.completion_time == 8

    def test_reset_clears_state(self):
        sim = Simulator(SimulatorConfig(num_machines=2, cpu_per_machine=4, memory_per_machine=8))
        jobs = [make_job(0, cpu=1, memory=1, duration=5, arrival_time=0)]
        sim.run(jobs, RandomScheduler(seed=42))
        assert len(sim.completed_jobs) == 1

        sim.reset()
        assert len(sim.completed_jobs) == 0
        assert len(sim.wait_queue) == 0
        assert sim.current_time == 0.0


class TestWithWorkloadGenerator:
    def test_full_pipeline(self):
        """Simulator works end-to-end with generated workloads."""
        from cluster_scheduler.workload import WorkloadConfig, WorkloadGenerator

        cfg = WorkloadConfig(num_jobs=50, arrival_rate=2.0,
                             cpu_range=(1, 3), memory_range=(1, 4),
                             duration_range=(2, 8))
        jobs = WorkloadGenerator(cfg, seed=42).generate()

        sim = Simulator(SimulatorConfig(num_machines=5, cpu_per_machine=8, memory_per_machine=16))
        completed = sim.run(jobs, RandomScheduler(seed=42))

        assert len(completed) == 50
        for job in completed:
            assert job.is_completed
            assert job.waiting_time >= 0
            assert job.turnaround_time > 0
