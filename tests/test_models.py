"""Tests for Job and Machine data models."""

import pytest
from cluster_scheduler.models import Job, Machine


# ── Job properties ──────────────────────────────────────────────

class TestJob:
    def test_new_job_is_queued(self):
        job = Job(job_id=1, cpu=2, memory=4, duration=10, arrival_time=0)
        assert job.is_queued
        assert not job.is_running
        assert not job.is_completed

    def test_waiting_time(self):
        job = Job(job_id=1, cpu=1, memory=1, duration=5, arrival_time=2)
        assert job.waiting_time is None  # still queued
        job.start_time = 7
        assert job.waiting_time == 5

    def test_turnaround_time(self):
        job = Job(job_id=1, cpu=1, memory=1, duration=5, arrival_time=2)
        assert job.turnaround_time is None
        job.start_time = 4
        job.completion_time = 9
        assert job.turnaround_time == 7

    def test_running_state(self):
        job = Job(job_id=1, cpu=1, memory=1, duration=5, arrival_time=0)
        job.start_time = 3
        assert job.is_running
        assert not job.is_queued
        assert not job.is_completed

    def test_completed_state(self):
        job = Job(job_id=1, cpu=1, memory=1, duration=5, arrival_time=0)
        job.start_time = 3
        job.completion_time = 8
        assert job.is_completed
        assert not job.is_running
        assert not job.is_queued


# ── Machine resource tracking ──────────────────────────────────

class TestMachineResources:
    def make_machine(self):
        return Machine(machine_id=0, cpu_capacity=8, memory_capacity=16)

    def test_empty_machine(self):
        m = self.make_machine()
        assert m.cpu_free == 8
        assert m.memory_free == 16
        assert m.cpu_utilization == 0.0
        assert m.memory_utilization == 0.0

    def test_utilization_after_placement(self):
        m = self.make_machine()
        job = Job(job_id=1, cpu=2, memory=4, duration=10, arrival_time=0)
        m.place_job(job, current_time=0)
        assert m.cpu_free == 6
        assert m.memory_free == 12
        assert m.cpu_utilization == pytest.approx(0.25)
        assert m.memory_utilization == pytest.approx(0.25)


# ── Placement logic ────────────────────────────────────────────

class TestPlacement:
    def test_can_fit(self):
        m = Machine(machine_id=0, cpu_capacity=4, memory_capacity=8)
        small = Job(job_id=1, cpu=2, memory=3, duration=5, arrival_time=0)
        big = Job(job_id=2, cpu=5, memory=3, duration=5, arrival_time=0)
        assert m.can_fit(small)
        assert not m.can_fit(big)

    def test_place_job_sets_times(self):
        m = Machine(machine_id=0, cpu_capacity=4, memory_capacity=8)
        job = Job(job_id=1, cpu=2, memory=3, duration=5, arrival_time=1)
        m.place_job(job, current_time=3)
        assert job.start_time == 3
        assert job.completion_time == 8
        assert job in m.running_jobs

    def test_place_job_rejects_if_no_room(self):
        m = Machine(machine_id=0, cpu_capacity=4, memory_capacity=8)
        job = Job(job_id=1, cpu=5, memory=3, duration=5, arrival_time=0)
        with pytest.raises(ValueError):
            m.place_job(job, current_time=0)

    def test_multiple_placements_consume_resources(self):
        m = Machine(machine_id=0, cpu_capacity=8, memory_capacity=16)
        j1 = Job(job_id=1, cpu=3, memory=6, duration=10, arrival_time=0)
        j2 = Job(job_id=2, cpu=4, memory=8, duration=10, arrival_time=0)
        m.place_job(j1, current_time=0)
        m.place_job(j2, current_time=0)
        assert m.cpu_free == 1
        assert m.memory_free == 2
        # Third job should not fit
        j3 = Job(job_id=3, cpu=2, memory=1, duration=5, arrival_time=0)
        assert not m.can_fit(j3)


# ── Release logic ──────────────────────────────────────────────

class TestRelease:
    def test_release_completed_jobs(self):
        m = Machine(machine_id=0, cpu_capacity=8, memory_capacity=16)
        j1 = Job(job_id=1, cpu=2, memory=4, duration=5, arrival_time=0)
        j2 = Job(job_id=2, cpu=3, memory=6, duration=10, arrival_time=0)
        m.place_job(j1, current_time=0)  # completes at t=5
        m.place_job(j2, current_time=0)  # completes at t=10

        completed = m.release_completed_jobs(current_time=5)
        assert len(completed) == 1
        assert completed[0].job_id == 1
        assert len(m.running_jobs) == 1
        assert m.cpu_free == 5  # freed j1's 2 cpu

    def test_release_nothing_if_none_done(self):
        m = Machine(machine_id=0, cpu_capacity=8, memory_capacity=16)
        job = Job(job_id=1, cpu=2, memory=4, duration=10, arrival_time=0)
        m.place_job(job, current_time=0)
        completed = m.release_completed_jobs(current_time=3)
        assert len(completed) == 0
        assert len(m.running_jobs) == 1

    def test_reset_clears_everything(self):
        m = Machine(machine_id=0, cpu_capacity=8, memory_capacity=16)
        job = Job(job_id=1, cpu=2, memory=4, duration=10, arrival_time=0)
        m.place_job(job, current_time=0)
        m.reset()
        assert len(m.running_jobs) == 0
        assert m.cpu_free == 8
        assert m.memory_free == 16
