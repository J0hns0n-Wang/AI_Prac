"""Core data models for the cluster scheduler: Job and Machine."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    """A computational job that requires CPU and memory resources.

    Attributes:
        job_id: Unique identifier for this job.
        cpu: CPU units required (e.g., number of cores).
        memory: Memory required (e.g., in GB).
        duration: How long the job runs once placed (in time steps).
        arrival_time: When the job entered the system.
        start_time: When the job was placed on a machine (None if still queued).
        completion_time: When the job finished (None if not yet complete).
    """

    job_id: int
    cpu: float
    memory: float
    duration: float
    arrival_time: float
    start_time: Optional[float] = None
    completion_time: Optional[float] = None

    @property
    def waiting_time(self) -> Optional[float]:
        """Time spent waiting in queue before placement."""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    @property
    def turnaround_time(self) -> Optional[float]:
        """Total time from arrival to completion."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time

    @property
    def is_completed(self) -> bool:
        return self.completion_time is not None

    @property
    def is_running(self) -> bool:
        return self.start_time is not None and self.completion_time is None

    @property
    def is_queued(self) -> bool:
        return self.start_time is None


@dataclass
class Machine:
    """A machine in the cluster with limited CPU and memory capacity.

    Attributes:
        machine_id: Unique identifier for this machine.
        cpu_capacity: Total CPU units available.
        memory_capacity: Total memory available.
        running_jobs: Jobs currently executing on this machine.
    """

    machine_id: int
    cpu_capacity: float
    memory_capacity: float
    running_jobs: list[Job] = field(default_factory=list)

    @property
    def cpu_used(self) -> float:
        return sum(job.cpu for job in self.running_jobs)

    @property
    def memory_used(self) -> float:
        return sum(job.memory for job in self.running_jobs)

    @property
    def cpu_free(self) -> float:
        return self.cpu_capacity - self.cpu_used

    @property
    def memory_free(self) -> float:
        return self.memory_capacity - self.memory_used

    @property
    def cpu_utilization(self) -> float:
        """Fraction of CPU in use (0.0 to 1.0)."""
        if self.cpu_capacity == 0:
            return 0.0
        return self.cpu_used / self.cpu_capacity

    @property
    def memory_utilization(self) -> float:
        """Fraction of memory in use (0.0 to 1.0)."""
        if self.memory_capacity == 0:
            return 0.0
        return self.memory_used / self.memory_capacity

    def can_fit(self, job: Job) -> bool:
        """Check if this machine has enough free resources for the job."""
        return self.cpu_free >= job.cpu and self.memory_free >= job.memory

    def place_job(self, job: Job, current_time: float) -> None:
        """Place a job on this machine. Raises ValueError if it doesn't fit."""
        if not self.can_fit(job):
            raise ValueError(
                f"Job {job.job_id} (cpu={job.cpu}, mem={job.memory}) "
                f"does not fit on Machine {self.machine_id} "
                f"(cpu_free={self.cpu_free}, mem_free={self.memory_free})"
            )
        job.start_time = current_time
        job.completion_time = current_time + job.duration
        self.running_jobs.append(job)

    def release_completed_jobs(self, current_time: float) -> list[Job]:
        """Remove and return all jobs that have completed by current_time."""
        completed = [j for j in self.running_jobs if j.completion_time <= current_time]
        self.running_jobs = [j for j in self.running_jobs if j.completion_time > current_time]
        return completed

    def reset(self) -> None:
        """Clear all running jobs from this machine."""
        self.running_jobs.clear()
