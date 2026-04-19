"""Workload generator for producing streams of jobs with stochastic arrivals."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cluster_scheduler.models import Job


@dataclass
class WorkloadConfig:
    """Configuration for workload generation.

    Attributes:
        num_jobs: Total number of jobs to generate.
        arrival_rate: Average arrivals per time unit (lambda for Poisson process).
        cpu_range: (min, max) CPU units per job.
        memory_range: (min, max) memory units per job.
        duration_range: (min, max) duration per job.
        burst_enabled: Whether to include bursty arrival periods.
        burst_arrival_rate: Arrival rate during burst periods (higher = more dense).
        burst_probability: Probability of entering a burst at each arrival.
        burst_length: Number of consecutive arrivals in a burst.
    """

    num_jobs: int = 100
    arrival_rate: float = 1.0
    cpu_range: tuple[float, float] = (1.0, 4.0)
    memory_range: tuple[float, float] = (1.0, 8.0)
    duration_range: tuple[float, float] = (1.0, 10.0)
    burst_enabled: bool = False
    burst_arrival_rate: float = 5.0
    burst_probability: float = 0.1
    burst_length: int = 5


class WorkloadGenerator:
    """Generates a sequence of Jobs with stochastic arrivals and resource demands."""

    def __init__(self, config: Optional[WorkloadConfig] = None, seed: Optional[int] = None):
        self.config = config or WorkloadConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self) -> list[Job]:
        """Generate a list of jobs according to the workload configuration."""
        cfg = self.config
        jobs: list[Job] = []
        current_time = 0.0
        burst_remaining = 0

        for i in range(cfg.num_jobs):
            # Determine arrival rate: normal or burst
            if cfg.burst_enabled and burst_remaining > 0:
                rate = cfg.burst_arrival_rate
                burst_remaining -= 1
            elif cfg.burst_enabled and self.rng.random() < cfg.burst_probability:
                rate = cfg.burst_arrival_rate
                burst_remaining = cfg.burst_length - 1  # -1 because this job counts
            else:
                rate = cfg.arrival_rate

            # Inter-arrival time from exponential distribution
            if i > 0:
                inter_arrival = self.rng.exponential(1.0 / rate)
                current_time += inter_arrival

            # Sample resource demands
            cpu = self.rng.uniform(*cfg.cpu_range)
            memory = self.rng.uniform(*cfg.memory_range)
            duration = self.rng.uniform(*cfg.duration_range)

            jobs.append(Job(
                job_id=i,
                cpu=round(cpu, 2),
                memory=round(memory, 2),
                duration=round(duration, 2),
                arrival_time=round(current_time, 4),
            ))

        return jobs
