"""Scheduler interface and basic implementations."""

from abc import ABC, abstractmethod

from cluster_scheduler.models import Job, Machine


class Scheduler(ABC):
    """Base class for all schedulers.

    A scheduler decides which machine should receive a given job.
    """

    @abstractmethod
    def select_machine(self, job: Job, machines: list[Machine]) -> int | None:
        """Select which machine to place the job on.

        Args:
            job: The job that needs placement.
            machines: List of all machines in the cluster.

        Returns:
            Index of the selected machine, or None if the job should wait.
        """
        ...


class RandomScheduler(Scheduler):
    """Places jobs on a random machine that has enough resources."""

    def __init__(self, seed: int | None = None):
        import numpy as np
        self.rng = np.random.default_rng(seed)

    def select_machine(self, job: Job, machines: list[Machine]) -> int | None:
        eligible = [i for i, m in enumerate(machines) if m.can_fit(job)]
        if not eligible:
            return None
        return self.rng.choice(eligible)
