"""Scheduler interface and heuristic implementations."""

from abc import ABC, abstractmethod

from cluster_scheduler.models import Job, Machine


class Scheduler(ABC):
    """Base class for all schedulers.

    A scheduler decides which job to place and on which machine.
    """

    @abstractmethod
    def schedule(self, queue: list[Job], machines: list[Machine]) -> tuple[int, int] | None:
        """Select a job from the queue and a machine to place it on.

        Args:
            queue: List of jobs waiting to be placed.
            machines: List of all machines in the cluster.

        Returns:
            (job_index, machine_index) or None if nothing can be placed.
        """
        ...


class RandomScheduler(Scheduler):
    """Picks a random job and places it on a random eligible machine."""

    def __init__(self, seed: int | None = None):
        import numpy as np
        self.rng = np.random.default_rng(seed)

    def schedule(self, queue: list[Job], machines: list[Machine]) -> tuple[int, int] | None:
        # Try each job in random order
        job_indices = list(range(len(queue)))
        self.rng.shuffle(job_indices)
        for ji in job_indices:
            eligible = [i for i, m in enumerate(machines) if m.can_fit(queue[ji])]
            if eligible:
                return ji, self.rng.choice(eligible)
        return None


class FirstFitScheduler(Scheduler):
    """Takes the first job in the queue, places it on the first machine that fits."""

    def schedule(self, queue: list[Job], machines: list[Machine]) -> tuple[int, int] | None:
        for ji, job in enumerate(queue):
            for mi, machine in enumerate(machines):
                if machine.can_fit(job):
                    return ji, mi
        return None


class BestFitScheduler(Scheduler):
    """Takes the first job in the queue, places it on the tightest-fitting machine.

    "Tightest" means the machine with the least remaining resources after placement,
    which reduces fragmentation.
    """

    def schedule(self, queue: list[Job], machines: list[Machine]) -> tuple[int, int] | None:
        for ji, job in enumerate(queue):
            best_mi = None
            best_remaining = float("inf")
            for mi, machine in enumerate(machines):
                if machine.can_fit(job):
                    remaining = (machine.cpu_free - job.cpu) + (machine.memory_free - job.memory)
                    if remaining < best_remaining:
                        best_remaining = remaining
                        best_mi = mi
            if best_mi is not None:
                return ji, best_mi
        return None


class ShortestJobFirstScheduler(Scheduler):
    """Picks the shortest-duration job from the queue, places it on the first machine that fits.

    This reduces average waiting time by getting small jobs out of the queue quickly.
    """

    def schedule(self, queue: list[Job], machines: list[Machine]) -> tuple[int, int] | None:
        # Sort job indices by duration (shortest first)
        sorted_indices = sorted(range(len(queue)), key=lambda i: queue[i].duration)
        for ji in sorted_indices:
            for mi, machine in enumerate(machines):
                if machine.can_fit(queue[ji]):
                    return ji, mi
        return None
