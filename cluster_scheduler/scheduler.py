"""Scheduler interface and heuristic implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from cluster_scheduler.models import Job, Machine

if TYPE_CHECKING:
    from cluster_scheduler.featurizers import Featurizer
    from cluster_scheduler.simulator import SimulatorConfig


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


class RLScheduler(Scheduler):
    """Wraps a trained (Maskable)PPO policy as a :class:`Scheduler`.

    Mirrors ``ClusterSchedulingEnv``'s decision semantics so that a policy
    trained in the env behaves identically inside the benchmark harness:

    1. Job choice is FIFO head-of-queue-that-fits: the first queued job that
       can fit on at least one machine.
    2. The observation is built by the same featurizer used during training,
       with the current job removed from the queue (the env pops before
       featurizing).
    3. The observation is clipped to ``[0, 1]`` to match the env's Box space.
    4. The action mask restricts the policy to fittable machines.

    Args:
        model: A trained policy exposing ``.predict(obs, action_masks=...,
            deterministic=...)`` (e.g. ``sb3_contrib.MaskablePPO``).
        sim_config: Cluster configuration used for capacity normalization.
        queue_capacity: Normalizer for the queue-length feature (typically
            ``workload_config.num_jobs`` used at training time).
        featurizer: Observation builder. Must match the training-time
            featurizer. Defaults to ``BasicFeaturizer``.
        deterministic: If True, use greedy argmax actions. Set False for
            stochastic rollouts (e.g., multiple-seed averaging with
            sampling).
    """

    def __init__(
        self,
        model: Any,
        sim_config: "SimulatorConfig",
        queue_capacity: int,
        featurizer: "Featurizer | None" = None,
        deterministic: bool = True,
    ):
        # Import inside __init__ to avoid a circular import at module load:
        # scheduler.py → featurizers.py → (no cycle, but keeps imports lazy
        # and makes the default consistent with env.py).
        from cluster_scheduler.featurizers import BasicFeaturizer

        self.model = model
        self.sim_config = sim_config
        self.queue_capacity = int(queue_capacity)
        self.featurizer = featurizer or BasicFeaturizer()
        self.deterministic = bool(deterministic)

    def schedule(self, queue: list[Job], machines: list[Machine]) -> tuple[int, int] | None:
        # Find the first queued job that fits somewhere (FIFO head-of-queue-that-fits).
        ji = None
        current_job: Job | None = None
        for i, job in enumerate(queue):
            if any(m.can_fit(job) for m in machines):
                ji = i
                current_job = job
                break
        if ji is None or current_job is None:
            return None

        # Queue passed to the featurizer excludes the current job, matching
        # the env's pop-then-featurize order.
        remaining_queue = [j for idx, j in enumerate(queue) if idx != ji]
        obs = self.featurizer.featurize(
            machines=machines,
            current_job=current_job,
            wait_queue=remaining_queue,
            sim_config=self.sim_config,
            queue_capacity=self.queue_capacity,
        )
        obs = np.clip(obs, 0.0, 1.0)

        mask = np.array([m.can_fit(current_job) for m in machines], dtype=bool)
        action, _ = self.model.predict(
            obs,
            action_masks=mask,
            deterministic=self.deterministic,
        )
        mi = int(np.asarray(action).item())

        # Defensive: the model should respect the mask, but fall back to the
        # first valid machine if it doesn't (e.g., a non-maskable model).
        if not mask[mi]:
            mi = int(np.where(mask)[0][0])
        return ji, mi
