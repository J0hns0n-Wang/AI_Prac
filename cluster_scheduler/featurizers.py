"""Pluggable state featurizers for the cluster scheduling environment.

A ``Featurizer`` turns the current simulator state (machines, current job,
wait queue) into a fixed-length observation vector. Different featurizers
support ablation studies without forking the environment.

The default ``BasicFeaturizer`` reproduces the env's original observation
byte-for-byte; ``RichFeaturizer`` adds queue-peek and headroom features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from cluster_scheduler.models import Job, Machine
from cluster_scheduler.simulator import SimulatorConfig


class Featurizer(ABC):
    """Builds observation vectors from cluster state.

    Observations are always clipped to ``[0, 1]`` by the env, so concrete
    featurizers should aim to keep features in that range but need not
    clip themselves.
    """

    @abstractmethod
    def observation_shape(self, sim_config: SimulatorConfig) -> tuple[int, ...]:
        """Return the shape of the observation vector for this sim config."""

    @abstractmethod
    def featurize(
        self,
        machines: Sequence[Machine],
        current_job: Optional[Job],
        wait_queue: Sequence[Job],
        sim_config: SimulatorConfig,
        queue_capacity: int,
    ) -> np.ndarray:
        """Build the observation vector.

        Args:
            machines: All machines in the cluster.
            current_job: The job the agent is about to place, or ``None``
                if the episode is terminating.
            wait_queue: Jobs currently queued (includes ``current_job`` only
                if the caller has not yet popped it).
            sim_config: Cluster configuration (for capacity normalization).
            queue_capacity: Reference queue length for normalization (e.g.
                the total number of jobs in the workload).

        Returns:
            A 1-D ``np.float32`` array of shape ``observation_shape``.
        """


class BasicFeaturizer(Featurizer):
    """The original observation layout.

    Layout (all in ``[0, 1]`` after clipping):
        - per machine: (cpu_utilization, memory_utilization)
        - current job: (cpu / cpu_per_machine, memory / memory_per_machine,
                        duration / duration_ref)  — or zeros if no job
        - system: (queue_length / queue_capacity, mean machine cpu utilization)
    """

    def __init__(self, duration_ref: float = 100.0):
        self.duration_ref = float(duration_ref)

    def observation_shape(self, sim_config: SimulatorConfig) -> tuple[int, ...]:
        return (sim_config.num_machines * 2 + 3 + 2,)

    def featurize(
        self,
        machines: Sequence[Machine],
        current_job: Optional[Job],
        wait_queue: Sequence[Job],
        sim_config: SimulatorConfig,
        queue_capacity: int,
    ) -> np.ndarray:
        obs: list[float] = []

        for m in machines:
            obs.append(m.cpu_utilization)
            obs.append(m.memory_utilization)

        if current_job is not None:
            obs.append(current_job.cpu / sim_config.cpu_per_machine)
            obs.append(current_job.memory / sim_config.memory_per_machine)
            obs.append(current_job.duration / self.duration_ref)
        else:
            obs.extend([0.0, 0.0, 0.0])

        obs.append(len(wait_queue) / queue_capacity if queue_capacity > 0 else 0.0)

        if machines:
            obs.append(sum(m.cpu_utilization for m in machines) / len(machines))
        else:
            obs.append(0.0)

        return np.array(obs, dtype=np.float32)


class RichFeaturizer(Featurizer):
    """Basic features plus queue-peek and post-placement headroom.

    Adds, on top of :class:`BasicFeaturizer`:
        - top-k queued jobs' (cpu, mem, duration), zero-padded when the queue
          is shorter than ``top_k``.
        - Total backlog CPU demand and memory demand, each normalized by the
          cluster's total capacity.
        - Per-machine post-placement headroom: how much CPU / memory each
          machine would have left *after* accepting the current job, as a
          fraction of the machine's capacity (clamped at 0 when the job
          doesn't fit).

    Args:
        top_k: Number of queued jobs to expose.
        duration_ref: Normalizer for duration features.
    """

    def __init__(self, top_k: int = 4, duration_ref: float = 100.0):
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        self.top_k = int(top_k)
        self.duration_ref = float(duration_ref)
        self._basic = BasicFeaturizer(duration_ref=duration_ref)

    def observation_shape(self, sim_config: SimulatorConfig) -> tuple[int, ...]:
        basic = self._basic.observation_shape(sim_config)[0]
        queue_peek = self.top_k * 3
        backlog = 2
        headroom = sim_config.num_machines * 2
        return (basic + queue_peek + backlog + headroom,)

    def featurize(
        self,
        machines: Sequence[Machine],
        current_job: Optional[Job],
        wait_queue: Sequence[Job],
        sim_config: SimulatorConfig,
        queue_capacity: int,
    ) -> np.ndarray:
        basic = self._basic.featurize(
            machines, current_job, wait_queue, sim_config, queue_capacity
        )

        cpu_cap = sim_config.cpu_per_machine
        mem_cap = sim_config.memory_per_machine

        # top-k queued jobs (FIFO peek)
        queue_peek: list[float] = []
        for i in range(self.top_k):
            if i < len(wait_queue):
                j = wait_queue[i]
                queue_peek.extend([
                    j.cpu / cpu_cap,
                    j.memory / mem_cap,
                    j.duration / self.duration_ref,
                ])
            else:
                queue_peek.extend([0.0, 0.0, 0.0])

        # backlog totals normalized by cluster capacity
        total_cpu_capacity = cpu_cap * len(machines)
        total_mem_capacity = mem_cap * len(machines)
        backlog_cpu = sum(j.cpu for j in wait_queue)
        backlog_mem = sum(j.memory for j in wait_queue)
        backlog = [
            backlog_cpu / total_cpu_capacity if total_cpu_capacity > 0 else 0.0,
            backlog_mem / total_mem_capacity if total_mem_capacity > 0 else 0.0,
        ]

        # per-machine post-placement headroom for the current job
        headroom: list[float] = []
        job_cpu = current_job.cpu if current_job is not None else 0.0
        job_mem = current_job.memory if current_job is not None else 0.0
        for m in machines:
            cpu_after = max(m.cpu_free - job_cpu, 0.0) / cpu_cap if cpu_cap > 0 else 0.0
            mem_after = max(m.memory_free - job_mem, 0.0) / mem_cap if mem_cap > 0 else 0.0
            headroom.extend([cpu_after, mem_after])

        return np.concatenate(
            [basic, np.array(queue_peek + backlog + headroom, dtype=np.float32)]
        ).astype(np.float32)
