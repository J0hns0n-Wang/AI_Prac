"""Gymnasium environment for cluster scheduling."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cluster_scheduler.models import Job, Machine
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig, WorkloadGenerator


class ClusterSchedulingEnv(gym.Env):
    """Gymnasium environment that wraps the cluster simulator.

    At each step, the agent picks which machine to place the current job on.
    The episode ends when all jobs have been placed and completed.

    Observation: fixed-length vector containing:
        - Per machine: (cpu_utilization, memory_utilization) × num_machines
        - Current job: (normalized_cpu, normalized_memory, normalized_duration)
        - System stats: (normalized_queue_length, overall_cpu_utilization)

    Action: Discrete(num_machines) — index of the machine to place the job on.

    Reward: negative waiting time for the placed job (0 if placed immediately,
            negative if it had to wait). Encourages minimizing delays.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_config: SimulatorConfig | None = None,
        workload_config: WorkloadConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.sim_config = sim_config or SimulatorConfig()
        self.workload_config = workload_config or WorkloadConfig()
        self.seed_value = seed

        num_machines = self.sim_config.num_machines

        # Observation: per-machine (cpu_util, mem_util) + job (cpu, mem, dur) + system (queue_len, cluster_util)
        obs_size = num_machines * 2 + 3 + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Action: pick a machine
        self.action_space = spaces.Discrete(num_machines)

        # Internal state
        self.machines: list[Machine] = []
        self.jobs: list[Job] = []
        self.job_index: int = 0
        self.wait_queue: list[Job] = []
        self.completed_jobs: list[Job] = []
        self.current_time: float = 0.0
        self.current_job: Job | None = None

    def _init_machines(self) -> None:
        cfg = self.sim_config
        self.machines = [
            Machine(
                machine_id=i,
                cpu_capacity=cfg.cpu_per_machine,
                memory_capacity=cfg.memory_per_machine,
            )
            for i in range(cfg.num_machines)
        ]

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        obs = []

        # Per-machine utilization
        for m in self.machines:
            obs.append(m.cpu_utilization)
            obs.append(m.memory_utilization)

        # Current job features (normalized by machine capacity)
        if self.current_job is not None:
            obs.append(self.current_job.cpu / self.sim_config.cpu_per_machine)
            obs.append(self.current_job.memory / self.sim_config.memory_per_machine)
            obs.append(self.current_job.duration / self.workload_config.duration_range[1])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # System stats
        max_queue = self.workload_config.num_jobs
        obs.append(len(self.wait_queue) / max_queue if max_queue > 0 else 0.0)

        total_cpu = sum(m.cpu_utilization for m in self.machines)
        obs.append(total_cpu / len(self.machines) if self.machines else 0.0)

        return np.array(obs, dtype=np.float32)

    def _get_action_mask(self) -> np.ndarray:
        """Return a boolean mask of valid actions (machines that can fit the current job)."""
        mask = np.zeros(self.sim_config.num_machines, dtype=bool)
        if self.current_job is not None:
            for i, m in enumerate(self.machines):
                mask[i] = m.can_fit(self.current_job)
        return mask

    def _release_completed_jobs(self) -> None:
        """Release all jobs that have finished by current_time."""
        for machine in self.machines:
            completed = machine.release_completed_jobs(self.current_time)
            self.completed_jobs.extend(completed)

    def _advance_to_next_decision(self) -> bool:
        """Advance the simulation to the next point where the agent needs to act.

        Returns True if there's a job to place, False if the episode is over.
        """
        while True:
            # Try to serve from the wait queue first
            if self.wait_queue:
                # Find a job in the queue that can fit somewhere
                for i, job in enumerate(self.wait_queue):
                    if any(m.can_fit(job) for m in self.machines):
                        self.current_job = self.wait_queue.pop(i)
                        return True

            # Check if there are more arriving jobs
            if self.job_index < len(self.jobs):
                next_arrival = self.jobs[self.job_index].arrival_time

                # Check if a completion happens before the next arrival
                next_completion = self._next_completion_time()
                if next_completion is not None and next_completion <= next_arrival:
                    self.current_time = next_completion
                    self._release_completed_jobs()
                    continue  # Re-check queue with freed resources

                # Advance to arrival
                self.current_time = next_arrival
                self._release_completed_jobs()

                job = self.jobs[self.job_index]
                self.job_index += 1

                # Can this job fit anywhere?
                if any(m.can_fit(job) for m in self.machines):
                    self.current_job = job
                    return True
                else:
                    self.wait_queue.append(job)
                    continue

            # No more arrivals — advance to next completion to free resources
            next_completion = self._next_completion_time()
            if next_completion is not None:
                self.current_time = next_completion
                self._release_completed_jobs()

                # Check if any queued job can now fit
                if self.wait_queue:
                    continue

            # Episode over: no more jobs to arrive or place
            if not self.wait_queue and self.job_index >= len(self.jobs):
                # Wait for remaining running jobs to finish
                while self._next_completion_time() is not None:
                    self.current_time = self._next_completion_time()
                    self._release_completed_jobs()
                self.current_job = None
                return False

            # Safety: if nothing can progress, we're stuck
            if next_completion is None and self.job_index >= len(self.jobs):
                self.current_job = None
                return False

    def _next_completion_time(self) -> float | None:
        earliest = None
        for machine in self.machines:
            for job in machine.running_jobs:
                if earliest is None or job.completion_time < earliest:
                    earliest = job.completion_time
        return earliest

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate workload
        wl_seed = seed if seed is not None else self.seed_value
        self.jobs = WorkloadGenerator(self.workload_config, seed=wl_seed).generate()

        # Reset state
        self._init_machines()
        self.job_index = 0
        self.wait_queue = []
        self.completed_jobs = []
        self.current_time = 0.0
        self.current_job = None

        self._advance_to_next_decision()
        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def step(self, action: int):
        assert self.current_job is not None, "No job to place — episode should have ended"

        machine = self.machines[action]

        # If invalid action, penalize and let agent try again
        if not machine.can_fit(self.current_job):
            return (
                self._get_obs(),
                -1.0,  # penalty for invalid action
                False,
                False,
                {"action_mask": self._get_action_mask(), "invalid_action": True},
            )

        # Place the job
        waiting_time = self.current_time - self.current_job.arrival_time
        machine.place_job(self.current_job, self.current_time)

        # Reward: negative waiting time (0 is best, more negative = worse)
        reward = -waiting_time

        # Advance to next decision point
        has_next = self._advance_to_next_decision()
        terminated = not has_next

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            {"action_mask": self._get_action_mask()},
        )
