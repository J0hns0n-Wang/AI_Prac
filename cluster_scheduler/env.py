"""Gymnasium environment for cluster scheduling."""

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cluster_scheduler.featurizers import BasicFeaturizer, Featurizer, SetFeaturizer
from cluster_scheduler.models import Job, Machine
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig, WorkloadGenerator


DURATION_REF: float = 100.0
"""Fixed reference for normalizing job durations in observations.

Decoupled from any particular ``WorkloadConfig.duration_range`` so that a policy
trained on one workload still produces in-range observations on held-out
workloads with different duration distributions.
"""


@dataclass
class RewardConfig:
    """Configuration for the environment's reward function.

    Attributes:
        mode: ``"dense"`` emits a reward on every placement step
            (``-wait_penalty_weight * waiting_time`` minus an optional
            backlog penalty). ``"sparse"`` emits 0 on every step except
            optionally at termination.
        wait_penalty_weight: Multiplier on the placed job's waiting time
            (dense mode only).
        backlog_penalty_weight: Multiplier on ``len(wait_queue)`` added as a
            per-step penalty in dense mode. Default 0 disables it.
        completion_bonus: Reward added on the terminal step (both modes).
    """

    mode: Literal["dense", "sparse"] = "dense"
    wait_penalty_weight: float = 1.0
    backlog_penalty_weight: float = 0.0
    completion_bonus: float = 0.0


class ClusterSchedulingEnv(gym.Env):
    """Gymnasium environment that wraps the cluster simulator.

    Decision semantics:
        The agent chooses the *machine*. The env chooses the *job*: it picks
        the first queued job (FIFO) that can fit on at least one machine in
        the current cluster state. Jobs that cannot fit anywhere remain in the
        queue; the sim advances to the next completion event and re-checks.
        This keeps the action space small (``Discrete(num_machines)``) and lets
        a categorical policy reason purely about placement.

    Observation: fixed-length vector containing:
        - Per machine: (cpu_utilization, memory_utilization) × num_machines
        - Current job: (normalized_cpu, normalized_memory, normalized_duration)
        - System stats: (normalized_queue_length, overall_cpu_utilization)

    Action: Discrete(num_machines) — index of the machine to place the job on.

    Action masking:
        Use ``action_masks()`` (sb3-contrib MaskablePPO convention) to restrict
        the policy to machines that can fit the current job. ``info["action_mask"]``
        is also returned for callers that prefer the info-dict route.
        The env does not validate the action in ``step`` — call sites that
        bypass masking must supply a fittable machine.

    Reward: configurable via ``RewardConfig``. Default is dense and returns
            ``-waiting_time`` on each placement.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_config: SimulatorConfig | None = None,
        workload_config: WorkloadConfig | None = None,
        reward_config: RewardConfig | None = None,
        featurizer: Featurizer | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.sim_config = sim_config or SimulatorConfig()
        self.workload_config = workload_config or WorkloadConfig()
        self.reward_config = reward_config or RewardConfig()
        self.featurizer = featurizer or BasicFeaturizer(duration_ref=DURATION_REF)
        self.seed_value = seed

        num_machines = self.sim_config.num_machines

        # Observation space is driven by the featurizer so ablations can
        # swap layouts without editing the env.
        obs_shape = self.featurizer.observation_shape(self.sim_config)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

        # Action space: normally Discrete(num_machines). SetFeaturizer pads
        # observations to MAX_MACHINES so a single trained policy can run on
        # any num_machines <= MAX; action_space must follow suit and unused
        # slots are masked off by action_masks().
        if isinstance(self.featurizer, SetFeaturizer):
            if num_machines > SetFeaturizer.MAX_MACHINES:
                raise ValueError(
                    f"SetFeaturizer supports at most MAX_MACHINES="
                    f"{SetFeaturizer.MAX_MACHINES} machines, got {num_machines}."
                )
            self.action_space = spaces.Discrete(SetFeaturizer.MAX_MACHINES)
        else:
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
        """Build the observation vector via the configured featurizer, clipped to [0, 1]."""
        obs = self.featurizer.featurize(
            machines=self.machines,
            current_job=self.current_job,
            wait_queue=self.wait_queue,
            sim_config=self.sim_config,
            queue_capacity=self.workload_config.num_jobs,
        )
        return np.clip(obs, 0.0, 1.0)

    def _get_action_mask(self) -> np.ndarray:
        """Return a boolean mask of valid actions (machines that can fit the current job).

        Length matches ``action_space.n``. For SetFeaturizer that's
        ``MAX_MACHINES``; slots beyond the real cluster are always False.
        """
        mask_len = int(self.action_space.n)
        mask = np.zeros(mask_len, dtype=bool)
        if self.current_job is not None:
            for i, m in enumerate(self.machines):
                if i >= mask_len:
                    break
                mask[i] = m.can_fit(self.current_job)
        return mask

    def action_masks(self) -> np.ndarray:
        """Return the current action mask.

        This is the method name sb3-contrib's ``MaskablePPO`` / ``ActionMasker``
        look for. Returns a boolean array of length ``num_machines`` where
        ``True`` means the machine can fit the current job.
        """
        return self._get_action_mask()

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

        # Callers must supply a fittable machine. Under MaskablePPO this is
        # guaranteed by the action mask; direct callers should consult
        # action_masks() before stepping.
        if not machine.can_fit(self.current_job):
            raise ValueError(
                f"Action {action} chose Machine {machine.machine_id} which cannot "
                f"fit Job {self.current_job.job_id} (cpu={self.current_job.cpu}, "
                f"mem={self.current_job.memory}). Respect action_masks()."
            )

        # Place the job
        waiting_time = self.current_time - self.current_job.arrival_time
        machine.place_job(self.current_job, self.current_time)

        # Advance to next decision point
        has_next = self._advance_to_next_decision()
        terminated = not has_next

        reward = self._compute_reward(waiting_time=waiting_time, terminated=terminated)

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            {"action_mask": self._get_action_mask()},
        )

    def _compute_reward(self, waiting_time: float, terminated: bool) -> float:
        """Compute the step reward from ``self.reward_config``.

        Dense mode returns ``-wait_penalty_weight * waiting_time`` minus an
        optional per-step backlog penalty. Sparse mode returns 0 on non-terminal
        steps. Both modes add ``completion_bonus`` on the terminal step.
        """
        cfg = self.reward_config
        if cfg.mode == "sparse":
            reward = 0.0
        else:
            reward = -cfg.wait_penalty_weight * waiting_time
            if cfg.backlog_penalty_weight:
                reward -= cfg.backlog_penalty_weight * len(self.wait_queue)
        if terminated and cfg.completion_bonus:
            reward += cfg.completion_bonus
        return float(reward)
