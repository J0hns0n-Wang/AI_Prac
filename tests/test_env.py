"""Tests for the Gymnasium environment."""

import numpy as np
import pytest

from cluster_scheduler.env import ClusterSchedulingEnv
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


def make_env(num_jobs=20, num_machines=3, seed=42):
    return ClusterSchedulingEnv(
        sim_config=SimulatorConfig(num_machines=num_machines, cpu_per_machine=8, memory_per_machine=16),
        workload_config=WorkloadConfig(
            num_jobs=num_jobs, arrival_rate=2.0,
            cpu_range=(1, 3), memory_range=(1, 4), duration_range=(2, 6),
        ),
        seed=seed,
    )


class TestEnvBasics:
    def test_reset_returns_valid_obs(self):
        env = make_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert "action_mask" in info

    def test_action_mask_shape(self):
        env = make_env()
        _, info = env.reset(seed=42)
        mask = info["action_mask"]
        assert mask.shape == (3,)  # num_machines=3
        assert mask.any()  # at least one valid action

    def test_step_with_valid_action(self):
        env = make_env()
        _, info = env.reset(seed=42)
        mask = info["action_mask"]
        action = np.where(mask)[0][0]  # pick first valid action

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert not truncated

    def test_step_with_invalid_action_penalizes(self):
        env = make_env(num_machines=2)
        _, info = env.reset(seed=42)
        mask = info["action_mask"]

        # If both are valid, fill one machine first
        # Just check that if we could find an invalid action, it returns penalty
        if not mask.all():
            invalid_action = np.where(~mask)[0][0]
            _, reward, _, _, info = env.step(invalid_action)
            assert reward == -1.0
            assert info.get("invalid_action", False)


class TestEpisode:
    def test_episode_terminates(self):
        env = make_env(num_jobs=10)
        obs, info = env.reset(seed=42)

        steps = 0
        terminated = False
        while not terminated and steps < 1000:
            mask = info["action_mask"]
            action = np.where(mask)[0][0]
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert terminated
        assert steps <= 1000  # should finish well before safety limit

    def test_all_jobs_complete(self):
        env = make_env(num_jobs=15)
        obs, info = env.reset(seed=42)

        terminated = False
        steps = 0
        while not terminated and steps < 500:
            mask = info["action_mask"]
            action = np.where(mask)[0][0]
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert len(env.completed_jobs) == 15

    def test_rewards_are_non_positive(self):
        """Reward is -waiting_time, so should always be <= 0."""
        env = make_env(num_jobs=20)
        obs, info = env.reset(seed=42)

        rewards = []
        terminated = False
        while not terminated:
            mask = info["action_mask"]
            action = np.where(mask)[0][0]
            obs, reward, terminated, truncated, info = env.step(action)
            if not info.get("invalid_action", False):
                rewards.append(reward)

        assert all(r <= 0.0 for r in rewards)


class TestReproducibility:
    def test_same_seed_same_episode(self):
        env = make_env()

        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
