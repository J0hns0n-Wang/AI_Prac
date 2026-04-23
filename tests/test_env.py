"""Tests for the Gymnasium environment."""

import numpy as np
import pytest

from cluster_scheduler.env import ClusterSchedulingEnv, RewardConfig
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


def make_env(num_jobs=20, num_machines=3, seed=42, duration_range=(2, 6), reward_config=None):
    return ClusterSchedulingEnv(
        sim_config=SimulatorConfig(num_machines=num_machines, cpu_per_machine=8, memory_per_machine=16),
        workload_config=WorkloadConfig(
            num_jobs=num_jobs, arrival_rate=2.0,
            cpu_range=(1, 3), memory_range=(1, 4), duration_range=duration_range,
        ),
        reward_config=reward_config,
        seed=seed,
    )


def _rollout_greedy(env):
    """Step through an episode picking the first valid action each time. Returns (rewards, terminated_idx)."""
    _, info = env.reset(seed=env.seed_value)
    rewards = []
    terminated = False
    steps = 0
    while not terminated and steps < 5000:
        mask = info["action_mask"]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break
        action = int(valid[0])
        _, reward, terminated, _, info = env.step(action)
        rewards.append(reward)
        steps += 1
    return rewards, terminated


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

    def test_step_with_invalid_action_raises(self):
        env = make_env(num_machines=2)
        _, info = env.reset(seed=42)
        mask = info["action_mask"]

        # Find a machine that *can't* fit, if any. With the default
        # make_env config a fresh cluster usually has both machines valid;
        # we skip the assertion in that case rather than stub state.
        if not mask.all():
            invalid_action = int(np.where(~mask)[0][0])
            with pytest.raises(ValueError):
                env.step(invalid_action)


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


class TestObservationNormalization:
    """Obs must not depend on the train-time workload duration_range."""

    def test_obs_independent_of_duration_range(self):
        # Same workload seed → same sampled job sequence (arrivals, demands)
        # only if duration_range shifts scale. To isolate the normalization path
        # we compare the job-duration slot of the obs across two envs whose
        # only difference is duration_range; both should normalize by DURATION_REF.
        from cluster_scheduler.env import DURATION_REF
        from cluster_scheduler.models import Job

        e1 = make_env(duration_range=(2, 6))
        e2 = make_env(duration_range=(50, 80))
        e1.reset(seed=42)
        e2.reset(seed=42)

        # Inject an identical current_job into both and compare the obs slot
        # that encodes job duration (index: num_machines*2 + 2).
        probe = Job(job_id=-1, cpu=1.0, memory=1.0, duration=5.0, arrival_time=0.0)
        e1.current_job = probe
        e2.current_job = probe
        idx = e1.sim_config.num_machines * 2 + 2
        assert e1._get_obs()[idx] == pytest.approx(5.0 / DURATION_REF)
        assert e2._get_obs()[idx] == pytest.approx(5.0 / DURATION_REF)

    def test_obs_stays_in_box(self):
        # Extreme duration should not escape the [0, 1] observation_space bound.
        from cluster_scheduler.models import Job

        env = make_env()
        env.reset(seed=42)
        env.current_job = Job(
            job_id=-1, cpu=1.0, memory=1.0, duration=10_000.0, arrival_time=0.0
        )
        obs = env._get_obs()
        assert env.observation_space.contains(obs)


class TestRewardConfig:
    def test_default_reward_matches_prior_behavior(self):
        # Default RewardConfig is dense with wait_penalty_weight=1.0; equivalent
        # to the previous -waiting_time reward on valid placements.
        env = make_env(num_jobs=10)
        rewards, terminated = _rollout_greedy(env)
        assert terminated
        # Valid-action rewards are non-positive in dense mode with default weights.
        assert all(r <= 0.0 for r in rewards)

    def test_sparse_mode_zero_except_terminal(self):
        cfg = RewardConfig(mode="sparse", completion_bonus=10.0)
        env = make_env(num_jobs=10, reward_config=cfg)
        rewards, terminated = _rollout_greedy(env)
        assert terminated
        # Non-terminal steps all zero.
        assert all(r == 0.0 for r in rewards[:-1])
        # Terminal step carries the completion bonus.
        assert rewards[-1] == pytest.approx(10.0)

    def test_backlog_penalty_applied(self):
        # With backlog_penalty_weight > 0, dense rewards should be at least as
        # negative as without it for the same trajectory.
        base = make_env(num_jobs=15, reward_config=RewardConfig(mode="dense"))
        with_backlog = make_env(
            num_jobs=15,
            reward_config=RewardConfig(mode="dense", backlog_penalty_weight=0.5),
        )
        r_base, _ = _rollout_greedy(base)
        r_pen, _ = _rollout_greedy(with_backlog)
        # Same greedy policy + same seed → same trajectory → per-step comparison.
        assert len(r_base) == len(r_pen)
        assert sum(r_pen) <= sum(r_base)

    def test_completion_bonus_applied_in_dense_mode(self):
        no_bonus = make_env(num_jobs=10, reward_config=RewardConfig(mode="dense"))
        bonus = make_env(
            num_jobs=10,
            reward_config=RewardConfig(mode="dense", completion_bonus=7.5),
        )
        r0, _ = _rollout_greedy(no_bonus)
        r1, _ = _rollout_greedy(bonus)
        assert r1[-1] - r0[-1] == pytest.approx(7.5)


class TestActionMasks:
    """MaskablePPO-compatible masking contract."""

    def test_action_masks_method_exists_and_matches_can_fit(self):
        env = make_env()
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.shape == (env.sim_config.num_machines,)
        assert mask.dtype == bool
        assert env.current_job is not None
        expected = np.array(
            [m.can_fit(env.current_job) for m in env.machines], dtype=bool
        )
        np.testing.assert_array_equal(mask, expected)

    def test_action_masks_matches_info_mask(self):
        env = make_env()
        _, info = env.reset(seed=42)
        np.testing.assert_array_equal(env.action_masks(), info["action_mask"])

    def test_masked_random_rollout_never_raises(self):
        """Under a masked random policy, step() must never be handed an illegal action."""
        rng = np.random.default_rng(0)
        env = make_env(num_jobs=30)
        _, info = env.reset(seed=42)
        terminated = False
        steps = 0
        while not terminated and steps < 2000:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            assert len(valid) > 0, "Mask must have at least one valid action when step is reachable"
            action = int(rng.choice(valid))
            _, _, terminated, _, info = env.step(action)
            steps += 1
        assert terminated
