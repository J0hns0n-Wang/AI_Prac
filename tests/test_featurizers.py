"""Tests for the pluggable observation featurizers."""

import numpy as np
import pytest

from cluster_scheduler.env import DURATION_REF, ClusterSchedulingEnv
from cluster_scheduler.featurizers import BasicFeaturizer, RichFeaturizer
from cluster_scheduler.models import Job, Machine
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


def make_sim_config(num_machines=3):
    return SimulatorConfig(num_machines=num_machines, cpu_per_machine=8, memory_per_machine=16)


def make_machines(sim_config):
    return [
        Machine(machine_id=i, cpu_capacity=sim_config.cpu_per_machine, memory_capacity=sim_config.memory_per_machine)
        for i in range(sim_config.num_machines)
    ]


def make_job(job_id=0, cpu=2.0, memory=4.0, duration=5.0, arrival_time=0.0):
    return Job(job_id=job_id, cpu=cpu, memory=memory, duration=duration, arrival_time=arrival_time)


class TestBasicFeaturizer:
    def test_shape(self):
        sc = make_sim_config(num_machines=4)
        f = BasicFeaturizer()
        assert f.observation_shape(sc) == (4 * 2 + 3 + 2,)

    def test_values_match_documented_layout(self):
        sc = make_sim_config(num_machines=2)
        machines = make_machines(sc)
        # Load machine 0 half-full in CPU
        machines[0].place_job(make_job(cpu=4.0, memory=0.0, duration=1.0), current_time=0.0)

        current = make_job(cpu=2.0, memory=4.0, duration=5.0)
        queue: list[Job] = []

        f = BasicFeaturizer(duration_ref=DURATION_REF)
        obs = f.featurize(
            machines=machines,
            current_job=current,
            wait_queue=queue,
            sim_config=sc,
            queue_capacity=100,
        )
        # Layout: [m0_cpu_util, m0_mem_util, m1_cpu_util, m1_mem_util,
        #         job_cpu_norm, job_mem_norm, job_dur_norm,
        #         queue_frac, mean_cluster_cpu_util]
        expected = np.array(
            [
                4.0 / 8,  # m0 cpu util
                0.0,      # m0 mem util
                0.0,      # m1 cpu util
                0.0,      # m1 mem util
                2.0 / 8,  # job cpu / cpu_per_machine
                4.0 / 16, # job mem / mem_per_machine
                5.0 / DURATION_REF,
                0.0,      # empty queue / 100
                (0.5 + 0.0) / 2,  # mean cpu util
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(obs, expected, rtol=1e-6)

    def test_zero_job_when_none(self):
        sc = make_sim_config(num_machines=2)
        f = BasicFeaturizer()
        obs = f.featurize(make_machines(sc), None, [], sc, queue_capacity=10)
        # Job slots (idx 4,5,6) should be zero.
        assert obs[4] == 0.0 and obs[5] == 0.0 and obs[6] == 0.0

    def test_default_env_obs_shape_unchanged(self):
        """Back-compat: constructing env without a featurizer preserves obs shape."""
        env = ClusterSchedulingEnv(
            sim_config=make_sim_config(num_machines=5),
            workload_config=WorkloadConfig(num_jobs=10),
        )
        assert env.observation_space.shape == (5 * 2 + 3 + 2,)


class TestRichFeaturizer:
    def test_shape_formula(self):
        sc = make_sim_config(num_machines=4)
        f = RichFeaturizer(top_k=3)
        expected = (4 * 2 + 3 + 2) + 3 * 3 + 2 + 4 * 2
        assert f.observation_shape(sc) == (expected,)

    def test_embeds_basic_prefix_exactly(self):
        sc = make_sim_config(num_machines=2)
        machines = make_machines(sc)
        current = make_job(cpu=2.0, memory=4.0, duration=5.0)

        basic = BasicFeaturizer().featurize(
            machines, current, [], sc, queue_capacity=100
        )
        rich = RichFeaturizer(top_k=2).featurize(
            machines, current, [], sc, queue_capacity=100
        )
        np.testing.assert_array_equal(rich[: basic.shape[0]], basic)

    def test_queue_peek_zero_padded(self):
        sc = make_sim_config(num_machines=2)
        machines = make_machines(sc)
        current = make_job()
        f = RichFeaturizer(top_k=3)

        obs = f.featurize(machines, current, [], sc, queue_capacity=10)
        basic_len = BasicFeaturizer().observation_shape(sc)[0]
        # top-k × 3 features immediately after basic prefix, all zero when queue is empty.
        peek = obs[basic_len : basic_len + 3 * 3]
        np.testing.assert_array_equal(peek, np.zeros(9, dtype=np.float32))

    def test_queue_peek_reflects_fifo_head(self):
        sc = make_sim_config(num_machines=2)
        machines = make_machines(sc)
        current = make_job()
        queue = [
            make_job(job_id=1, cpu=1.0, memory=2.0, duration=3.0),
            make_job(job_id=2, cpu=2.0, memory=4.0, duration=6.0),
        ]
        f = RichFeaturizer(top_k=2)
        obs = f.featurize(machines, current, queue, sc, queue_capacity=10)
        basic_len = BasicFeaturizer().observation_shape(sc)[0]
        peek = obs[basic_len : basic_len + 6]
        expected = np.array(
            [
                1.0 / 8, 2.0 / 16, 3.0 / 100.0,
                2.0 / 8, 4.0 / 16, 6.0 / 100.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(peek, expected, rtol=1e-6)

    def test_headroom_reflects_current_job(self):
        sc = make_sim_config(num_machines=2)
        machines = make_machines(sc)
        current = make_job(cpu=2.0, memory=4.0, duration=5.0)
        f = RichFeaturizer(top_k=0)  # drop queue peek to simplify indexing
        obs = f.featurize(machines, current, [], sc, queue_capacity=10)
        basic_len = BasicFeaturizer().observation_shape(sc)[0]
        # After basic prefix: 0 queue peek + 2 backlog + (num_machines * 2) headroom
        headroom = obs[basic_len + 2 :]
        # Each machine should have cpu_free - job.cpu = 8 - 2 = 6 → 6/8 = 0.75
        # mem_free - job.mem = 16 - 4 = 12 → 12/16 = 0.75
        np.testing.assert_allclose(headroom, np.array([0.75, 0.75, 0.75, 0.75], dtype=np.float32))

    def test_bounds_under_clipping(self):
        """Featurizer values can go out-of-box; the env clips, so integration with env must stay in [0, 1]."""
        env = ClusterSchedulingEnv(
            sim_config=make_sim_config(num_machines=3),
            workload_config=WorkloadConfig(num_jobs=20),
            featurizer=RichFeaturizer(top_k=2),
        )
        obs, _ = env.reset(seed=7)
        assert env.observation_space.contains(obs)

    def test_rejects_negative_top_k(self):
        with pytest.raises(ValueError):
            RichFeaturizer(top_k=-1)


class TestEnvWithFeaturizer:
    def test_env_uses_rich_featurizer_shape(self):
        sc = make_sim_config(num_machines=3)
        featurizer = RichFeaturizer(top_k=2)
        env = ClusterSchedulingEnv(
            sim_config=sc,
            workload_config=WorkloadConfig(num_jobs=20),
            featurizer=featurizer,
        )
        expected = featurizer.observation_shape(sc)
        assert env.observation_space.shape == expected
        obs, _ = env.reset(seed=0)
        assert obs.shape == expected

    def test_env_step_with_rich_featurizer_runs(self):
        env = ClusterSchedulingEnv(
            sim_config=make_sim_config(num_machines=3),
            workload_config=WorkloadConfig(num_jobs=10),
            featurizer=RichFeaturizer(top_k=2),
        )
        _, info = env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated and steps < 500:
            mask = env.action_masks()
            action = int(np.where(mask)[0][0])
            _, _, terminated, _, info = env.step(action)
            steps += 1
        assert terminated
