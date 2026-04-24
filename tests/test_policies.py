"""Tests for SetFeaturizer, set-attention extractor, and the custom policy."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_scheduler.env import ClusterSchedulingEnv
from cluster_scheduler.featurizers import SetFeaturizer
from cluster_scheduler.models import Job, Machine
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


def _sim_config(num_machines: int) -> SimulatorConfig:
    return SimulatorConfig(num_machines=num_machines, cpu_per_machine=8, memory_per_machine=16)


def _make_machines(sc: SimulatorConfig) -> list[Machine]:
    return [
        Machine(machine_id=i, cpu_capacity=sc.cpu_per_machine, memory_capacity=sc.memory_per_machine)
        for i in range(sc.num_machines)
    ]


def _make_job(cpu: float = 2.0, memory: float = 4.0, duration: float = 5.0) -> Job:
    return Job(job_id=0, cpu=cpu, memory=memory, duration=duration, arrival_time=0.0)


class TestSetFeaturizer:
    def test_shape_is_independent_of_num_machines(self):
        sf = SetFeaturizer()
        # 197 = 32*6 + 3 + 2
        expected = (SetFeaturizer.MAX_MACHINES * SetFeaturizer.D_M
                    + SetFeaturizer.D_J + SetFeaturizer.D_G,)
        assert sf.observation_shape(_sim_config(4)) == expected
        assert sf.observation_shape(_sim_config(10)) == expected
        assert sf.observation_shape(_sim_config(20)) == expected

    def test_rejects_over_max_machines(self):
        sc = SimulatorConfig(num_machines=SetFeaturizer.MAX_MACHINES + 1)
        sf = SetFeaturizer()
        machines = [Machine(machine_id=i, cpu_capacity=8, memory_capacity=16)
                    for i in range(sc.num_machines)]
        with pytest.raises(ValueError, match="MAX_MACHINES"):
            sf.featurize(machines, _make_job(), [], sc, queue_capacity=100)

    def test_is_active_bits_match_num_machines(self):
        """Feature index 4 of each machine block is is_active (1 real / 0 pad)."""
        for n in (4, 10, 20):
            sc = _sim_config(n)
            sf = SetFeaturizer()
            obs = sf.featurize(_make_machines(sc), _make_job(), [], sc, queue_capacity=100)
            machine_block = obs[: SetFeaturizer.MAX_MACHINES * SetFeaturizer.D_M].reshape(
                SetFeaturizer.MAX_MACHINES, SetFeaturizer.D_M
            )
            is_active = machine_block[:, 4]
            assert int(is_active.sum()) == n
            assert is_active[:n].min() == 1.0
            assert is_active[n:].max() == 0.0

    def test_values_stay_in_unit_range(self):
        sc = _sim_config(10)
        sf = SetFeaturizer()
        obs = sf.featurize(
            _make_machines(sc),
            _make_job(cpu=3.0, memory=8.0, duration=9.0),
            [],
            sc,
            queue_capacity=100,
        )
        assert float(obs.min()) >= 0.0
        assert float(obs.max()) <= 1.0

    def test_can_fit_bit_toggles(self):
        sc = _sim_config(2)
        machines = _make_machines(sc)
        # Fill machine 0 so the probe can't fit there.
        machines[0].place_job(_make_job(cpu=8.0, memory=0.0, duration=10.0), current_time=0.0)
        sf = SetFeaturizer()
        obs = sf.featurize(machines, _make_job(cpu=2.0, memory=4.0, duration=5.0),
                           [], sc, queue_capacity=100)
        mb = obs[: SetFeaturizer.MAX_MACHINES * SetFeaturizer.D_M].reshape(
            SetFeaturizer.MAX_MACHINES, SetFeaturizer.D_M
        )
        assert mb[0, 5] == 0.0  # can_fit=0 for full machine 0
        assert mb[1, 5] == 1.0  # can_fit=1 for empty machine 1


class TestEnvWithSetFeaturizer:
    def test_action_space_is_max_machines(self):
        env = ClusterSchedulingEnv(
            sim_config=_sim_config(5),
            workload_config=WorkloadConfig(num_jobs=20),
            featurizer=SetFeaturizer(),
        )
        assert env.action_space.n == SetFeaturizer.MAX_MACHINES
        assert env.observation_space.shape == (
            SetFeaturizer.MAX_MACHINES * SetFeaturizer.D_M
            + SetFeaturizer.D_J + SetFeaturizer.D_G,
        )

    def test_action_mask_is_padded(self):
        env = ClusterSchedulingEnv(
            sim_config=_sim_config(3),
            workload_config=WorkloadConfig(num_jobs=20),
            featurizer=SetFeaturizer(),
        )
        env.reset(seed=0)
        mask = env.action_masks()
        assert mask.shape == (SetFeaturizer.MAX_MACHINES,)
        # Slots beyond num_machines (=3) must be False.
        assert not mask[3:].any()

    def test_rejects_num_machines_over_max(self):
        with pytest.raises(ValueError, match="MAX_MACHINES"):
            ClusterSchedulingEnv(
                sim_config=_sim_config(SetFeaturizer.MAX_MACHINES + 1),
                workload_config=WorkloadConfig(num_jobs=10),
                featurizer=SetFeaturizer(),
            )

    def test_full_episode_completes(self):
        env = ClusterSchedulingEnv(
            sim_config=_sim_config(5),
            workload_config=WorkloadConfig(
                num_jobs=15, arrival_rate=2.0,
                cpu_range=(1, 3), memory_range=(1, 4), duration_range=(2, 6),
            ),
            featurizer=SetFeaturizer(),
            seed=0,
        )
        _, info = env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 1000:
            mask = env.action_masks()
            action = int(np.where(mask)[0][0])
            _, _, terminated, _, info = env.step(action)
            steps += 1
        assert terminated
        assert len(env.completed_jobs) == 15


# ── Set-attention extractor + policy (require torch, so lazy-import). ──


torch = pytest.importorskip("torch")


class TestSetAttentionExtractor:
    def _build_extractor(self):
        from cluster_scheduler.policies import SetAttentionExtractor
        from gymnasium import spaces
        obs_space = spaces.Box(0.0, 1.0, shape=(
            SetFeaturizer.MAX_MACHINES * SetFeaturizer.D_M
            + SetFeaturizer.D_J + SetFeaturizer.D_G,
        ), dtype=np.float32)
        return SetAttentionExtractor(
            obs_space,
            embed_dim=16, n_heads=2, n_layers=1,  # tiny for speed
        )

    def _make_obs(self, n_active: int, batch: int = 3, seed: int = 0) -> "torch.Tensor":
        rng = np.random.default_rng(seed)
        obs = np.zeros(
            (batch, SetFeaturizer.MAX_MACHINES * SetFeaturizer.D_M
             + SetFeaturizer.D_J + SetFeaturizer.D_G),
            dtype=np.float32,
        )
        # Random per-machine features for active slots.
        for b in range(batch):
            for m in range(n_active):
                base = m * SetFeaturizer.D_M
                obs[b, base : base + SetFeaturizer.D_M] = rng.random(SetFeaturizer.D_M).astype(np.float32)
                obs[b, base + 4] = 1.0  # is_active
        # Job + global features
        obs[:, -(SetFeaturizer.D_J + SetFeaturizer.D_G):] = rng.random(
            (batch, SetFeaturizer.D_J + SetFeaturizer.D_G)
        ).astype(np.float32)
        return torch.as_tensor(obs)

    def test_output_shape(self):
        ext = self._build_extractor()
        obs = self._make_obs(n_active=10)
        out = ext(obs)
        expected_dim = SetFeaturizer.MAX_MACHINES * ext.embed_dim + SetFeaturizer.MAX_MACHINES
        assert out.shape == (3, expected_dim)

    def test_is_active_tail_is_preserved(self):
        """Extractor appends is_active bits as the tail of its output."""
        ext = self._build_extractor()
        obs = self._make_obs(n_active=7)
        out = ext(obs)
        tail = out[:, SetFeaturizer.MAX_MACHINES * ext.embed_dim :]
        # First 7 slots real, rest padding.
        assert tail.shape == (3, SetFeaturizer.MAX_MACHINES)
        assert (tail[:, :7] > 0.5).all()
        assert (tail[:, 7:] < 0.5).all()

    def test_padded_machines_do_not_change_active_embeddings(self):
        """Permutation-like check: zeroing padded slot features leaves active embeddings intact."""
        ext = self._build_extractor()
        ext.eval()
        obs = self._make_obs(n_active=5, seed=42)
        with torch.no_grad():
            out_a = ext(obs)

        # Mutate *padded* slots' features (they're already zero anyway but make them noisy),
        # keeping is_active=0 so attention still ignores them.
        obs2 = obs.clone()
        pad_slice = slice(5 * SetFeaturizer.D_M, SetFeaturizer.MAX_MACHINES * SetFeaturizer.D_M)
        noise = torch.randn_like(obs2[:, pad_slice])
        obs2[:, pad_slice] = noise
        # Force is_active=0 on padded slots.
        for m in range(5, SetFeaturizer.MAX_MACHINES):
            obs2[:, m * SetFeaturizer.D_M + 4] = 0.0
        with torch.no_grad():
            out_b = ext(obs2)

        # The per-machine embeddings for active slots (0..4) should be unchanged
        # because the attention key_padding_mask zeros out padded contributions.
        active_dim = 5 * ext.embed_dim
        torch.testing.assert_close(out_a[:, :active_dim], out_b[:, :active_dim])


# ── Slow training smoke through the real SetMaskablePolicy. ──────────

pytest.importorskip("sb3_contrib")


@pytest.mark.slow
class TestSetMaskablePolicyTraining:
    def test_smoke_train_and_sidecar(self, tmp_path):
        import json

        from cluster_scheduler.train import main as train_main

        save_path = tmp_path / "set_smoke.zip"
        train_main([
            "--timesteps", "256",
            "--seed", "0",
            "--num-machines", "3",
            "--num-jobs", "10",
            "--arrival-rate", "2.0",
            "--policy", "set_attention",
            "--n-steps", "64",
            "--batch-size", "32",
            "--n-epochs", "1",
            "--device", "cpu",
            "--log-dir", str(tmp_path / "runs"),
            "--save-path", str(save_path),
        ])
        assert save_path.exists()
        meta = json.loads((tmp_path / "set_smoke.zip.meta.json").read_text())
        assert meta["policy"] == "set_attention"
        assert meta["featurizer"]["name"] == "set"

    def test_cross_cluster_predict(self, tmp_path):
        """A set-attention policy trained at num_machines=3 can predict on num_machines=5."""
        from sb3_contrib import MaskablePPO

        from cluster_scheduler.env import ClusterSchedulingEnv
        from cluster_scheduler.train import main as train_main

        save_path = tmp_path / "set_cross.zip"
        train_main([
            "--timesteps", "256",
            "--seed", "0",
            "--num-machines", "3",
            "--num-jobs", "10",
            "--arrival-rate", "2.0",
            "--policy", "set_attention",
            "--n-steps", "64",
            "--batch-size", "32",
            "--n-epochs", "1",
            "--device", "cpu",
            "--log-dir", str(tmp_path / "runs"),
            "--save-path", str(save_path),
        ])
        model = MaskablePPO.load(str(save_path), device="cpu")
        # Build a 5-machine env with the same featurizer.
        env = ClusterSchedulingEnv(
            sim_config=_sim_config(5),
            workload_config=WorkloadConfig(num_jobs=10, arrival_rate=2.0),
            featurizer=SetFeaturizer(),
            seed=1,
        )
        obs, _ = env.reset()
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        assert env.action_masks()[int(action)]
