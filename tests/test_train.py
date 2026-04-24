"""Tests for the MaskablePPO training entrypoint.

All tests are marked ``slow`` because importing ``sb3_contrib`` and
``stable_baselines3`` pulls in PyTorch, which is expensive on cold start.
Run with ``pytest -m slow``.
"""

import pytest

pytest.importorskip("sb3_contrib")

from pathlib import Path

import numpy as np

from cluster_scheduler.env import RewardConfig
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.train import make_env, train_maskable_ppo
from cluster_scheduler.workload import WorkloadConfig


pytestmark = pytest.mark.slow


def _small_factory(seed=0):
    return make_env(
        sim_config=SimulatorConfig(num_machines=3, cpu_per_machine=8, memory_per_machine=16),
        workload_config=WorkloadConfig(
            num_jobs=15, arrival_rate=2.0,
            cpu_range=(1, 3), memory_range=(1, 4), duration_range=(2, 6),
        ),
        reward_config=RewardConfig(mode="dense"),
        seed=seed,
    )


class TestMakeEnv:
    def test_factory_returns_env_with_action_masks(self):
        env = _small_factory()()
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        # ActionMasker exposes action_masks() on the wrapped env.
        mask = env.action_masks()
        assert mask.shape == (3,)
        assert mask.dtype == bool


class TestTraining:
    def test_smoke_training_runs(self, tmp_path: Path):
        model = train_maskable_ppo(
            _small_factory(seed=0),
            total_timesteps=512,
            seed=0,
            log_dir=str(tmp_path / "runs"),
            save_path=str(tmp_path / "ppo.zip"),
            ppo_kwargs={"n_steps": 128, "batch_size": 32, "n_epochs": 2},
        )
        assert (tmp_path / "ppo.zip").exists()
        # Monitor CSV present
        runs = tmp_path / "runs"
        assert any(runs.glob("monitor_seed0*.csv"))
        # Can predict a valid action
        env = _small_factory(seed=0)()
        obs, _ = env.reset()
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        assert env.action_masks()[int(action)]

    def test_save_load_roundtrip(self, tmp_path: Path):
        from sb3_contrib import MaskablePPO

        save_path = tmp_path / "ppo.zip"
        model = train_maskable_ppo(
            _small_factory(seed=1),
            total_timesteps=128,
            seed=1,
            log_dir=None,
            save_path=str(save_path),
            ppo_kwargs={"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        )

        env = _small_factory(seed=1)()
        obs, _ = env.reset()
        mask = env.action_masks()

        action_before, _ = model.predict(obs, action_masks=mask, deterministic=True)
        loaded = MaskablePPO.load(str(save_path))
        action_after, _ = loaded.predict(obs, action_masks=mask, deterministic=True)
        assert int(action_before) == int(action_after)

    def test_sidecar_written_with_expected_fields(self, tmp_path: Path):
        import json

        from cluster_scheduler.train import main as train_main

        save_path = tmp_path / "ppo.zip"
        log_dir = tmp_path / "runs"
        train_main([
            "--timesteps", "128",
            "--seed", "0",
            "--num-machines", "3",
            "--num-jobs", "10",
            "--arrival-rate", "2.0",
            "--featurizer", "rich",
            "--rich-top-k", "2",
            "--reward-mode", "dense",
            "--backlog-penalty-weight", "0.1",
            "--n-steps", "64",
            "--batch-size", "32",
            "--n-epochs", "1",
            "--log-dir", str(log_dir),
            "--save-path", str(save_path),
        ])
        sidecar = tmp_path / "ppo.zip.meta.json"
        assert save_path.exists()
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["featurizer"] == {"name": "rich", "top_k": 2, "duration_ref": 100.0}
        assert meta["num_machines"] == 3
        assert meta["reward_config"]["mode"] == "dense"
        assert meta["reward_config"]["backlog_penalty_weight"] == 0.1
        assert meta["ppo_kwargs"]["n_steps"] == 64

    def test_n_envs_and_policy_kwargs_roundtrip(self, tmp_path: Path):
        """--n-envs > 1 + custom MLP arch both take effect and persist in the sidecar."""
        import json

        from cluster_scheduler.train import main as train_main

        save_path = tmp_path / "ppo_big.zip"
        train_main([
            "--timesteps", "256",
            "--seed", "0",
            "--num-machines", "3",
            "--num-jobs", "10",
            "--arrival-rate", "2.0",
            "--n-envs", "2",
            "--device", "cpu",
            "--policy-hidden", "32", "32",
            "--activation", "gelu",
            "--n-steps", "64",
            "--batch-size", "32",
            "--n-epochs", "1",
            "--log-dir", str(tmp_path / "runs"),
            "--save-path", str(save_path),
        ])
        sidecar = tmp_path / "ppo_big.zip.meta.json"
        assert save_path.exists()
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["n_envs"] == 2
        assert meta["device"] == "cpu"
        assert meta["policy_kwargs"]["net_arch"] == [32, 32]
        assert meta["policy_kwargs"]["activation_fn"] == "gelu"

    def test_normalize_reward_and_resume_from(self, tmp_path: Path):
        """--normalize-reward writes stats; --resume-from warm-starts from a .zip."""
        import json

        from cluster_scheduler.train import main as train_main

        # First training pass with VecNormalize reward stats.
        seed_path = tmp_path / "seed.zip"
        train_main([
            "--timesteps", "128",
            "--seed", "0",
            "--num-machines", "3",
            "--num-jobs", "10",
            "--arrival-rate", "2.0",
            "--normalize-reward",
            "--n-steps", "64",
            "--batch-size", "32",
            "--n-epochs", "1",
            "--log-dir", str(tmp_path / "runs_seed"),
            "--save-path", str(seed_path),
        ])
        stats = tmp_path / "seed.zip.vecnormalize.pkl"
        seed_meta = json.loads((tmp_path / "seed.zip.meta.json").read_text())
        assert seed_path.exists()
        assert stats.exists(), "VecNormalize stats must be saved next to the .zip"
        assert seed_meta["normalize_reward"] is True

        # Fine-tune from the seed checkpoint with a linear LR schedule.
        out_path = tmp_path / "tuned.zip"
        train_main([
            "--timesteps", "128",
            "--seed", "1",
            "--num-machines", "3",
            "--num-jobs", "10",
            "--arrival-rate", "2.0",
            "--normalize-reward",
            "--learning-rate", "1e-4",
            "--lr-schedule", "linear",
            "--clip-range-vf", "0.2",
            "--n-steps", "64",
            "--batch-size", "32",
            "--n-epochs", "1",
            "--resume-from", str(seed_path),
            "--log-dir", str(tmp_path / "runs_tuned"),
            "--save-path", str(out_path),
        ])
        assert out_path.exists()
        meta = json.loads((tmp_path / "tuned.zip.meta.json").read_text())
        assert meta["lr_schedule"] == "linear"
        assert meta["resume_from"] == str(seed_path)
        assert meta["ppo_kwargs"]["clip_range_vf"] == 0.2
        # learning_rate is a callable with a linear schedule, stringified.
        assert str(meta["ppo_kwargs"]["learning_rate"]).startswith("<callable:")

    def test_checkpoints_written(self, tmp_path: Path):
        save_path = tmp_path / "ppo.zip"
        train_maskable_ppo(
            _small_factory(seed=0),
            total_timesteps=256,
            seed=0,
            log_dir=None,
            save_path=str(save_path),
            checkpoint_freq=128,
            ppo_kwargs={"n_steps": 128, "batch_size": 32, "n_epochs": 1},
        )
        # CheckpointCallback writes files named "<stem>_<steps>_steps.zip"
        ckpts = list(tmp_path.glob("ppo_*_steps.zip"))
        assert len(ckpts) >= 1
