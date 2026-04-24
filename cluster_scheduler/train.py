"""MaskablePPO training entrypoint for the cluster scheduling env.

Usage (library):
    from cluster_scheduler.train import make_env, train_maskable_ppo

    factory = make_env(
        sim_config=SimulatorConfig(num_machines=10),
        workload_config=WorkloadConfig(num_jobs=100, arrival_rate=2.0),
        seed=0,
    )
    model = train_maskable_ppo(
        factory,
        total_timesteps=200_000,
        seed=0,
        log_dir="runs/ppo_run",
        save_path="artifacts/ppo.zip",
    )

Usage (CLI):
    python -m cluster_scheduler.train --timesteps 200000 \
        --num-jobs 100 --num-machines 10 \
        --log-dir runs/ppo_run --save-path artifacts/ppo.zip
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from cluster_scheduler.env import ClusterSchedulingEnv, RewardConfig
from cluster_scheduler.featurizers import Featurizer
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


EnvFactory = Callable[[], gym.Env]

# TensorBoard is optional. SB3 raises if tensorboard_log is set but the
# tensorboard package isn't installed, so we only enable it when the
# dependency is actually present. Monitor CSV logging is independent.
_HAS_TENSORBOARD: bool = importlib.util.find_spec("tensorboard") is not None


def _action_mask_fn(env: ClusterSchedulingEnv):
    """Mask extractor for ActionMasker. Unwraps through Monitor if needed."""
    inner = env
    while hasattr(inner, "env") and not hasattr(inner, "action_masks"):
        inner = inner.env
    return inner.action_masks()


def make_env(
    sim_config: SimulatorConfig | None = None,
    workload_config: WorkloadConfig | None = None,
    reward_config: RewardConfig | None = None,
    featurizer: Featurizer | None = None,
    seed: int | None = None,
) -> EnvFactory:
    """Build a zero-arg factory that returns an ActionMasker-wrapped env.

    The factory is what Stable-Baselines3's ``DummyVecEnv`` expects. The
    returned env exposes ``action_masks()`` (via the ``ActionMasker`` wrapper),
    which ``MaskablePPO`` uses to restrict the policy to fittable machines.
    """

    def _factory() -> gym.Env:
        env = ClusterSchedulingEnv(
            sim_config=sim_config,
            workload_config=workload_config,
            reward_config=reward_config,
            featurizer=featurizer,
            seed=seed,
        )
        return ActionMasker(env, _action_mask_fn)

    return _factory


def train_maskable_ppo(
    env_factory: EnvFactory,
    total_timesteps: int,
    seed: int = 0,
    log_dir: str | Path | None = None,
    save_path: str | Path | None = None,
    checkpoint_freq: int = 0,
    ppo_kwargs: dict[str, Any] | None = None,
) -> MaskablePPO:
    """Train a MaskablePPO policy on the given env factory.

    Args:
        env_factory: Zero-arg callable returning an ActionMasker-wrapped env.
            Usually produced by :func:`make_env`.
        total_timesteps: Number of env steps to train for.
        seed: Seed for PPO and the env.
        log_dir: Directory for TensorBoard logs and Monitor CSVs. ``None``
            disables logging.
        save_path: Path (including ``.zip`` suffix) to save the final model.
            ``None`` skips the final save.
        checkpoint_freq: If > 0 and ``save_path`` is set, write periodic
            checkpoints every ``checkpoint_freq`` steps next to ``save_path``.
        ppo_kwargs: Extra kwargs forwarded to ``MaskablePPO(...)``.

    Returns:
        The trained ``MaskablePPO`` instance.
    """
    log_path: Path | None = Path(log_dir) if log_dir else None
    if log_path is not None:
        log_path.mkdir(parents=True, exist_ok=True)

    def _monitored_factory() -> gym.Env:
        env = env_factory()
        if log_path is not None:
            env = Monitor(env, filename=str(log_path / f"monitor_seed{seed}.csv"))
        return env

    vec_env = DummyVecEnv([_monitored_factory])

    kwargs: dict[str, Any] = dict(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        tensorboard_log=str(log_path) if (log_path and _HAS_TENSORBOARD) else None,
        verbose=0,
    )
    if ppo_kwargs:
        kwargs.update(ppo_kwargs)

    model = MaskablePPO(**kwargs)

    callbacks: list[Any] = []
    if checkpoint_freq > 0 and save_path is not None:
        ckpt_dir = Path(save_path).parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(ckpt_dir),
                name_prefix=Path(save_path).stem,
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=callbacks or None)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))

    return model


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MaskablePPO on the cluster scheduling env.")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-machines", type=int, default=10)
    p.add_argument("--cpu-per-machine", type=float, default=8.0)
    p.add_argument("--memory-per-machine", type=float, default=16.0)
    p.add_argument("--num-jobs", type=int, default=100)
    p.add_argument("--arrival-rate", type=float, default=2.0)
    p.add_argument("--log-dir", type=str, default="runs/ppo_run")
    p.add_argument("--save-path", type=str, default="artifacts/ppo.zip")
    p.add_argument("--checkpoint-freq", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    factory = make_env(
        sim_config=SimulatorConfig(
            num_machines=args.num_machines,
            cpu_per_machine=args.cpu_per_machine,
            memory_per_machine=args.memory_per_machine,
        ),
        workload_config=WorkloadConfig(
            num_jobs=args.num_jobs,
            arrival_rate=args.arrival_rate,
        ),
        seed=args.seed,
    )
    train_maskable_ppo(
        factory,
        total_timesteps=args.timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
        save_path=args.save_path,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()
