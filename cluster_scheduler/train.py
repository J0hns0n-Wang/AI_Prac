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
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from cluster_scheduler.env import ClusterSchedulingEnv, RewardConfig
from cluster_scheduler.featurizers import BasicFeaturizer, Featurizer, RichFeaturizer
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


_ACTIVATION_BY_NAME: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


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

    The factory is what Stable-Baselines3's vec-env classes expect. The
    returned env exposes ``action_masks()`` (via the ``ActionMasker`` wrapper),
    which ``MaskablePPO`` uses to restrict the policy to fittable machines.

    For parallel training, call :func:`make_env` multiple times with distinct
    seeds (e.g. ``seed + i``) to give each ``SubprocVecEnv`` worker an
    independent workload.
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
    n_envs: int = 1,
    env_factory_builder: Callable[[int], EnvFactory] | None = None,
    device: str = "auto",
    policy_kwargs: dict[str, Any] | None = None,
) -> MaskablePPO:
    """Train a MaskablePPO policy on the given env factory.

    Args:
        env_factory: Zero-arg callable returning an ActionMasker-wrapped env.
            Used when ``n_envs == 1`` or when ``env_factory_builder`` is None
            (replicated n_envs times under a SubprocVecEnv).
        total_timesteps: Number of env steps to train for.
        seed: Seed for PPO and the base env.
        log_dir: Directory for TensorBoard logs and Monitor CSVs. ``None``
            disables logging.
        save_path: Path (including ``.zip`` suffix) to save the final model.
            ``None`` skips the final save.
        checkpoint_freq: If > 0 and ``save_path`` is set, write periodic
            checkpoints every ``checkpoint_freq`` steps next to ``save_path``.
        ppo_kwargs: Extra kwargs forwarded to ``MaskablePPO(...)``.
        n_envs: Number of parallel environments. ``1`` uses ``DummyVecEnv``
            (single-process). ``>1`` uses ``SubprocVecEnv`` with per-worker
            seeds so each trajectory is independent.
        env_factory_builder: Optional callable ``i -> EnvFactory`` that
            builds the factory for worker ``i``. Used to give each worker a
            distinct seed/workload. When None, ``env_factory`` is reused.
        device: ``"auto" | "cpu" | "cuda"``. Forwarded to MaskablePPO.
        policy_kwargs: Extra kwargs for the policy network, e.g.
            ``{"net_arch": [256, 256, 128], "activation_fn": torch.nn.GELU}``.

    Returns:
        The trained ``MaskablePPO`` instance.
    """
    log_path: Path | None = Path(log_dir) if log_dir else None
    if log_path is not None:
        log_path.mkdir(parents=True, exist_ok=True)

    def _wrap(worker_idx: int, factory: EnvFactory) -> Callable[[], gym.Env]:
        """Return a zero-arg factory that optionally attaches Monitor to worker 0."""
        def _build() -> gym.Env:
            env = factory()
            if log_path is not None and worker_idx == 0:
                env = Monitor(env, filename=str(log_path / f"monitor_seed{seed}.csv"))
            return env
        return _build

    vec_env: Any
    if n_envs <= 1:
        vec_env = DummyVecEnv([_wrap(0, env_factory)])
    else:
        builder = env_factory_builder or (lambda _: env_factory)
        factories = [_wrap(i, builder(i)) for i in range(n_envs)]
        # spawn is safe cross-platform (macOS, Colab Linux). fork is faster
        # on Linux but chokes on re-imports; spawn is the robust default.
        vec_env = SubprocVecEnv(factories, start_method="spawn")

    kwargs: dict[str, Any] = dict(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        tensorboard_log=str(log_path) if (log_path and _HAS_TENSORBOARD) else None,
        verbose=0,
        device=device,
    )
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs
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


def describe_featurizer(featurizer: Featurizer) -> dict[str, Any]:
    """JSON-serializable description of a featurizer for sidecar metadata."""
    if isinstance(featurizer, RichFeaturizer):
        return {"name": "rich", "top_k": featurizer.top_k, "duration_ref": featurizer.duration_ref}
    if isinstance(featurizer, BasicFeaturizer):
        return {"name": "basic", "duration_ref": featurizer.duration_ref}
    return {"name": featurizer.__class__.__name__}


def build_featurizer(spec: dict[str, Any]) -> Featurizer:
    """Inverse of :func:`describe_featurizer`. Accepts a dict from a sidecar."""
    name = spec.get("name", "basic")
    if name == "rich":
        return RichFeaturizer(
            top_k=int(spec.get("top_k", 4)),
            duration_ref=float(spec.get("duration_ref", 100.0)),
        )
    if name == "basic":
        return BasicFeaturizer(duration_ref=float(spec.get("duration_ref", 100.0)))
    raise ValueError(f"Unknown featurizer name: {name!r}")


def _describe_policy_kwargs(policy_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """JSON-safe serialization of policy_kwargs (torch activation → name)."""
    if not policy_kwargs:
        return {}
    out: dict[str, Any] = {}
    for k, v in policy_kwargs.items():
        if k == "activation_fn" and isinstance(v, type) and issubclass(v, nn.Module):
            out[k] = v.__name__.lower()
        elif k == "net_arch":
            out[k] = list(v)
        else:
            out[k] = v
    return out


def write_model_sidecar(
    save_path: str | Path,
    *,
    sim_config: SimulatorConfig,
    workload_config: WorkloadConfig,
    reward_config: RewardConfig,
    featurizer: Featurizer,
    timesteps: int,
    seed: int,
    ppo_kwargs: dict[str, Any] | None = None,
    n_envs: int = 1,
    device: str = "auto",
    policy_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Write a small JSON describing how ``save_path`` was trained.

    Downstream tools (e.g. ``scripts/run_experiments.py``) read this to
    reconstruct the exact featurizer and queue normalization used at
    training time, so eval observations match training observations.
    """
    out = Path(save_path)
    sidecar = out.with_suffix(out.suffix + ".meta.json") if out.suffix else out.with_suffix(".meta.json")
    payload = {
        "featurizer": describe_featurizer(featurizer),
        "num_machines": sim_config.num_machines,
        "cpu_per_machine": sim_config.cpu_per_machine,
        "memory_per_machine": sim_config.memory_per_machine,
        "num_jobs": workload_config.num_jobs,
        "arrival_rate": workload_config.arrival_rate,
        "reward_config": asdict(reward_config),
        "timesteps": int(timesteps),
        "seed": int(seed),
        "ppo_kwargs": ppo_kwargs or {},
        "n_envs": int(n_envs),
        "device": device,
        "policy_kwargs": _describe_policy_kwargs(policy_kwargs),
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return sidecar


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MaskablePPO on the cluster scheduling env.")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    # Cluster shape
    p.add_argument("--num-machines", type=int, default=10)
    p.add_argument("--cpu-per-machine", type=float, default=8.0)
    p.add_argument("--memory-per-machine", type=float, default=16.0)
    # Workload
    p.add_argument("--num-jobs", type=int, default=100)
    p.add_argument("--arrival-rate", type=float, default=2.0)
    # Featurizer
    p.add_argument("--featurizer", choices=["basic", "rich"], default="basic")
    p.add_argument("--rich-top-k", type=int, default=4,
                   help="top-k queued jobs exposed by RichFeaturizer (ignored for basic).")
    # Reward shaping
    p.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    p.add_argument("--wait-penalty-weight", type=float, default=1.0)
    p.add_argument("--backlog-penalty-weight", type=float, default=0.0)
    p.add_argument("--completion-bonus", type=float, default=0.0)
    # PPO hyperparams (unset = SB3 default)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--clip-range", type=float, default=None)
    # Compute / parallelism
    p.add_argument("--n-envs", type=int, default=1,
                   help="Parallel environments via SubprocVecEnv. 1 keeps the "
                        "single-process DummyVecEnv path (unchanged default).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Device for the policy network. 'auto' picks CUDA when available.")
    # Policy architecture
    p.add_argument("--policy-hidden", type=int, nargs="+", default=None,
                   help="MLP hidden layer sizes, e.g. --policy-hidden 256 256 128. "
                        "Omit for SB3 default ([64, 64]).")
    p.add_argument("--activation", choices=sorted(_ACTIVATION_BY_NAME), default=None,
                   help="Activation for the policy MLP. Omit for SB3 default (tanh).")
    # IO
    p.add_argument("--log-dir", type=str, default="runs/ppo_run")
    p.add_argument("--save-path", type=str, default="artifacts/ppo.zip")
    p.add_argument("--checkpoint-freq", type=int, default=0)
    return p


def _ppo_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("learning_rate", "n_steps", "batch_size", "n_epochs",
            "ent_coef", "gamma", "gae_lambda", "clip_range")
    return {k: getattr(args, k) for k in keys if getattr(args, k) is not None}


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    sim_cfg = SimulatorConfig(
        num_machines=args.num_machines,
        cpu_per_machine=args.cpu_per_machine,
        memory_per_machine=args.memory_per_machine,
    )
    workload_cfg = WorkloadConfig(
        num_jobs=args.num_jobs,
        arrival_rate=args.arrival_rate,
    )
    reward_cfg = RewardConfig(
        mode=args.reward_mode,
        wait_penalty_weight=args.wait_penalty_weight,
        backlog_penalty_weight=args.backlog_penalty_weight,
        completion_bonus=args.completion_bonus,
    )
    if args.featurizer == "rich":
        featurizer: Featurizer = RichFeaturizer(top_k=args.rich_top_k)
    else:
        featurizer = BasicFeaturizer()

    ppo_kwargs = _ppo_kwargs_from_args(args)

    policy_kwargs: dict[str, Any] = {}
    if args.policy_hidden:
        policy_kwargs["net_arch"] = list(args.policy_hidden)
    if args.activation:
        policy_kwargs["activation_fn"] = _ACTIVATION_BY_NAME[args.activation]

    # Factory used when n_envs==1; for parallel training we build one
    # factory per worker so each has its own seed.
    single_factory = make_env(
        sim_config=sim_cfg,
        workload_config=workload_cfg,
        reward_config=reward_cfg,
        featurizer=featurizer,
        seed=args.seed,
    )
    def _factory_for(worker_idx: int) -> EnvFactory:
        return make_env(
            sim_config=sim_cfg,
            workload_config=workload_cfg,
            reward_config=reward_cfg,
            featurizer=featurizer,
            seed=args.seed + worker_idx,
        )

    train_maskable_ppo(
        single_factory,
        total_timesteps=args.timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
        save_path=args.save_path,
        checkpoint_freq=args.checkpoint_freq,
        ppo_kwargs=ppo_kwargs or None,
        n_envs=args.n_envs,
        env_factory_builder=_factory_for if args.n_envs > 1 else None,
        device=args.device,
        policy_kwargs=policy_kwargs or None,
    )

    if args.save_path:
        write_model_sidecar(
            args.save_path,
            sim_config=sim_cfg,
            workload_config=workload_cfg,
            reward_config=reward_cfg,
            featurizer=featurizer,
            timesteps=args.timesteps,
            seed=args.seed,
            ppo_kwargs=ppo_kwargs,
            n_envs=args.n_envs,
            device=args.device,
            policy_kwargs=policy_kwargs or None,
        )


if __name__ == "__main__":
    main()
