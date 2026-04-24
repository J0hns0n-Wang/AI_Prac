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
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
try:
    from stable_baselines3.common.utils import LinearSchedule as _LinearSchedule
    def _linear_schedule(start: float) -> Any:
        return _LinearSchedule(start=start, end=0.0, end_fraction=1.0)
except ImportError:  # older SB3
    from stable_baselines3.common.utils import get_linear_fn as _get_linear_fn
    def _linear_schedule(start: float) -> Any:
        return _get_linear_fn(start=start, end=0.0, end_fraction=1.0)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from cluster_scheduler.env import ClusterSchedulingEnv, RewardConfig
from cluster_scheduler.featurizers import (
    BasicFeaturizer,
    Featurizer,
    RichFeaturizer,
    SetFeaturizer,
)
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


_ACTIVATION_BY_NAME: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class HeartbeatCallback(BaseCallback):
    """Prints a one-line progress heartbeat to stdout at regular intervals.

    Unlike SB3's ``progress_bar`` or ``verbose`` output, this always flushes
    stdout, so it survives Python's block-buffering under non-TTY stdout
    (Colab's ``!command``, Jupyter cells, piped logs, etc.). That's the
    difference between seeing ``[heartbeat] 500000/8000000 ...`` every few
    seconds and staring at a blank cell for an hour wondering if the run
    is actually progressing.
    """

    def __init__(self, total_timesteps: int, every: int = 0):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        # If every==0 pick ~50 heartbeats over the run, minimum 1000 steps.
        self.every = (
            int(every)
            if every > 0
            else max(self.total_timesteps // 50, 1000)
        )
        self._t0: float = 0.0
        self._last_report: int = 0

    def _on_training_start(self) -> None:  # type: ignore[override]
        self._t0 = time.time()
        self._last_report = 0
        print(
            f"[heartbeat] starting run of {self.total_timesteps:,} steps "
            f"(report every {self.every:,})",
            flush=True,
        )

    def _on_step(self) -> bool:  # type: ignore[override]
        # self.num_timesteps already aggregates across all vec-env workers.
        if self.num_timesteps - self._last_report >= self.every:
            self._last_report = self.num_timesteps
            elapsed = time.time() - self._t0
            pct = 100.0 * self.num_timesteps / max(self.total_timesteps, 1)
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
            eta_s = (
                (self.total_timesteps - self.num_timesteps) / fps
                if fps > 0 else float("inf")
            )
            print(
                f"[heartbeat] {self.num_timesteps:,}/{self.total_timesteps:,} "
                f"({pct:5.1f}%) elapsed={elapsed:6.0f}s "
                f"fps={fps:6.0f} eta={eta_s:6.0f}s",
                flush=True,
            )
        return True


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
    normalize_reward: bool = False,
    reward_clip: float = 10.0,
    resume_from: str | Path | None = None,
    progress_bar: bool = False,
    verbose: int = 0,
    heartbeat_every: int = 0,
    policy_class: Any = None,
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

    if normalize_reward:
        # Obs is already in [0, 1] via the featurizer, so only reward
        # normalization is useful here. It keeps the critic's target scale
        # stable across workloads, which matters when different regimes
        # have very different reward magnitudes.
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=reward_clip,
            gamma=(ppo_kwargs or {}).get("gamma", 0.99),
        )

    kwargs: dict[str, Any] = dict(
        policy=policy_class if policy_class is not None else "MlpPolicy",
        env=vec_env,
        seed=seed,
        tensorboard_log=str(log_path) if (log_path and _HAS_TENSORBOARD) else None,
        verbose=verbose,
        device=device,
    )
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs
    if ppo_kwargs:
        kwargs.update(ppo_kwargs)

    if resume_from is not None:
        # Warm-start from a prior checkpoint. SB3 lets .load() override the
        # env plus select kwargs (learning rate, clip range, ent coef, etc.),
        # so the user can fine-tune a saved policy on a harder regime or a
        # slower learning schedule.
        # If the prior run saved VecNormalize stats (same path + .vecnormalize.pkl),
        # reuse them so the reward normalizer continues from the trained scale
        # instead of resetting its running mean.
        if isinstance(vec_env, VecNormalize):
            resume_path = Path(resume_from)
            stats_path = resume_path.with_suffix(resume_path.suffix + ".vecnormalize.pkl")
            if stats_path.exists():
                vec_env = VecNormalize.load(str(stats_path), vec_env.venv)
                vec_env.norm_reward = True  # keep reward norm on during continued training

        override = {k: v for k, v in kwargs.items() if k not in ("policy", "env", "seed")}
        model = MaskablePPO.load(
            str(resume_from),
            env=vec_env,
            device=device,
            custom_objects=override,
        )
        model.set_env(vec_env)
    else:
        model = MaskablePPO(**kwargs)

    callbacks: list[Any] = []
    if heartbeat_every >= 0:
        callbacks.append(HeartbeatCallback(total_timesteps, every=heartbeat_every))
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

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks or None,
            progress_bar=progress_bar,
        )
    except ImportError as exc:
        # SB3's progress_bar=True requires tqdm + rich. Fall back silently.
        if progress_bar and ("tqdm" in str(exc) or "rich" in str(exc)):
            print(
                "[train] --progress requested but tqdm/rich not installed; "
                "falling back to heartbeat only. Fix with: pip install tqdm rich",
                flush=True,
            )
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks or None,
                progress_bar=False,
            )
        else:
            raise

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        # Persist VecNormalize stats next to the model so --resume-from can
        # restore the reward normalizer instead of restarting its running mean.
        if isinstance(vec_env, VecNormalize):
            stats_path = out.with_suffix(out.suffix + ".vecnormalize.pkl")
            vec_env.save(str(stats_path))

    return model


def describe_featurizer(featurizer: Featurizer) -> dict[str, Any]:
    """JSON-serializable description of a featurizer for sidecar metadata."""
    if isinstance(featurizer, RichFeaturizer):
        return {"name": "rich", "top_k": featurizer.top_k, "duration_ref": featurizer.duration_ref}
    if isinstance(featurizer, SetFeaturizer):
        return {"name": "set", "duration_ref": featurizer.duration_ref}
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
    if name == "set":
        return SetFeaturizer(duration_ref=float(spec.get("duration_ref", 100.0)))
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


def _json_safe(obj: Any) -> Any:
    """Replace callables / torch classes with a descriptive string."""
    if callable(obj):
        return f"<callable:{getattr(obj, '__name__', repr(obj))}>"
    return str(obj)


def _describe_ppo_kwargs(ppo_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """JSON-safe ppo_kwargs (callables such as lr schedules stringified)."""
    if not ppo_kwargs:
        return {}
    out: dict[str, Any] = {}
    for k, v in ppo_kwargs.items():
        if callable(v):
            out[k] = f"<callable:{getattr(v, '__name__', repr(v))}>"
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
    extra: dict[str, Any] | None = None,
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
        "ppo_kwargs": _describe_ppo_kwargs(ppo_kwargs),
        "n_envs": int(n_envs),
        "device": device,
        "policy_kwargs": _describe_policy_kwargs(policy_kwargs),
    }
    if extra:
        payload.update(extra)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe))
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
    p.add_argument("--featurizer", choices=["basic", "rich", "set"], default="basic",
                   help="'set' is auto-selected and required when --policy set_attention.")
    p.add_argument("--rich-top-k", type=int, default=4,
                   help="top-k queued jobs exposed by RichFeaturizer (ignored for basic/set).")
    # Policy architecture
    p.add_argument("--policy", choices=["mlp", "set_attention"], default="mlp",
                   help="'mlp' is SB3's default MlpPolicy with --policy-hidden/--activation. "
                        "'set_attention' uses SetAttentionExtractor + per-machine scoring head "
                        "and forces --featurizer set (for cross-cluster generalization).")
    # Reward shaping
    p.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    p.add_argument("--wait-penalty-weight", type=float, default=1.0)
    p.add_argument("--backlog-penalty-weight", type=float, default=0.0)
    p.add_argument("--completion-bonus", type=float, default=0.0)
    # PPO hyperparams (unset = SB3 default)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant",
                   help="'linear' decays learning_rate to 0 over training. Requires --learning-rate.")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--clip-range", type=float, default=None)
    p.add_argument("--clip-range-vf", type=float, default=None,
                   help="Clip range for the value function (stabilizes critic). SB3 default is None (unclipped).")
    # Stability / continuation
    p.add_argument("--normalize-reward", action="store_true",
                   help="Wrap the vec env in VecNormalize(norm_reward=True). "
                        "Stabilizes critic learning across regimes with very different reward magnitudes.")
    p.add_argument("--reward-clip", type=float, default=10.0,
                   help="VecNormalize reward clip magnitude (only applies with --normalize-reward).")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to an existing .zip to warm-start from. Hyperparameter flags on this "
                        "invocation override the checkpoint's originals (fine-tuning).")
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
    # IO + progress
    p.add_argument("--log-dir", type=str, default="runs/ppo_run")
    p.add_argument("--save-path", type=str, default="artifacts/ppo.zip")
    p.add_argument("--checkpoint-freq", type=int, default=0)
    p.add_argument("--progress", action="store_true",
                   help="Show a tqdm/rich progress bar during training "
                        "(requires 'tqdm' and 'rich' to be installed; both come "
                        "with stable-baselines3 extras).")
    p.add_argument("--verbose", type=int, default=0,
                   help="SB3 verbosity (0 silent, 1 info, 2 debug). With --progress, "
                        "set 1 to also see per-update stats like fps and ep_rew_mean.")
    p.add_argument("--heartbeat-every", type=int, default=0,
                   help="Print a stdout heartbeat every N env steps. 0 = auto "
                        "(~50 heartbeats over the run). Set -1 to disable.")
    return p


def _ppo_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("learning_rate", "n_steps", "batch_size", "n_epochs",
            "ent_coef", "gamma", "gae_lambda", "clip_range", "clip_range_vf")
    kwargs = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
    # Linear LR decay: SB3 accepts a callable(progress_remaining) → lr.
    if args.lr_schedule == "linear" and args.learning_rate is not None:
        kwargs["learning_rate"] = _linear_schedule(args.learning_rate)
    return kwargs


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    # Early banner (flushed) so the user sees *something* within the first
    # second of the training cell, even before torch / sb3 heavy imports
    # finish inside subprocess workers.
    print(
        f"[train] starting: timesteps={args.timesteps:,} "
        f"n_envs={args.n_envs} device={args.device} "
        f"featurizer={args.featurizer} num_machines={args.num_machines} "
        f"arrival_rate={args.arrival_rate} save_path={args.save_path}",
        flush=True,
    )

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
    # --policy set_attention forces --featurizer set; --policy-hidden and
    # --activation don't apply (the set-attention policy has its own heads).
    if args.policy == "set_attention":
        if args.featurizer != "set":
            if args.featurizer != "basic":
                print(
                    f"[train] --policy set_attention requires --featurizer set "
                    f"(got {args.featurizer!r}); overriding.",
                    flush=True,
                )
            args.featurizer = "set"

    if args.featurizer == "rich":
        featurizer: Featurizer = RichFeaturizer(top_k=args.rich_top_k)
    elif args.featurizer == "set":
        featurizer = SetFeaturizer()
    else:
        featurizer = BasicFeaturizer()

    ppo_kwargs = _ppo_kwargs_from_args(args)

    policy_kwargs: dict[str, Any] = {}
    policy_class: Any = None
    if args.policy == "set_attention":
        # Import lazily so users without torch >= 2.0 / recent sb3-contrib
        # still get the default mlp path.
        from cluster_scheduler.policies import SetMaskablePolicy
        policy_class = SetMaskablePolicy
        if args.policy_hidden or args.activation:
            print(
                "[train] --policy-hidden / --activation are ignored when "
                "--policy set_attention (custom heads).",
                flush=True,
            )
    else:
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
        normalize_reward=args.normalize_reward,
        reward_clip=args.reward_clip,
        resume_from=args.resume_from,
        progress_bar=args.progress,
        verbose=args.verbose,
        heartbeat_every=args.heartbeat_every,
        policy_class=policy_class,
    )
    print(f"[train] done. model saved to {args.save_path}", flush=True)

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
            extra={
                "normalize_reward": bool(args.normalize_reward),
                "reward_clip": float(args.reward_clip),
                "lr_schedule": args.lr_schedule,
                "resume_from": args.resume_from,
                "policy": args.policy,
            },
        )


if __name__ == "__main__":
    main()
