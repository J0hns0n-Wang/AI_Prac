# Cluster Scheduler (CS 4701)

Learning generalizable cluster resource scheduling policies via reinforcement
learning. A discrete-event cluster simulator, heuristic baselines (First-Fit,
Best-Fit, Shortest-Job-First), a Gymnasium environment, and a PPO training
pipeline for comparing learned policies against the heuristics.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+.

## Quickstart

Run the heuristic baselines on a generated workload:

```python
from cluster_scheduler import (
    FirstFitScheduler, BestFitScheduler, ShortestJobFirstScheduler,
    SimulatorConfig, WorkloadConfig, run_benchmark,
)

results = run_benchmark(
    schedulers={
        "first_fit": FirstFitScheduler(),
        "best_fit": BestFitScheduler(),
        "sjf": ShortestJobFirstScheduler(),
    },
    workload_config=WorkloadConfig(num_jobs=200, arrival_rate=2.0),
    sim_config=SimulatorConfig(num_machines=10),
    seeds=[0, 1, 2],
)

for r in results:
    print(r.scheduler_name, r.summary())
```

## Tests

```bash
pytest
```

Long-running tests (e.g., PPO smoke training) are marked `slow` and skipped by
default. Run them with `pytest -m slow`.

## Training on a GPU (A100 / Colab)

PPO on this env is env-step-bound on a single process. Use `--n-envs` to run
many parallel simulators via `SubprocVecEnv` so the GPU stays fed, and bump
the policy network size — a `[256, 256, 128]` MLP is microscopic for an A100.

```bash
python -m cluster_scheduler.train \
    --timesteps 8000000 \
    --n-envs 32 --device cuda \
    --policy-hidden 256 256 128 --activation gelu \
    --num-machines 10 --num-jobs 300 --arrival-rate 8.0 \
    --featurizer rich --rich-top-k 4 \
    --reward-mode dense --backlog-penalty-weight 0.05 \
    --n-steps 512 --batch-size 4096 \
    --learning-rate 2.5e-4 --ent-coef 0.01 \
    --gamma 0.995 --gae-lambda 0.95 \
    --save-path artifacts/ppo_m10_big.zip \
    --log-dir runs/ppo_m10_big
```

`n_steps=512 × n_envs=32` = 16384 rollout samples per update;
`batch_size=4096` gives 4 minibatches per epoch. Each command also writes
`<save_path>.meta.json` recording the featurizer, reward config, and
network architecture used, so `scripts/run_experiments.py` reconstructs
the right evaluation setup automatically.
