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
