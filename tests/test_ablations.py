"""Tests for the reward × featurizer ablation runner."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the scripts/ directory importable for the test.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_ablations  # noqa: E402

from cluster_scheduler.env import RewardConfig  # noqa: E402
from cluster_scheduler.evaluation import RegimeSpec  # noqa: E402
from cluster_scheduler.featurizers import BasicFeaturizer, RichFeaturizer  # noqa: E402
from cluster_scheduler.simulator import SimulatorConfig  # noqa: E402
from cluster_scheduler.workload import WorkloadConfig  # noqa: E402


def _tiny_regime(name="r1", num_jobs=10, num_machines=3):
    return RegimeSpec(
        name=name,
        workload_config=WorkloadConfig(
            num_jobs=num_jobs, arrival_rate=2.0,
            cpu_range=(1, 3), memory_range=(1, 4), duration_range=(2, 6),
        ),
        sim_config=SimulatorConfig(num_machines=num_machines, cpu_per_machine=8, memory_per_machine=16),
    )


class TestRunAblationsDryRun:
    def test_grid_shape(self):
        reward_variants = {
            "sparse": RewardConfig(mode="sparse"),
            "dense": RewardConfig(mode="dense"),
        }
        featurizer_variants = {
            "basic": lambda: BasicFeaturizer(),
            "rich": lambda: RichFeaturizer(top_k=2),
        }
        train_seeds = [0, 1]
        eval_seeds = [100, 101]
        regimes = [_tiny_regime("r1"), _tiny_regime("r2", num_machines=4)]

        df = run_ablations.run_ablations(
            reward_variants=reward_variants,
            featurizer_variants=featurizer_variants,
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            regimes=regimes,
            dry_run=True,
        )
        # rewards × featurizers × train_seeds × regimes × eval_seeds
        expected = 2 * 2 * 2 * 2 * 2
        assert len(df) == expected

    def test_columns_include_grid_labels_and_metrics(self):
        df = run_ablations.run_ablations(
            reward_variants={"dense": RewardConfig(mode="dense")},
            featurizer_variants={"basic": lambda: BasicFeaturizer()},
            train_seeds=[0],
            eval_seeds=[100],
            regimes=[_tiny_regime()],
            dry_run=True,
        )
        for c in ("reward", "featurizer", "train_seed", "scheduler", "regime", "seed"):
            assert c in df.columns
        # At least a couple of SchedulerMetrics fields present.
        assert "avg_waiting_time" in df.columns
        assert "utilization" in df.columns

    def test_reward_and_featurizer_labels_are_exhaustive(self):
        reward_variants = {
            "sparse": RewardConfig(mode="sparse"),
            "dense_backlog": RewardConfig(mode="dense", backlog_penalty_weight=0.1),
        }
        featurizer_variants = {
            "basic": lambda: BasicFeaturizer(),
            "rich": lambda: RichFeaturizer(top_k=2),
        }
        df = run_ablations.run_ablations(
            reward_variants=reward_variants,
            featurizer_variants=featurizer_variants,
            train_seeds=[0],
            eval_seeds=[100],
            regimes=[_tiny_regime()],
            dry_run=True,
        )
        assert set(df["reward"].unique()) == set(reward_variants.keys())
        assert set(df["featurizer"].unique()) == set(featurizer_variants.keys())


class TestWriteArtifacts:
    def test_writes_csv_and_heatmaps(self, tmp_path: Path):
        df = run_ablations.run_ablations(
            reward_variants={
                "sparse": RewardConfig(mode="sparse"),
                "dense": RewardConfig(mode="dense"),
            },
            featurizer_variants={
                "basic": lambda: BasicFeaturizer(),
                "rich": lambda: RichFeaturizer(top_k=2),
            },
            train_seeds=[0],
            eval_seeds=[100],
            regimes=[_tiny_regime()],
            dry_run=True,
        )
        run_ablations.write_artifacts(df, tmp_path)
        assert (tmp_path / "ablations_raw.csv").exists()
        for metric in run_ablations.HEATMAP_METRICS:
            assert (tmp_path / f"heatmap_{metric}.png").exists()


class TestPlotAblationHeatmap:
    def test_renders_with_synthetic_df(self, tmp_path: Path):
        from cluster_scheduler.report import plot_ablation_heatmap

        rng = np.random.default_rng(0)
        rows = []
        for r in ("sparse", "dense", "dense_backlog"):
            for f in ("basic", "rich"):
                for seed in range(3):
                    rows.append({
                        "reward": r, "featurizer": f, "seed": seed,
                        "avg_waiting_time": float(rng.uniform(1.0, 5.0)),
                    })
        df = pd.DataFrame(rows)
        out = tmp_path / "heatmap.png"
        returned = plot_ablation_heatmap(df, metric="avg_waiting_time", out_path=out)
        assert returned == out
        assert out.exists() and out.stat().st_size > 0

    def test_missing_columns_raises(self, tmp_path: Path):
        from cluster_scheduler.report import plot_ablation_heatmap

        df = pd.DataFrame({"reward": ["dense"], "featurizer": ["basic"]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_ablation_heatmap(df, metric="avg_waiting_time", out_path=tmp_path / "x.png")
