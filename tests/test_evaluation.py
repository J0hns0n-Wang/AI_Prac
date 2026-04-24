"""Tests for the multi-regime evaluation harness."""

import numpy as np
import pandas as pd
import pytest

from cluster_scheduler.evaluation import (
    METRIC_FIELDS,
    RegimeSpec,
    arrival_rate_regimes,
    default_regimes,
    summarize,
    sweep,
)
from cluster_scheduler.scheduler import FirstFitScheduler, RandomScheduler
from cluster_scheduler.simulator import SimulatorConfig
from cluster_scheduler.workload import WorkloadConfig


def _small_regimes():
    return [
        RegimeSpec(
            name="regime_a",
            workload_config=WorkloadConfig(
                num_jobs=15, arrival_rate=2.0,
                cpu_range=(1, 3), memory_range=(1, 4), duration_range=(2, 6),
            ),
            sim_config=SimulatorConfig(num_machines=3, cpu_per_machine=8, memory_per_machine=16),
        ),
        RegimeSpec(
            name="regime_b",
            workload_config=WorkloadConfig(
                num_jobs=20, arrival_rate=1.5,
                cpu_range=(1, 4), memory_range=(1, 6), duration_range=(2, 10),
            ),
            sim_config=SimulatorConfig(num_machines=4, cpu_per_machine=8, memory_per_machine=16),
        ),
    ]


class TestSweep:
    def test_shape_and_columns(self):
        schedulers = {
            "first_fit": FirstFitScheduler(),
            "random": RandomScheduler(seed=0),
        }
        regimes = _small_regimes()
        seeds = [0, 1]
        df = sweep(schedulers, regimes, seeds)
        # 2 regimes × 2 seeds × 2 schedulers = 8 rows
        assert len(df) == 8
        assert {"scheduler", "regime", "seed"}.issubset(df.columns)
        for m in METRIC_FIELDS:
            assert m in df.columns
        assert set(df["scheduler"].unique()) == {"first_fit", "random"}
        assert set(df["regime"].unique()) == {"regime_a", "regime_b"}

    def test_all_jobs_complete_per_cell(self):
        schedulers = {"first_fit": FirstFitScheduler()}
        df = sweep(schedulers, _small_regimes(), [0, 1])
        # regime_a has num_jobs=15, regime_b has 20
        for _, row in df.iterrows():
            expected = 15 if row["regime"] == "regime_a" else 20
            assert int(row["total_jobs"]) == expected

    def test_deterministic_for_fixed_seeds(self):
        schedulers = {"first_fit": FirstFitScheduler()}
        df1 = sweep(schedulers, _small_regimes(), [0, 1])
        df2 = sweep(schedulers, _small_regimes(), [0, 1])
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True), df2.reset_index(drop=True)
        )

    def test_same_seed_same_jobs_across_schedulers(self):
        """Every scheduler in a (regime, seed) cell must see the same workload."""
        df = sweep(
            {"first_fit": FirstFitScheduler(), "random": RandomScheduler(seed=0)},
            _small_regimes()[:1],
            [7],
        )
        # total_jobs reflects the workload, not the scheduler — both rows must agree.
        assert df["total_jobs"].nunique() == 1


class TestSummarize:
    def test_shape_and_fields(self):
        df = sweep(
            {"first_fit": FirstFitScheduler(), "random": RandomScheduler(seed=0)},
            _small_regimes(),
            [0, 1, 2],
        )
        summary = summarize(df)
        # 2 schedulers × 2 regimes × len(METRIC_FIELDS) metrics
        assert len(summary) == 2 * 2 * len(METRIC_FIELDS)
        assert set(summary.columns) == {
            "scheduler", "regime", "metric", "mean", "std", "ci_lo", "ci_hi", "n",
        }
        assert (summary["n"] == 3).all()

    def test_ci_brackets_mean(self):
        df = sweep(
            {"first_fit": FirstFitScheduler()},
            _small_regimes(),
            [0, 1, 2, 3, 4],
        )
        summary = summarize(df)
        assert (summary["ci_lo"] <= summary["mean"] + 1e-9).all()
        assert (summary["ci_hi"] >= summary["mean"] - 1e-9).all()

    def test_single_seed_ci_collapses_to_mean(self):
        df = sweep({"first_fit": FirstFitScheduler()}, _small_regimes()[:1], [0])
        summary = summarize(df)
        # With one sample, CI bounds equal the mean.
        np.testing.assert_allclose(summary["ci_lo"], summary["mean"])
        np.testing.assert_allclose(summary["ci_hi"], summary["mean"])
        assert (summary["n"] == 1).all()


class TestArrivalRateRegimes:
    def test_names_and_rates(self):
        regimes = arrival_rate_regimes(
            rates=[1.0, 2.5, 8.0],
            sim_config=SimulatorConfig(num_machines=6),
        )
        assert [r.name for r in regimes] == ["rate_1.0", "rate_2.5", "rate_8.0"]
        assert [r.workload_config.arrival_rate for r in regimes] == [1.0, 2.5, 8.0]

    def test_sim_config_is_shared(self):
        sc = SimulatorConfig(num_machines=7, cpu_per_machine=8, memory_per_machine=16)
        regimes = arrival_rate_regimes(rates=[1.0, 2.0], sim_config=sc)
        assert regimes[0].sim_config is sc and regimes[1].sim_config is sc

    def test_workload_knobs_propagate(self):
        base = WorkloadConfig(
            num_jobs=42, arrival_rate=99.0,  # arrival_rate overridden per regime
            cpu_range=(2.0, 5.0), memory_range=(3.0, 7.0),
            duration_range=(4.0, 9.0),
        )
        regimes = arrival_rate_regimes(rates=[1.0, 4.0], base_workload=base)
        for r in regimes:
            assert r.workload_config.num_jobs == 42
            assert r.workload_config.cpu_range == (2.0, 5.0)
            assert r.workload_config.memory_range == (3.0, 7.0)
            assert r.workload_config.duration_range == (4.0, 9.0)


class TestDefaultRegimes:
    def test_names_are_unique_and_nonempty(self):
        regimes = default_regimes()
        names = [r.name for r in regimes]
        assert len(names) == len(set(names))
        assert len(regimes) >= 4  # project needs enough diversity
