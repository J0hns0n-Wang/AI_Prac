"""Tests for plotting and CSV artifact generation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cluster_scheduler.report import (
    plot_arrival_sweep,
    plot_regime_bars,
    write_results_csv,
)


def _synthetic_regime_df(schedulers=("first_fit", "rl"), regimes=("light", "heavy"), seeds=(0, 1, 2)):
    rng = np.random.default_rng(0)
    rows = []
    for s in schedulers:
        for r in regimes:
            for seed in seeds:
                rows.append({
                    "scheduler": s,
                    "regime": r,
                    "seed": seed,
                    "avg_waiting_time": float(rng.uniform(1.0, 5.0)),
                    "utilization": float(rng.uniform(0.5, 0.9)),
                })
    return pd.DataFrame(rows)


def _synthetic_arrival_df(
    schedulers=("first_fit", "rl"), rates=(0.5, 1.0, 2.0, 3.0), seeds=(0, 1, 2)
):
    rng = np.random.default_rng(1)
    rows = []
    for s in schedulers:
        for rate in rates:
            for seed in seeds:
                rows.append({
                    "scheduler": s,
                    "arrival_rate": rate,
                    "seed": seed,
                    "avg_waiting_time": float(rng.uniform(0.5, 4.0) * rate),
                })
    return pd.DataFrame(rows)


class TestWriteResultsCsv:
    def test_writes_csv_and_creates_parent(self, tmp_path: Path):
        df = _synthetic_regime_df()
        out = tmp_path / "nested" / "dir" / "results.csv"
        returned = write_results_csv(df, out)
        assert returned == out
        assert out.exists()
        loaded = pd.read_csv(out)
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)


class TestPlotRegimeBars:
    def test_writes_png(self, tmp_path: Path):
        df = _synthetic_regime_df()
        out = tmp_path / "regime_bars.png"
        returned = plot_regime_bars(df, metric="avg_waiting_time", out_path=out)
        assert returned == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_requires_expected_columns(self, tmp_path: Path):
        df = pd.DataFrame({"scheduler": ["a"], "seed": [0]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_regime_bars(df, metric="avg_waiting_time", out_path=tmp_path / "x.png")

    def test_empty_input_raises(self, tmp_path: Path):
        df = pd.DataFrame({"scheduler": [], "regime": [], "avg_waiting_time": []})
        with pytest.raises(ValueError):
            plot_regime_bars(df, metric="avg_waiting_time", out_path=tmp_path / "x.png")


class TestPlotArrivalSweep:
    def test_writes_png(self, tmp_path: Path):
        df = _synthetic_arrival_df()
        out = tmp_path / "arrival.png"
        returned = plot_arrival_sweep(df, metric="avg_waiting_time", out_path=out)
        assert returned == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_x_col(self, tmp_path: Path):
        df = _synthetic_arrival_df().rename(columns={"arrival_rate": "load"})
        out = tmp_path / "arrival.png"
        plot_arrival_sweep(df, metric="avg_waiting_time", out_path=out, x_col="load")
        assert out.exists()

    def test_missing_columns_raises(self, tmp_path: Path):
        df = pd.DataFrame({"scheduler": ["a"], "arrival_rate": [1.0]})  # missing metric
        with pytest.raises(ValueError, match="missing required columns"):
            plot_arrival_sweep(
                df, metric="avg_waiting_time", out_path=tmp_path / "x.png"
            )
