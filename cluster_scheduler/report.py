"""Plotting and CSV utilities for final-report artifacts.

Consume tidy DataFrames produced by :func:`cluster_scheduler.evaluation.sweep`
(one row per ``(scheduler, regime, seed)``) or equivalent per-parameter sweeps.
All plot functions return the written path so callers can log or assert.

Rendering is forced to the non-interactive ``Agg`` backend so these work in
headless CI and test sandboxes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend set)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cluster_scheduler.metrics import mean_ci  # noqa: E402


def write_results_csv(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Write ``df`` to ``out_path`` as CSV (no index). Creates parent dirs."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def plot_arrival_sweep(
    df: pd.DataFrame,
    metric: str,
    out_path: Path | str,
    x_col: str = "arrival_rate",
    title: str | None = None,
    alpha: float = 0.05,
) -> Path:
    """Line plot of ``metric`` vs ``x_col``, one line per scheduler with 95% CI bands.

    Args:
        df: Tidy DataFrame containing ``scheduler``, ``x_col``, ``metric``.
            Multiple rows per (scheduler, x) value (one per seed) are aggregated.
        metric: Column name to plot on the y-axis.
        out_path: PNG destination.
        x_col: Column to use for the x-axis (default ``"arrival_rate"``).
        title: Optional plot title; defaults to ``"<metric> vs <x_col>"``.
        alpha: Significance level for the CI band.
    """
    _require_columns(df, {"scheduler", x_col, metric})
    fig, ax = plt.subplots(figsize=(7, 4))
    for scheduler, group in df.groupby("scheduler"):
        xs = sorted(group[x_col].unique())
        means: list[float] = []
        los: list[float] = []
        his: list[float] = []
        for x in xs:
            vals = group.loc[group[x_col] == x, metric].to_numpy(dtype=float)
            m, lo, hi = mean_ci(vals, alpha=alpha)
            means.append(m)
            los.append(lo)
            his.append(hi)
        ax.plot(xs, means, marker="o", label=str(scheduler))
        ax.fill_between(xs, los, his, alpha=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {x_col}")
    ax.legend()
    ax.grid(alpha=0.3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_regime_bars(
    df: pd.DataFrame,
    metric: str,
    out_path: Path | str,
    title: str | None = None,
    alpha: float = 0.05,
) -> Path:
    """Grouped bar chart: one bar-group per regime, one bar per scheduler, CI error bars.

    Args:
        df: Tidy DataFrame with columns ``scheduler``, ``regime``, ``metric``,
            one row per seed within each ``(scheduler, regime)`` cell.
        metric: Column name for bar heights.
        out_path: PNG destination.
        title: Optional plot title.
        alpha: CI level for error bars.
    """
    _require_columns(df, {"scheduler", "regime", metric})
    schedulers = sorted(df["scheduler"].unique())
    regimes = sorted(df["regime"].unique())
    if not schedulers or not regimes:
        raise ValueError("df must contain at least one scheduler and one regime")

    width = 0.8 / len(schedulers)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(regimes) + 2), 4.5))

    for i, sched in enumerate(schedulers):
        means: list[float] = []
        lo_err: list[float] = []
        hi_err: list[float] = []
        for regime in regimes:
            vals = df.loc[
                (df["scheduler"] == sched) & (df["regime"] == regime), metric
            ].to_numpy(dtype=float)
            m, lo, hi = mean_ci(vals, alpha=alpha)
            means.append(m)
            lo_err.append(max(m - lo, 0.0))
            hi_err.append(max(hi - m, 0.0))
        xs = np.arange(len(regimes)) + i * width
        ax.bar(
            xs,
            means,
            width=width,
            yerr=[lo_err, hi_err],
            label=str(sched),
            capsize=3,
        )

    ax.set_xticks(np.arange(len(regimes)) + width * (len(schedulers) - 1) / 2)
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by regime")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _require_columns(df: pd.DataFrame, cols: set[str]) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
