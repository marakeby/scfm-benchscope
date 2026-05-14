
"""
AUPRC plotting utilities for long-format CSVs (with a "Metrics" column).

Usage example:
    from auprc_plots import summarize_by_experiment, plot_grouped_by_experiment, plot_single_method
    files = {"vote": "vote.csv", "avg": "avg.csv", "mil": "mil.csv"}
    summary = summarize_by_experiment(files)
    plot_grouped_by_experiment(summary, out_path="auprc_by_experiment.png")
    plot_single_method(summary, "vote", out_path="auprc_by_experiment_vote.png")
    plot_single_method(summary, "avg",  out_path="auprc_by_experiment_avg.png")
    plot_single_method(summary, "mil",  out_path="auprc_by_experiment_mil.png")
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Internal helpers
# ----------------------------
def _find_metric_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["Metrics", "Metric", "metrics", "metric"]:
        if cand in df.columns:
            return cand
    return None


def _detect_value_col_for_metric(df: pd.DataFrame, metric_col: str, metric_name: str) -> Optional[str]:
    # filter to target metric (case-insensitive exact match)
    m = df[metric_col].astype(str).str.lower() == metric_name.lower()
    sub = df.loc[m].copy()
    if sub.empty:
        return None

    # choose the first numeric-like column that is not metadata
    exclude = {metric_col, "fold", "experiment", "Unnamed: 0", "seed", "run", "split"}
    candidate_cols = [c for c in sub.columns if c not in exclude]
    for c in candidate_cols:
        vals = pd.to_numeric(sub[c], errors="coerce")
        if vals.notna().any():
            return c
    return None


# ----------------------------
# Public API
# ----------------------------
def summarize_by_experiment(
    file_paths: Dict[str, str],
    metric_name: str = "AUPRC",
    experiment_col: str = "experiment",
) -> pd.DataFrame:
    """
    Load each CSV (long format, one row per metric) and aggregate the target metric
    by `experiment` for each method (file key).

    Args:
        file_paths: mapping like {"vote": "/path/vote.csv", "avg": "/path/avg.csv", "mil": "/path/mil.csv"}
        metric_name: which metric to extract (default: "AUPRC")
        experiment_col: column that defines the experiment name (default: "experiment")

    Returns:
        DataFrame with columns: [experiment, method, mean, std, n, note]
    """
    rows = []
    for method, path in file_paths.items():
        if not os.path.exists(path):
            rows.append({
                "experiment": np.nan, "method": method, "mean": np.nan, "std": np.nan, "n": 0,
                "note": f"File not found: {path}",
            })
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            rows.append({
                "experiment": np.nan, "method": method, "mean": np.nan, "std": np.nan, "n": 0,
                "note": f"Read error: {e}",
            })
            continue

        metric_col = _find_metric_col(df)
        if metric_col is None:
            rows.append({
                "experiment": np.nan, "method": method, "mean": np.nan, "std": np.nan, "n": 0,
                "note": f"No 'Metrics' column found. Columns: {list(df.columns)}",
            })
            continue

        mask = df[metric_col].astype(str).str.lower() == metric_name.lower()
        df_metric = df.loc[mask].copy()
        if df_metric.empty:
            rows.append({
                "experiment": np.nan, "method": method, "mean": np.nan, "std": np.nan, "n": 0,
                "note": f"No rows found for metric='{metric_name}'",
            })
            continue

        value_col = _detect_value_col_for_metric(df, metric_col, metric_name)
        if value_col is None:
            rows.append({
                "experiment": np.nan, "method": method, "mean": np.nan, "std": np.nan, "n": 0,
                "note": f"Could not detect numeric value column for metric='{metric_name}'",
            })
            continue

        df_metric[value_col] = pd.to_numeric(df_metric[value_col], errors="coerce")
        grouped = df_metric.groupby(experiment_col, dropna=False)[value_col].agg(["mean", "std", "count"]).reset_index()

        for _, r in grouped.iterrows():
            count = int(r["count"])
            std = float(r["std"]) if count > 1 else (0.0 if count == 1 else np.nan)
            rows.append({
                "experiment": r[experiment_col],
                "method": method,
                "mean": float(r["mean"]),
                "std": std,
                "n": count,
                "note": f"value_col='{value_col}'",
            })

    out = pd.DataFrame(rows).sort_values(["experiment", "method"]).reset_index(drop=True)
    return out


def plot_grouped_by_experiment(
    summary: pd.DataFrame,
    methods_order: Optional[Iterable[str]] = None,
    experiments_order: Optional[Iterable[str]] = None,
    out_path: Optional[str] = None,
    title: str = "AUPRC (mean ± std) by experiment and method",
) -> None:
    """
    Make a grouped bar chart: x-axis experiments; bars per group = methods; error bars = std.
    Each chart is a single matplotlib figure (no subplots).

    Args:
        summary: DataFrame from `summarize_by_experiment`.
        methods_order: iterable of method names to control legend/order (default: sorted unique).
        experiments_order: iterable of experiments to control x-axis order (default: appearance order).
        out_path: if given, saves the figure to this path.
        title: chart title.
    """
    methods = list(methods_order) if methods_order is not None else list(summary["method"].unique())

    if experiments_order is not None:
        experiments = list(experiments_order)
    else:
        # preserve appearance order while dropping NaNs
        experiments = [e for e in summary["experiment"].dropna().unique()]

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(experiments))
    width = 0.8 / max(1, len(methods))

    plt.figure(figsize=(max(7, 1.4 * len(experiments)), 5))

    for i, method in enumerate(methods):
        sub = summary[summary["method"] == method]
        means, stds = [], []
        for exp in experiments:
            row = sub[sub["experiment"] == exp]
            if not row.empty and not np.isnan(row["mean"].values[0]):
                means.append(row["mean"].values[0])
                stds.append(row["std"].values[0] if not np.isnan(row["std"].values[0]) else 0.0)
            else:
                means.append(np.nan)
                stds.append(np.nan)
        positions = x + (i - (len(methods)-1)/2) * width
        plt.bar(positions, means, yerr=stds, width=width, capsize=6, label=method)

    plt.xticks(x, experiments, rotation=0)
    plt.ylabel("AUPRC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_single_method(
    summary: pd.DataFrame,
    method: str,
    out_path: Optional[str] = None,
    sort_experiments_by_mean_desc: bool = True,
    title_prefix: str = "AUPRC (mean ± std) by experiment — ",
) -> None:
    """
    Make a single-method bar chart (one figure) with error bars.

    Args:
        summary: DataFrame from `summarize_by_experiment`.
        method: which method to plot (e.g., "vote", "avg", "mil").
        out_path: if given, saves the figure to this path.
        sort_experiments_by_mean_desc: if True, sort experiments by mean descending.
        title_prefix: prefix for the chart title.
    """
    sub = summary[summary["method"] == method].copy()
    sub = sub.dropna(subset=["mean"])

    if sort_experiments_by_mean_desc:
        sub = sub.sort_values("mean", ascending=False)

    exps = sub["experiment"].astype(str).tolist()
    means = sub["mean"].values
    stds = sub["std"].fillna(0.0).values

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(exps))
    plt.figure(figsize=(max(7, 0.7 * len(exps)), 5))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, exps, rotation=45, ha="right")
    plt.ylabel("AUPRC")
    plt.title(f"{title_prefix}{method}")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
