#!/usr/bin/env python3
"""
Collect embedding evaluation CSVs into one table.

Only includes ``embedding_metrics.csv`` (exact name — no bootstrap suffixes
like ``embedding_metrics_<id>.csv``). Only files under a directory tree where
at least one path component contains the substring ``cell_type`` (e.g.
``brca_cell_type_*`` run folders).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from analysis_utils import EXPERIMENT_NAME_MAP, MODEL_NAME_MAP, map_groups
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "embedding.metrics.csv"

def path_has_folder_substring(path: Path, needle: str) -> bool:
    nl = needle.lower()
    return any(nl in part.lower() for part in path.parts)


def run_dir_from_metrics(csv_path: Path) -> Path:
    """Absolute filesystem path to the experiment run directory."""
    return csv_path.parent.resolve()


def find_embedding_metrics_files(root: Path, folder_substring: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    out: list[Path] = []
    for p in root.rglob("embedding_metrics.csv"):
        if not path_has_folder_substring(p, folder_substring):
            continue
        out.append(p)
    return sorted(out)


def load_run_row(csv_path: Path) -> dict[str, object] | None:
    run_summary_path = csv_path.parent / "run_summary.json"
    try:
        with open(run_summary_path, encoding="utf-8") as f:
            run_summary = json.load(f)
        run_id = run_summary["run_id"]
        exp = Path(run_summary["config_path"]).stem
        model = str(run_id).replace(f"_{exp}", "")
    except OSError as e:
        print(f"Cannot load run_summary.json, skipping {csv_path}: {e}", file=sys.stderr)
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Invalid run_summary.json, skipping {csv_path}: {e}", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        print(f"Cannot read {csv_path}: {e}", file=sys.stderr)
        return None

    if df.empty:
        print(f"Empty metrics file, skipping {csv_path}", file=sys.stderr)
        return None

    row: dict[str, object] = {
        "run_id": run_id,
        "model": model,
        "exp": exp,
        "exp_path": str(run_dir_from_metrics(csv_path)),
        "embedding_key": ",".join(str(c) for c in df.columns),
        "metrics_path": str(csv_path.resolve()),
    }

    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        for metric_name, val in series.items():
            row[str(metric_name)] = val
    else:
        for col in df.columns:
            series = df[col]
            for metric_name, val in series.items():
                row[f"{col}::{metric_name}"] = val

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(
        os.environ.get("SCFM_EMBEDDING_OUTPUT_ROOT")
        or os.environ.get("SCFM_CLASSIFICATION_OUTPUT_ROOT")
        or "/home/haitham/mnt/__output_v7/exp"
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=default_root,
        help="Root directory to search (default: $SCFM_EMBEDDING_OUTPUT_ROOT, "
        "then $SCFM_CLASSIFICATION_OUTPUT_ROOT, then mount path)",
    )
    parser.add_argument(
        "--folder-substring",
        default="cell_type",
        help="Require this substring in at least one path component (default: cell_type)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists() and root.resolve() == default_root.resolve():
        alt = REPO_ROOT / "output" / "exp"
        if alt.exists():
            print(f"Using {alt} (default output root not found).", file=sys.stderr)
            root = alt

    try:
        files = find_embedding_metrics_files(root, args.folder_substring)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    for csv_path in files:
        rec = load_run_row(csv_path)
        if rec is not None:
            rows.append(rec)

    if not rows:
        print(
            f"No embedding_metrics.csv files found under {root} "
            f"with folder substring {args.folder_substring!r}.",
            file=sys.stderr,
        )
        return 1

    combined = pd.DataFrame(rows)
    combined["group"] = combined["model"].map(map_groups)
    combined["model_display"] = combined["model"].map(MODEL_NAME_MAP)
    combined["exp_display"] = combined["exp"].map(EXPERIMENT_NAME_MAP)

    meta_cols = [
        "run_id",
        "model",
        "model_display",
        "group",
        "exp",
        "exp_path",
        "exp_display",
        "embedding_key",
        "metrics_path",
    ]
    meta_present = [c for c in meta_cols if c in combined.columns]
    metric_cols = [c for c in combined.columns if c not in meta_present]
    combined = combined[meta_present + sorted(metric_cols, key=str)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
