#!/usr/bin/env python3
"""
Repeated subsampled embedding evaluation (bootstrap-style) under an output root.

Wrapper around ``scfm-eval report --bootstrap``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scfm_cancer_eval.reporting import create_bootstrap_metrics_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "embedding_bootstrap"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(
        os.environ.get("SCFM_CALC_METRICS_ROOT")
        or os.environ.get("SCFM_OUTPUT_PATH")
        or "output"
    )
    parser.add_argument("root", nargs="?", type=Path, default=default_root)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help=f"Aggregate output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument("--folder-substring", default="cell_type")
    parser.add_argument("--experiment-marker", default="brca_cell_type")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-merge-results", action="store_true")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        alt = REPO_ROOT / "output" / "exp"
        if alt.exists():
            print(f"Using {alt} (default output root not found).", file=sys.stderr)
            root = alt

    try:
        bundle = create_bootstrap_metrics_bundle(
            root,
            args.output,
            folder_substring=args.folder_substring,
            experiment_marker=args.experiment_marker,
            sample_size=args.sample_size,
            n_runs=args.n_runs,
            base_seed=args.base_seed,
            n_jobs=args.n_jobs,
            merge_into_results_json=not args.no_merge_results,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1 if isinstance(exc, ValueError) else 2

    print(f"Wrote aggregate CSVs under {args.output}")
    print(f"Bootstrap CSV: {bundle.bootstrap_csv}")
    if bundle.failed_csv is not None:
        print(f"Failed models: {bundle.failed_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
