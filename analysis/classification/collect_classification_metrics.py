#!/usr/bin/env python3
"""Collect classification CV metric CSVs into one table (CLI wrapper)."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from scfm_cancer_eval.reporting.collect_metrics import create_collect_metrics_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "classification.metrics.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(
            os.environ.get(
                "SCFM_CLASSIFICATION_OUTPUT_ROOT",
                os.environ.get("SCFM_OUTPUT_PATH", "output"),
            )
        ),
        help="Root directory to search",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--score-col",
        default="randomforest",
        help="Preferred classifier score column in *cv_metrics.csv",
    )
    parser.add_argument(
        "--keep-luad-cancer-stage",
        action="store_true",
        help="Do not drop rows where exp == luad_cancer_stage",
    )
    parser.add_argument(
        "--include-arxiv",
        action="store_true",
        help="Include runs whose path contains 'arxiv'",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        alt = REPO_ROOT / "output" / "exp"
        if alt.exists():
            print(f"Using {alt} (default output root not found).", file=sys.stderr)
            root = alt

    try:
        bundle = create_collect_metrics_bundle(
            root,
            args.output.parent,
            kind="classification",
            score_col=args.score_col,
            keep_luad_cancer_stage=args.keep_luad_cancer_stage,
            include_arxiv=args.include_arxiv,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1 if isinstance(exc, ValueError) else 2

    assert bundle.classification is not None
    if args.output.suffix.lower() == ".csv":
        shutil.copy2(bundle.classification.csv_path, args.output)
        print(f"Wrote {bundle.classification.row_count} rows to {args.output}")
    else:
        print(
            f"Wrote {bundle.classification.row_count} rows to "
            f"{bundle.classification.csv_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
