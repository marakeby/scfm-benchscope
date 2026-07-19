#!/usr/bin/env python3
"""Collect embedding evaluation CSVs into one table (CLI wrapper)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scfm_cancer_eval.reporting.collect_metrics import create_collect_metrics_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "embedding.metrics.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(
        os.environ.get("SCFM_EMBEDDING_OUTPUT_ROOT")
        or os.environ.get("SCFM_CLASSIFICATION_OUTPUT_ROOT")
        or os.environ.get("SCFM_OUTPUT_PATH")
        or "output"
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=default_root,
        help="Root directory to search",
    )
    parser.add_argument(
        "--folder-substring",
        default="cell_type",
        help="Require this substring in at least one path component",
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
    if not root.exists():
        alt = REPO_ROOT / "output" / "exp"
        if alt.exists():
            print(f"Using {alt} (default output root not found).", file=sys.stderr)
            root = alt

    try:
        bundle = create_collect_metrics_bundle(
            root,
            args.output.parent,
            kind="embedding",
            folder_substring=args.folder_substring,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1 if isinstance(exc, ValueError) else 2

    assert bundle.embedding is not None
    # Preserve historical single-file output path when -o points at a CSV.
    if args.output.suffix.lower() == ".csv":
        import shutil

        shutil.copy2(bundle.embedding.csv_path, args.output)
        print(f"Wrote {bundle.embedding.row_count} rows to {args.output}")
    else:
        print(
            f"Wrote {bundle.embedding.row_count} rows to {bundle.embedding.csv_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
