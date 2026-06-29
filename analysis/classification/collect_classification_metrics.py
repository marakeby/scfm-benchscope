#!/usr/bin/env python3
"""
Scan experiment output trees for vote / avg / MIL CV metrics CSVs, attach run
metadata from run_summary.json, and write one combined table (same logic as
collect_classification_metrics.ipynb).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "classification.metrics.csv"

EXPERIMENT_NAME_MAP: dict[str, str] = {
    "pre_post": "Treatment Naive vs Anti PD1",
    "brca_full_pre_post": "Treatment Naive vs Anti PD1",
    "brca_pre_post": "Treatment Naive vs Anti PD1",
    "chemo": "Treatment Naive vs Neoadjuvant Chemo",
    "brca_full_chemo": "Treatment Naive vs Neoadjuvant Chemo",
    "brca_chemo": "Treatment Naive vs Neoadjuvant Chemo",
    "luad2": "Treatment Naive vs TKI treated",
    "luad_tki": "Treatment Naive vs TKI treated",
    "luad1": "Early stage vs Late stage",
    "outcome": "T-cell exhaustion",
    "brca_full_outcome": "T-cell exhaustion",
    "brca_outcome": "T-cell exhaustion",
    "subtype": "ER+ vs TNBC",
    "brca_full_subtype": "ER+ vs TNBC",
    "brca_subtype": "ER+ vs TNBC",
    "brca_cell_type": "BRCA Cell Type",
    "melanoma_response": "IO Response",
    "crc_mmr": "MMRd vs MMRp",
}

# Display names for run_id stems (aligned with collect_classification_metrics.ipynb)
MODEL_NAME_MAP: dict[str, str] = {
    "hvg": "HVG",
    "pca": "PCA",
    "scgpt": "scGPT",
    "scgpt_cancer": "scGPT [cancer]",
    "scvi": "scVI",
    "scvi_donor_id": "scVI",
    "scfoundation": "scFoundation",
    "scimilarity": "SCimiarity",
    "cellplm": "CellPLM",
    "gf-6L-30M-i2048": "GF-V1",
    "gf-6L-30M-i2048_continue": "GF-V1 [continue]",
    "Geneformer-V2-104M_CLcancer": "GF-V2 [cancer]",
    "Geneformer-V2-104M": "GF-V2",
    "Geneformer-V2-104M_continue": "GF-V2 [continue]",
    "Geneformer-V2-316M": "GF-V2-Deep",
    "gf-6L-30M-i2048_finetune": "GF-V1 [finetune]",
    "Geneformer-V2-104M_finetune": "GF-V2 [finetune]",
    "hvg_seurat_4096": "HVG",
    "state_se600m_epoch16": "STATE",
    "scfoundation_brca_cancer_cells": "scFoundation",
    "geneformer_V2-104M_CLcancer-i4096": "GF-V2 [cancer]",
    "geneformer_V2-316M-i4096": "GF-V2-Deep",
    "continue_geneformer_V1-10M-i2048_continue": "GF-V1 [continue]",
    "geneformer_V1-10M-i2048": "GF-V1",
    "continue_geneformer_V2-104M-i4096_continue": "GF-V2 [continue]",
    "geneformer_V2-104M-i4096": "GF-V2",
    "scgpt_cancer-i2048": "scGPT [cancer]",
    "scgpt_human-i2048": "scGPT",
    "cellplm_85M-20231027": "CellPLM",
    "scimilarity_v1.1": "SCimiarity",
    "pca_n100": "PCA [100]",
    "pca_n50": "PCA [50]",
    "pca_n20": "PCA [20]",
    "scconcept_corpus30m": "scConcept",
    "nicheformer_nicheformer": "Nicheformer",
}


def map_groups(model_id: str) -> str:
    """Broad model family for plotting (same logic as analysis_utils.map_groups)."""
    exp = model_id.lower()
    if "gf" in exp:
        return "Geneformer"
    if "geneformer" in exp:
        return "Geneformer"
    if "scfoundation" in exp:
        return "Other"
    if "scimilarity" in exp:
        return "Other"
    if "scgpt" in exp:
        return "scGPT"
    if "cellplm" in exp:
        return "Other"
    if any(x in exp for x in ("hvg", "pca", "scvi")):
        return "Baseline"
    return "Other"


def find_files(root_folder: Path, filename: str) -> list[Path]:
    if not root_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_folder}")
    return list(root_folder.rglob(filename))


def _load_run_frame(
    matched_file: Path,
    strategy: str,
    score_col: str,
) -> pd.DataFrame | None:
    df = pd.read_csv(matched_file)
    if score_col not in df.columns:
        print(
            f"Skipping {matched_file}: no '{score_col}' column "
            f"(have {list(df.columns)})",
            file=sys.stderr,
        )
        return None
    df = df.pivot(index="Metrics", columns="fold", values=score_col).T
    run_summary_path = matched_file.parent.parent / "run_summary.json"
    try:
        with open(run_summary_path, encoding="utf-8") as f:
            run_summary = json.load(f)
        model = run_summary["run_id"]
        exp = Path(run_summary["config_path"]).stem
        model = model.replace(f"_{exp}", "")
    except OSError as e:
        print(f"Cannot load summary file, skipping: {run_summary_path} ({e})", file=sys.stderr)
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Invalid run_summary.json, skipping: {run_summary_path} ({e})", file=sys.stderr)
        return None

    df["model"] = model
    df["exp"] = exp
    df["strategy"] = strategy
    return df


def collect_strategy_dfs(
    files: list[Path],
    strategy: str,
    score_col: str,
) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    for matched_file in files:
        frame = _load_run_frame(matched_file, strategy, score_col)
        if frame is not None:
            out.append(frame)
    return out


def concat_reset(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs)
    return combined.reset_index().drop(columns=["fold"], errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(
            os.environ.get("SCFM_CLASSIFICATION_OUTPUT_ROOT", "/home/haitham/mnt/__output_v7/exp")
        ),
        help="Root directory to search (default: $SCFM_CLASSIFICATION_OUTPUT_ROOT or "
        "/home/haitham/mnt/__output_v7/exp)",
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
        help="Column name in *cv_metrics.csv holding classifier scores (default: randomforest)",
    )
    parser.add_argument(
        "--keep-luad-cancer-stage",
        action="store_true",
        help="Do not drop rows where exp == luad_cancer_stage (notebook drops them)",
    )
    args = parser.parse_args()

    root = args.root
    default_root = Path(
        os.environ.get("SCFM_CLASSIFICATION_OUTPUT_ROOT", "/home/haitham/mnt/__output_v7/exp")
    )
    if not root.exists() and root.resolve() == default_root.resolve():
        alt = REPO_ROOT / "output" / "exp"
        if alt.exists():
            print(f"Using {alt} (default output root not found).", file=sys.stderr)
            root = alt

    try:
        vote_files = find_files(root, "vote_cv_metrics.csv")
        avg_files = find_files(root, "avg_cv_metrics.csv")
        mil_files = find_files(root, "mil_cv_metrics.csv")
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        print(
            "Pass a valid output tree (containing .../cv/*_cv_metrics.csv) or set "
            "SCFM_CLASSIFICATION_OUTPUT_ROOT.",
            file=sys.stderr,
        )
        return 2

    vote_df = concat_reset(collect_strategy_dfs(vote_files, "vote", args.score_col))
    avg_df = concat_reset(collect_strategy_dfs(avg_files, "avg", args.score_col))
    mil_df = concat_reset(collect_strategy_dfs(mil_files, "MIL", args.score_col))

    parts = [d for d in (vote_df, mil_df, avg_df) if not d.empty]
    if not parts:
        print(f"No metrics found under {root}", file=sys.stderr)
        return 1

    classification_metrics = pd.concat(parts, ignore_index=True)

    if not args.keep_luad_cancer_stage and "exp" in classification_metrics.columns:
        classification_metrics = classification_metrics[classification_metrics["exp"] != "luad_cancer_stage"]

    classification_metrics["group"] = classification_metrics["model"].map(map_groups)
    classification_metrics["model"] = classification_metrics["model"].map(MODEL_NAME_MAP)
    classification_metrics["exp"] = classification_metrics["exp"].map(EXPERIMENT_NAME_MAP)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    classification_metrics.to_csv(args.output, index=False)
    print(f"Wrote {len(classification_metrics)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
