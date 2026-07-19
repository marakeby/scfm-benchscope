"""Compatibility wrapper; implementation lives in scfm_cancer_eval.reporting."""

from __future__ import annotations

from scfm_cancer_eval.reporting.embedding_bootstrap import *  # noqa: F403
from scfm_cancer_eval.reporting.embedding_bootstrap import (
    META_JSON,
    RUNS_CSV,
    densify_embedding_obsm,
    default_stratify_key,
    load_embedding_h5ad,
    merge_embedding_bootstrap_into_results_json,
    rebuild_results_json_from_run_dir,
    run_bootstrap_for_run_dir,
    run_subsampled_eval,
    save_embedding_bootstrap_artifacts,
)

__all__ = [
    "META_JSON",
    "RUNS_CSV",
    "densify_embedding_obsm",
    "default_stratify_key",
    "load_embedding_h5ad",
    "merge_embedding_bootstrap_into_results_json",
    "rebuild_results_json_from_run_dir",
    "run_bootstrap_for_run_dir",
    "run_subsampled_eval",
    "save_embedding_bootstrap_artifacts",
]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Repeated stratified subsample embedding metrics; writes CSV/meta "
            "and updates results.json."
        )
    )
    p.add_argument("run_dir", help="Experiment output dir containing data.h5ad")
    p.add_argument("--embedding-key", required=True, help="obsm key, e.g. X_pca")
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument(
        "--stratify-by",
        default=None,
        help="obs column; default batch|batch+label like calc_metrics.py",
    )
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--no-merge-results",
        action="store_true",
        help="Do not update results.json",
    )
    args = p.parse_args()
    run_bootstrap_for_run_dir(
        args.run_dir,
        embedding_key=args.embedding_key,
        n_runs=args.n_runs,
        sample_size=args.sample_size,
        stratify_by=args.stratify_by,
        base_seed=args.base_seed,
        n_jobs=args.n_jobs,
        merge_into_results_json=not args.no_merge_results,
    )
