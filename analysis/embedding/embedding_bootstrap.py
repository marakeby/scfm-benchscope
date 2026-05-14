"""
Repeated stratified subsampling for embedding metrics (post-hoc / batch analysis).

This matches the workflow in ``calc_metrics.py``: subsample cells several times,
run :class:`evaluation.eval.EmbeddingEvaluator` on each subsample, then aggregate.

Artifacts written under a run directory (same folder as ``data.h5ad``):

- ``embedding_bootstrap_runs.csv`` — one row per repeat; column ``run`` + metric columns
- ``embedding_bootstrap_meta.json`` — protocol parameters (n_runs, sample_size, …)
- Optionally updates ``results.json`` via :func:`merge_embedding_bootstrap_into_results_json`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scfm_cancer_eval.evaluation.eval import EmbeddingEvaluator
from scfm_cancer_eval.utils.sampling import sample_adata

RUNS_CSV = "embedding_bootstrap_runs.csv"
META_JSON = "embedding_bootstrap_meta.json"


def run_subsampled_eval(
    adata: ad.AnnData,
    *,
    embedding_key: str,
    n_runs: int = 3,
    sample_size: int = 10000,
    stratify_by: str | None = "batch",
    base_seed: int = 42,
    save_dir: str = "./results",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Repeatedly subsample ``adata``, evaluate embedding metrics, return one row per run.
    """
    mid_results: list[dict[str, Any]] = []
    for i in range(n_runs):
        subsampled_adata = sample_adata(
            adata,
            sample_size=sample_size,
            stratify_by=stratify_by,
            random_state=base_seed + i,
        )
        evaluator = EmbeddingEvaluator(
            subsampled_adata,
            embedding_key,
            save_dir=save_dir,
            save_id=f"{embedding_key}_{i}",
            n_jobs=n_jobs,
            random_state=base_seed + i,
        )
        results_dict = evaluator.evaluate()
        mid_results.append(results_dict)

    return pd.DataFrame(mid_results)


def save_embedding_bootstrap_artifacts(
    run_dir: str,
    df: pd.DataFrame,
    *,
    embedding_key: str,
    n_runs: int,
    sample_size: int,
    stratify_by: str | None,
    base_seed: int,
    n_jobs: int,
    merge_into_results_json: bool = True,
) -> None:
    """
    Write bootstrap/subsample run table + meta next to ``run_dir/data.h5ad``.
    Optionally merge a matching ``embedding`` evaluation into ``results.json``.
    """
    os.makedirs(run_dir, exist_ok=True)
    out_csv = os.path.join(run_dir, RUNS_CSV)
    df_out = df.copy()
    df_out.insert(0, "run", range(len(df_out)))
    df_out.to_csv(out_csv, index=False)

    meta = {
        "embedding_eval_protocol": "repeated_stratified_subsample",
        "embedding_key": embedding_key,
        "n_runs": n_runs,
        "sample_size": sample_size,
        "stratify_by": stratify_by,
        "base_seed": base_seed,
        "n_jobs": n_jobs,
    }
    with open(os.path.join(run_dir, META_JSON), "w") as f:
        json.dump(meta, f, indent=2)

    if merge_into_results_json:
        results_path = os.path.join(run_dir, "results.json")
        summary_path = os.path.join(run_dir, "run_summary.json")
        if os.path.isfile(results_path):
            merge_embedding_bootstrap_into_results_json(run_dir, embedding_key=embedding_key)
        elif os.path.isfile(summary_path):
            # Full results.json (includes bootstrap) from run metadata + metrics.json
            rebuild_results_json_from_run_dir(run_dir)
        # else: orphan analysis dir — skip


def load_embedding_h5ad(run_dir: str) -> ad.AnnData:
    """Load ``data.h5ad`` from a standard experiment output directory."""
    path = os.path.join(run_dir, "data.h5ad")
    embs = ad.read_h5ad(path)
    return embs


def densify_embedding_obsm(embs: ad.AnnData, embedding_key: str) -> None:
    """Ensure ``obsm[embedding_key]`` is dense ndarray (in-place)."""
    x = embs.obsm[embedding_key]
    if sp.issparse(x):
        embs.obsm[embedding_key] = np.asarray(x.toarray())


def default_stratify_key(embs: ad.AnnData) -> str | None:
    """Match ``calc_metrics.py``: batch+label composite when both exist."""
    if "batch" in embs.obs.columns and "label" in embs.obs.columns:
        embs.obs["batch_label"] = (
            embs.obs["batch"].astype(str) + "|" + embs.obs["label"].astype(str)
        )
        return "batch_label"
    if "batch" in embs.obs.columns:
        return "batch"
    return None


def run_bootstrap_for_run_dir(
    run_dir: str,
    *,
    embedding_key: str,
    n_runs: int = 10,
    sample_size: int = 10000,
    stratify_by: str | None = None,
    base_seed: int = 42,
    n_jobs: int = -1,
    merge_into_results_json: bool = True,
) -> pd.DataFrame:
    """
    End-to-end: load ``run_dir/data.h5ad``, run repeated subsampled embedding eval, save artifacts.
    If ``stratify_by`` is None, uses the same default as ``calc_metrics.py``.
    """
    embs = load_embedding_h5ad(run_dir)
    densify_embedding_obsm(embs, embedding_key)
    if stratify_by is None:
        stratify_by = default_stratify_key(embs)

    df = run_subsampled_eval(
        embs,
        embedding_key=embedding_key,
        n_runs=n_runs,
        sample_size=sample_size,
        stratify_by=stratify_by,
        base_seed=base_seed,
        save_dir=run_dir,
        n_jobs=n_jobs,
    )
    save_embedding_bootstrap_artifacts(
        run_dir,
        df,
        embedding_key=embedding_key,
        n_runs=n_runs,
        sample_size=sample_size,
        stratify_by=stratify_by,
        base_seed=base_seed,
        n_jobs=n_jobs,
        merge_into_results_json=merge_into_results_json,
    )
    return df


def merge_embedding_bootstrap_into_results_json(
    run_dir: str,
    *,
    embedding_key: str | None = None,
) -> bool:
    """Delegate to :func:`utils.results_json.merge_embedding_bootstrap_into_results_json`."""
    from scfm_cancer_eval.utils.results_json import merge_embedding_bootstrap_into_results_json as _merge

    return _merge(run_dir, embedding_key=embedding_key)


def rebuild_results_json_from_run_dir(run_dir: str, *, repo_root: str | None = None) -> None:
    """
    Rebuild ``results.json`` from ``run_summary.json`` + on-disk metrics (including bootstrap).
    Use when ``results.json`` was never written or is stale.
    """
    from scfm_cancer_eval.utils.results_json import build_results_json, read_run_inputs_for_results_json, write_results_json

    if repo_root is None:
        repo_root = str(_PROJECT_ROOT)

    inputs = read_run_inputs_for_results_json(run_dir, repo_root=repo_root)
    write_results_json(os.path.join(run_dir, "results.json"), build_results_json(**inputs))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Repeated stratified subsample embedding metrics; writes CSV/meta and updates results.json."
    )
    p.add_argument("run_dir", help="Experiment output dir containing data.h5ad")
    p.add_argument("--embedding-key", required=True, help="obsm key, e.g. X_pca")
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument("--stratify-by", default=None, help="obs column; default batch|batch+label like calc_metrics.py")
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
