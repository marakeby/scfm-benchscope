"""Batch embedding bootstrap aggregation for dashboard exports."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import pandas as pd

from scfm_cancer_eval.reporting.collect_metrics import write_table_exports
from scfm_cancer_eval.reporting.display_names import MODEL_NAME_MAP, map_groups
from scfm_cancer_eval.reporting.embedding_bootstrap import (
    META_JSON,
    RUNS_CSV,
    densify_embedding_obsm,
    default_stratify_key,
    run_subsampled_eval,
    save_embedding_bootstrap_artifacts,
)


@dataclass(frozen=True)
class BootstrapMetricsBundle:
    mean_csv: Path
    std_csv: Path
    median_csv: Path
    bootstrap_csv: Path
    bootstrap_json: Path
    failed_csv: Path | None
    model_count: int
    failure_count: int


def _log(message: str) -> None:
    """Print a progress line immediately (important on slow mounts / long evals)."""
    print(message, flush=True)


def get_embedding_key(model_name: str) -> str | None:
    key = None
    name = model_name.lower()
    if "pca" in name:
        key = "X_pca"
    if "hvg" in name:
        key = "X_hvg"
    if "scvi" in name:
        key = "X_scVI"
    if "scgpt" in name:
        key = "X_scGPT"
    if "geneformer" in name or "gf" in name:
        key = "X_geneformer"
    if "scfoundation" in name:
        key = "X_scfoundation"
    if "cellplm" in name:
        key = "X_CellPLM"
    if "scimilarity" in name:
        key = "X_scimilarity"
    if "nicheformer" in name:
        key = "X_nicheformer"
    if "scconcept" in name:
        key = "X_scconcept"
    if "state" in name:
        key = "X_state"
    return key


def extract_model_name(filename: str, marker: str) -> str:
    """Extract text between two occurrences of ``marker`` in a path basename/path."""
    parts = filename.split(marker)
    if len(parts) < 3:
        raise ValueError(f"Marker '{marker}' must appear at least twice.")
    return parts[1].strip("_")


def find_embedding_run_folders(root: Path, folder_substring: str) -> list[Path]:
    _log(
        f"Scanning for run folders under {root} "
        f"(name contains {folder_substring!r})…"
    )
    started = time.perf_counter()
    folders: list[Path] = []
    dirs_visited = 0
    for dirpath, dirnames, _filenames in os.walk(root):
        dirs_visited += 1
        if dirs_visited == 1 or dirs_visited % 100 == 0:
            _log(
                f"  walked {dirs_visited} director(ies); "
                f"matched {len(folders)} so far"
            )
        for directory in dirnames:
            if folder_substring in directory:
                folders.append(Path(dirpath) / directory)
    folders = sorted(folders)
    _log(
        f"Found {len(folders)} matching run folder(s) "
        f"after walking {dirs_visited} director(ies) "
        f"in {time.perf_counter() - started:.1f}s."
    )
    return folders


def _annotate_by_model_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ids = out.index.astype(str)
    out.insert(0, "group", ids.map(map_groups))
    out.insert(0, "model_display", ids.map(MODEL_NAME_MAP))
    return out


def _annotate_by_model_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" not in out.columns:
        return out
    ids = out["model"].astype(str)
    loc = list(out.columns).index("model") + 1
    out.insert(loc, "model_display", ids.map(MODEL_NAME_MAP))
    out.insert(loc + 1, "group", ids.map(map_groups))
    return out


def create_bootstrap_metrics_bundle(
    root: str | Path,
    output_dir: str | Path,
    *,
    folder_substring: str = "cell_type",
    experiment_marker: str = "brca_cell_type",
    sample_size: int = 10000,
    n_runs: int = 10,
    base_seed: int = 42,
    n_jobs: int = -1,
    merge_into_results_json: bool = True,
) -> BootstrapMetricsBundle:
    """
    Run repeated subsampled embedding evaluation for matching run folders.

    Writes aggregate tables under ``output_dir`` (compatible with
    ``docs/embeddings.html``):

    - ``embedding.metrics.mean.csv``
    - ``embedding.metrics.std.csv``
    - ``embedding.metrics.median.csv``
    - ``embedding.metrics.bootstrap.csv`` / ``.json``
    """
    root_path = Path(root)
    out = Path(output_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {root_path}")

    overall_started = time.perf_counter()
    _log(f"Bootstrap metrics: root={root_path}")
    _log(f"Bootstrap metrics: output={out}")
    _log(
        f"Bootstrap metrics: n_runs={n_runs}, sample_size={sample_size}, "
        f"base_seed={base_seed}, n_jobs={n_jobs}, "
        f"marker={experiment_marker!r}, merge_results={merge_into_results_json}"
    )

    folders = find_embedding_run_folders(root_path, folder_substring)
    if not folders:
        raise ValueError(
            f"No folders under {root_path!r} contain {folder_substring!r} "
            "in their name."
        )

    bootstrap_mirror_dir = out / "bootstrap"
    results_mean: list[pd.Series] = []
    results_std: list[pd.Series] = []
    results_median: list[pd.Series] = []
    all_results: dict[str, pd.DataFrame] = {}
    failed_models: list[dict[str, object]] = []

    total = len(folders)
    _log(f"Running bootstrap for {total} folder(s)…")

    for folder_index, folder in enumerate(folders, start=1):
        model_name: str | None = None
        embedding_key: str | None = None
        folder_str = str(folder)
        _log(f"[{folder_index}/{total}] {folder.name}")
        try:
            model_name = extract_model_name(folder_str, experiment_marker)
            embedding_key = get_embedding_key(model_name)
            if embedding_key is None:
                raise ValueError(f"No embedding key found for model {model_name!r}")
            _log(f"  model={model_name} embedding_key={embedding_key}")

            h5ad_path = folder / "data.h5ad"
            _log(f"  loading {h5ad_path}…")
            load_started = time.perf_counter()
            embs = ad.read_h5ad(h5ad_path)
            _log(
                f"  loaded AnnData n_obs={embs.n_obs}, n_vars={embs.n_vars} "
                f"in {time.perf_counter() - load_started:.1f}s"
            )
            if embedding_key not in embs.obsm:
                raise AssertionError(
                    f"embedding key {embedding_key} not found in {list(embs.obsm.keys())}"
                )
            densify_embedding_obsm(embs, embedding_key)
            stratify_key = default_stratify_key(embs)
            _log(f"  stratify_by={stratify_key!r}")

            def _run_progress(message: str, *, _idx: int = folder_index) -> None:
                _log(f"  [{_idx}/{total}] {message}")

            _log(
                f"  starting {n_runs} subsampled evaluation(s) "
                f"(sample_size={sample_size})…"
            )
            time_start = time.perf_counter()
            df = run_subsampled_eval(
                embs,
                embedding_key=embedding_key,
                n_runs=n_runs,
                sample_size=sample_size,
                stratify_by=stratify_key,
                base_seed=base_seed,
                save_dir=folder_str,
                n_jobs=n_jobs,
                progress=_run_progress,
            )
            time_end = time.perf_counter()

            _log("  saving bootstrap artifacts / updating results.json…")
            save_embedding_bootstrap_artifacts(
                folder_str,
                df,
                embedding_key=embedding_key,
                n_runs=n_runs,
                sample_size=sample_size,
                stratify_by=stratify_key,
                base_seed=base_seed,
                n_jobs=n_jobs,
                merge_into_results_json=merge_into_results_json,
            )

            bootstrap_mirror_dir.mkdir(parents=True, exist_ok=True)
            run_tag = folder.name
            src_runs = folder / RUNS_CSV
            src_meta = folder / META_JSON
            try:
                shutil.copy2(src_runs, bootstrap_mirror_dir / f"{run_tag}_{RUNS_CSV}")
                shutil.copy2(src_meta, bootstrap_mirror_dir / f"{run_tag}_{META_JSON}")
                _log(f"  mirrored artifacts → {bootstrap_mirror_dir}")
            except OSError as exc:
                _log(f"[WARN] Could not mirror bootstrap artifacts: {exc}")

            mean_row = df.mean(numeric_only=True)
            mean_row["model"] = model_name
            results_mean.append(mean_row)

            std_row = df.std(numeric_only=True, ddof=1)
            std_row["model"] = model_name
            results_std.append(std_row)

            median_row = df.median(numeric_only=True)
            median_row["model"] = model_name
            results_median.append(median_row)

            all_results[model_name] = df
            _log(
                f"[{folder_index}/{total}] OK model={model_name} "
                f"embedding_key={embedding_key} "
                f"n_runs={n_runs} sample_size={sample_size} "
                f"time={time_end - time_start:.2f}s "
                f"(successes={len(results_mean)}, failures={len(failed_models)})"
            )
        except Exception as exc:
            failed_models.append(
                {
                    "folder": folder_str,
                    "model": model_name,
                    "embedding_key": embedding_key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"[FAILED] [{folder_index}/{total}] folder={folder_str} "
                f"model={model_name} embedding={embedding_key} err={exc!r}",
                file=sys.stderr,
                flush=True,
            )

    out.mkdir(parents=True, exist_ok=True)
    failed_csv: Path | None = None
    if failed_models:
        failed_csv = out / "failed_models.csv"
        pd.DataFrame(failed_models).to_csv(failed_csv, index=False)
        _log(f"Wrote failure details → {failed_csv}")

    if not results_mean:
        details = (
            f"folders_scanned={len(folders)}, failures={len(failed_models)}"
        )
        raise ValueError(
            "No models completed embedding bootstrap successfully "
            f"({details})"
        )

    _log(
        f"Aggregating results for {len(results_mean)} successful model(s) "
        f"({len(failed_models)} failed)…"
    )
    mean_df = _annotate_by_model_index(
        pd.concat(results_mean, axis=1).T.set_index("model")
    )
    std_df = _annotate_by_model_index(
        pd.concat(results_std, axis=1).T.set_index("model")
    )
    median_df = _annotate_by_model_index(
        pd.concat(results_median, axis=1).T.set_index("model")
    )
    bootstrap_df = _annotate_by_model_column(
        pd.concat(all_results, names=["model", "run"]).reset_index(level=0)
    )

    mean_csv = out / "embedding.metrics.mean.csv"
    std_csv = out / "embedding.metrics.std.csv"
    median_csv = out / "embedding.metrics.median.csv"
    _log(f"Writing aggregate CSVs under {out}…")
    mean_df.to_csv(mean_csv)
    std_df.to_csv(std_csv)
    median_df.to_csv(median_csv)

    bootstrap_export = write_table_exports(
        bootstrap_df.reset_index(),
        out / "embedding.metrics.bootstrap",
        schema_name="scfm_eval.embedding_bootstrap_metrics",
    )
    _log(
        f"Wrote bootstrap table → {bootstrap_export.csv_path} / "
        f"{bootstrap_export.json_path}"
    )

    # Keep a compact aggregate sidecar for mean/std/median as JSON too.
    aggregate_payload = {
        "schema": {
            "name": "scfm_eval.embedding_bootstrap_aggregates",
            "version": "1.0.0",
        },
        "mean": json.loads(mean_df.reset_index().to_json(orient="records")),
        "std": json.loads(std_df.reset_index().to_json(orient="records")),
        "median": json.loads(median_df.reset_index().to_json(orient="records")),
    }
    aggregates_path = out / "embedding.metrics.aggregates.json"
    aggregates_path.write_text(
        json.dumps(aggregate_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"Wrote aggregates JSON → {aggregates_path}")
    _log(
        f"Bootstrap metrics finished in "
        f"{time.perf_counter() - overall_started:.1f}s."
    )

    return BootstrapMetricsBundle(
        mean_csv=mean_csv,
        std_csv=std_csv,
        median_csv=median_csv,
        bootstrap_csv=bootstrap_export.csv_path,
        bootstrap_json=bootstrap_export.json_path,
        failed_csv=failed_csv,
        model_count=len(results_mean),
        failure_count=len(failed_models),
    )
