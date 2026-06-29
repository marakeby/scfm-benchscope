#!/usr/bin/env python3
"""
Repeated subsampled embedding evaluation (bootstrap-style) for runs under
``SCFM_CALC_METRICS_ROOT``, mirroring ``calc_metrics.py`` logic.

Writes aggregate CSVs and mirrored per-run bootstrap artifacts under the repo
``results`` tree (same convention as ``collect_embedding_metrics.py``):
``<repo>/results/embedding_bootstrap/``.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import pandas as pd

try:
    from .embedding_bootstrap import (
        META_JSON,
        RUNS_CSV,
        densify_embedding_obsm,
        default_stratify_key,
        run_subsampled_eval,
        save_embedding_bootstrap_artifacts,
    )
except ImportError:
    from embedding_bootstrap import (
        META_JSON,
        RUNS_CSV,
        densify_embedding_obsm,
        default_stratify_key,
        run_subsampled_eval,
        save_embedding_bootstrap_artifacts,
    )
from os.path import join

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "embedding_bootstrap"
bootstrap_mirror_dir = RESULTS_DIR / "bootstrap"

base_dir: str
_env_root = os.environ.get("SCFM_CALC_METRICS_ROOT")
_default_mount = Path("/home/haitham/mnt/__output_v7/exp")
if _env_root is not None:
    base_path = Path(_env_root)
else:
    base_path = _default_mount
if not base_path.exists() and _env_root is None:
    alt = REPO_ROOT / "output" / "exp"
    if alt.exists():
        print(f"Using {alt} (default SCFM_CALC_METRICS_ROOT mount not found).", file=sys.stderr)
        base_path = alt
base_dir = str(base_path)
if not Path(base_dir).exists():
    print(f"Experiment root does not exist: {base_dir}", file=sys.stderr)
    raise SystemExit(2)

# Display names for extracted model stems (aligned with collect_classification_metrics.py)
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
    """Broad model family for plotting (same logic as collect_classification_metrics.py)."""
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


def _annotate_by_model_index(df: pd.DataFrame) -> pd.DataFrame:
    """Insert model_display and group; index is raw model id."""
    out = df.copy()
    ids = out.index.astype(str)
    out.insert(0, "group", ids.map(map_groups))
    out.insert(0, "model_display", ids.map(MODEL_NAME_MAP))
    return out


def _annotate_by_model_column(df: pd.DataFrame) -> pd.DataFrame:
    """Insert model_display and group right after the ``model`` column if present."""
    out = df.copy()
    if "model" not in out.columns:
        return out
    ids = out["model"].astype(str)
    loc = list(out.columns).index("model") + 1
    out.insert(loc, "model_display", ids.map(MODEL_NAME_MAP))
    out.insert(loc + 1, "group", ids.map(map_groups))
    return out


def get_embedding_key(model_name: str) -> str | None:
    key = None
    if "pca" in model_name:
        key = "X_pca"
    if "hvg" in model_name:
        key = "X_hvg"
    if "scvi" in model_name:
        key = "X_scVI"
    if "scgpt" in model_name:
        key = "X_scGPT"
    if "geneformer" in model_name:
        key = "X_geneformer"
    if "gf" in model_name:
        key = "X_geneformer"
    if "scfoundation" in model_name:
        key = "X_scfoundation"
    if "cellplm" in model_name:
        key = "X_CellPLM"
    if "scimilarity" in model_name:
        key = "X_scimilarity"
    if "nicheformer" in model_name:
        key = "X_nicheformer"
    if "scconcept" in model_name:
        key = "X_scconcept"
    if "state" in model_name:
        key = "X_state"
    if key is None:
        print("no key found")
    return key


def extract_model_name(filename: str, marker: str) -> str:
    """
    Extract the text between two occurrences of `marker`.

    Example:
        filename = "brca_cell_type_hvg_seurat_4096_brca_cell_type"
        marker = "brca_cell_type"
        returns: "hvg_seurat_4096"
    """
    parts = filename.split(marker)
    if len(parts) < 3:
        raise ValueError(f"Marker '{marker}' must appear at least twice.")
    return parts[1].strip("_")


search_text = "cell_type"
folders: list[str] = []
for root, dirs, _files in os.walk(base_dir):
    for directory in dirs:
        if search_text in directory:
            print(os.path.join(root, directory))
            folders.append(os.path.join(root, directory))

sample_size = 10000
n_runs = 10
base_seed = 42
n_jobs = -1
save_dir = str(RESULTS_DIR)

results_mean: list[pd.Series] = []
results_std: list[pd.Series] = []
results_median: list[pd.Series] = []
all_results: dict[str, pd.DataFrame] = {}
failed_models: list[dict] = []

for f in folders:
    model_name = None
    embedding_key = None
    try:
        model_name = extract_model_name(f, "brca_cell_type")
        embedding_key = get_embedding_key(model_name)
        print(embedding_key)

        data_fname = join(f, "data.h5ad")
        embs = ad.read_h5ad(data_fname)
        print(embs)
        assert embedding_key in embs.obsm, (
            f"embedding key {embedding_key} not found in {list(embs.obsm.keys())}"
        )
        print(f"embedding key {embedding_key} found in {embs.obsm.keys()}")

        densify_embedding_obsm(embs, embedding_key)

        time_start = time.perf_counter()

        stratify_key = default_stratify_key(embs)

        df = run_subsampled_eval(
            embs,
            embedding_key=embedding_key,
            n_runs=n_runs,
            sample_size=sample_size,
            stratify_by=stratify_key,
            base_seed=base_seed,
            save_dir=f,
            n_jobs=n_jobs,
        )
        time_end = time.perf_counter()

        save_embedding_bootstrap_artifacts(
            f,
            df,
            embedding_key=embedding_key,
            n_runs=n_runs,
            sample_size=sample_size,
            stratify_by=stratify_key,
            base_seed=base_seed,
            n_jobs=n_jobs,
            merge_into_results_json=True,
        )

        bootstrap_mirror_dir.mkdir(parents=True, exist_ok=True)
        run_tag = os.path.basename(os.path.normpath(f))
        src_runs = join(f, RUNS_CSV)
        src_meta = join(f, META_JSON)
        dst_runs = str(bootstrap_mirror_dir / f"{run_tag}_{RUNS_CSV}")
        dst_meta = str(bootstrap_mirror_dir / f"{run_tag}_{META_JSON}")
        try:
            shutil.copy2(src_runs, dst_runs)
            shutil.copy2(src_meta, dst_meta)
        except OSError as exc:
            print(
                f"[WARN] Could not mirror bootstrap artifacts to {bootstrap_mirror_dir}: {exc}",
                flush=True,
            )

        dd = df.mean(numeric_only=True)
        dd["model"] = model_name
        results_mean.append(dd)

        dd = df.std(numeric_only=True, ddof=1)
        dd["model"] = model_name
        results_std.append(dd)

        dd = df.median(numeric_only=True)
        dd["model"] = model_name
        results_median.append(dd)

        all_results[model_name] = df

        print(f"model: {model_name}")
        print(f"embedding_key: {embedding_key}")
        print(f"sample_size: {sample_size}")
        print(f"n_runs: {n_runs}")
        print(f"base_seed: {base_seed}")
        print(f"aggregate_save_dir: {save_dir}")
        print(f"n_jobs: {n_jobs}")
        print(f"time taken: {time_end - time_start:.2f} seconds")
    except Exception as exc:
        failed_models.append(
            {
                "folder": f,
                "model": model_name,
                "embedding_key": embedding_key,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        print(f"[FAILED] folder={f} model={model_name} embedding={embedding_key} err={exc!r}")
        continue

if not results_mean:
    os.makedirs(save_dir, exist_ok=True)
    print(
        "No models completed successfully; skipping aggregate CSVs "
        f"(folders_scanned={len(folders)}, failures={len(failed_models)})."
    )
    if not folders:
        print(f"No directories under {base_dir!r} contain {search_text!r} in their name.")
    if failed_models:
        failed_df = pd.DataFrame(failed_models)
        failed_df.to_csv(f"{save_dir}/failed_models.csv", index=False)
        print(f"Wrote failure details to {save_dir}/failed_models.csv")
    raise SystemExit(1)

results_mean_df = _annotate_by_model_index(
    pd.concat(results_mean, axis=1).T.set_index("model")
)
results_std_df = _annotate_by_model_index(
    pd.concat(results_std, axis=1).T.set_index("model")
)
results_median_df = _annotate_by_model_index(
    pd.concat(results_median, axis=1).T.set_index("model")
)

all_results_df = _annotate_by_model_column(
    pd.concat(all_results, names=["model", "run"]).reset_index(level=0)
)
os.makedirs(save_dir, exist_ok=True)
results_mean_df.to_csv(f"{save_dir}/embedding.metrics.mean.csv")
results_std_df.to_csv(f"{save_dir}/embedding.metrics.std.csv")
results_median_df.to_csv(f"{save_dir}/embedding.metrics.median.csv")
all_results_df.to_csv(f"{save_dir}/embedding.metrics.bootstrap.csv")
print(f"Wrote aggregate CSVs under {RESULTS_DIR}")
print(f"Mirrored per-run bootstrap CSV/JSON under {bootstrap_mirror_dir}")

if failed_models:
    failed_df = pd.DataFrame(failed_models)
    failed_df.to_csv(f"{save_dir}/failed_models.csv", index=False)
    print(f"Failed models: {len(failed_models)} (saved to {save_dir}/failed_models.csv)")
else:
    print("No failed models.")
