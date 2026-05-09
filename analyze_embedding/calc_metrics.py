import anndata as ad
import pandas as pd
import scanpy as sc
import os
import random
import numpy as np
import scipy.sparse as sp
import time
import sys
from pathlib import Path


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
    Repeatedly subsample `adata`, evaluate embedding metrics, and return per-run results.

    Returns a DataFrame where each row is one run.
    """
    mid_results: list[dict] = []
    for i in range(n_runs):
        subsampled_adata = sample_adata(
            adata,
            sample_size=sample_size,
            stratify_by=stratify_by,
            random_state=base_seed + i,
        )
        results = EmbeddingEvaluator(
            subsampled_adata,
            embedding_key,
            save_dir=save_dir,
            save_id=f"{embedding_key}_{i}",
            n_jobs=n_jobs,
            random_state=base_seed + i,
        )
        results_dict = results.evaluate()
        print(i)
        mid_results.append(results_dict)

    return pd.DataFrame(mid_results)


def get_embd_key(model_name):

    key = None
    if 'pca' in model_name :
        key = f'X_pca'    
        
    if 'hvg' in model_name :
        key = f'X_hvg'    
        
    if 'scvi' in model_name :
        key = f'X_scVI'
    if 'scgpt' in model_name :
        key = f'X_scGPT'
    if 'geneformer' in model_name :
        key = f'X_geneformer'
    if 'gf' in model_name :
        key = f'X_geneformer'
        
    if 'scfoundation' in model_name:
        key = f'X_scfoundation'
    
    if 'cellplm' in model_name:
        key = f'X_CellPLM'
    
    if 'scimilarity' in model_name:
        key = f'X_scimilarity'
        
    if 'nicheformer' in model_name:
        key = f'X_nicheformer'

    if 'scconcept' in model_name:
        key = f'X_scconcept'

    if 'state' in model_name:
        key = f'X_state'
        
    if key is None:
        print ('no key found')
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

    model_name = parts[1].strip("_")
    return model_name

# Ensure `scFM_eval/` is on sys.path so `evaluation.*` imports work
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.eval import EmbeddingEvaluator
from utils.sampling import sample_adata

base_dir = '/home/haitham/mnt/__output_v3/exp'
from os.path import join

search_text = 'cell_type'
folders= [] 
# Use os.walk to search through all subdirectories
for root, dirs, files in os.walk(base_dir):
    for directory in dirs:
        # print(directory)
        if search_text in directory:
            # Print the full path of matching folders
            print(os.path.join(root, directory))
            folders.append(os.path.join(root, directory))


sample_size = 10000
n_runs = 10
base_seed = 42
n_jobs = -1
save_dir = './results'
results_mean= []
results_std= []
results_median = []
all_results= {}
failed_models: list[dict] = []

for f in folders:
    model_name = None
    embedding_key = None
    try:
        model_name = extract_model_name(f, "brca_cell_type")
        embedding_key = get_embd_key(model_name)
        print(embedding_key)

        data_fname = join(f, "data.h5ad")
        embs = ad.read_h5ad(data_fname)
        print(embs)
        assert (
            embedding_key in embs.obsm.keys()
        ), f"embedding key {embedding_key} not found in {embs.obsm.keys()}"
        print(f"embedding key {embedding_key} found in {embs.obsm.keys()}")

        if sp.issparse(embs.obsm[embedding_key]):
            embs.obsm[embedding_key] = embs.obsm[embedding_key].toarray()
            embs.obsm[embedding_key] = np.asarray(embs.obsm[embedding_key])

        time_start = time.perf_counter()

        # Prefer combined stratification; falls back handled in `run_subsampled_eval` if needed.
        if "batch" in embs.obs.columns and "label" in embs.obs.columns:
            embs.obs["batch_label"] = (
                embs.obs["batch"].astype(str) + "|" + embs.obs["label"].astype(str)
            )
            stratify_key = "batch_label"
        else:
            stratify_key = "batch"

        df = run_subsampled_eval(
            embs,
            embedding_key=embedding_key,
            n_runs=n_runs,
            sample_size=sample_size,
            stratify_by=stratify_key,
            base_seed=base_seed,
            save_dir=save_dir,
            n_jobs=n_jobs,
        )
        time_end = time.perf_counter()

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
        print(f"save_dir: {save_dir}")
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
results_mean_df = pd.concat(results_mean, axis=1).T.set_index("model")
results_std_df = pd.concat(results_std, axis=1).T.set_index("model")
results_median_df = pd.concat(results_median, axis=1).T.set_index("model")

# `all_results` maps model_name -> per-run DataFrame.
# Concatenate into one long DataFrame with a `model` column.
all_results_df = (
    pd.concat(all_results, names=["model", "run"])
    .reset_index(level=0)
)
results_mean_df.to_csv(f'{save_dir}/results_mean.csv')
results_std_df.to_csv(f'{save_dir}/results_std.csv')
results_median_df.to_csv(f"{save_dir}/results_median.csv")
all_results_df.to_csv(f'{save_dir}/all_results.csv')

if failed_models:
    failed_df = pd.DataFrame(failed_models)
    failed_df.to_csv(f"{save_dir}/failed_models.csv", index=False)
    print(f"Failed models: {len(failed_models)} (saved to {save_dir}/failed_models.csv)")
else:
    print("No failed models.")