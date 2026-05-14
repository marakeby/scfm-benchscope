import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _labels_for_stratified_split(labels: pd.Series) -> pd.Series | None:
    """
    sklearn requires every stratum to have at least 2 members. Merge singleton
    strata into one bucket; if stratification is still impossible, return None
    (caller should fall back to uniform sampling).
    """
    s = labels.astype("string")
    vc = s.value_counts()
    if vc.empty:
        return None
    if vc.min() >= 2:
        return s
    rare = vc[vc < 2].index
    merged = s.where(~s.isin(rare), "__stratum_lt2__")
    vc2 = merged.value_counts()
    if vc2.min() >= 2:
        return merged
    return None


def sample_adata(
    adata: ad.AnnData,
    sample_size: int = 2000,
    stratify_by: str = None,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Sample a subset of cells from an AnnData object.

    Args:
        adata (AnnData): The full AnnData object to sample from.
        sample_size (int): Number of cells to sample.
        stratify_by (str, optional): Column name in adata.obs to perform stratified sampling.
        random_state (int): Seed for reproducibility.

    Returns:
        AnnData: A sampled AnnData object.
    """
    n_cells = adata.n_obs
    sample_size = min(sample_size, n_cells)
    rng = np.random.default_rng(random_state)

    # sklearn's train_test_split rejects train_size == n_samples (no hold-out split).
    if sample_size >= n_cells:
        return adata.copy()

    if stratify_by and stratify_by in adata.obs.columns:
        stratify_labels = _labels_for_stratified_split(adata.obs[stratify_by])
        if stratify_labels is not None:
            sample_idx, _ = train_test_split(
                np.arange(n_cells),
                train_size=sample_size,
                stratify=stratify_labels,
                random_state=random_state,
            )
        else:
            sample_idx = rng.choice(n_cells, size=sample_size, replace=False)
    else:
        sample_idx = rng.choice(n_cells, size=sample_size, replace=False)

    return adata[sample_idx].copy()