import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, List
from utils.logs_ import get_logger

logger = get_logger()


def sample_adata(
    adata: ad.AnnData, 
    sample_size: int = 2000, 
    stratify_by: str = None,
    random_state: int = 42
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

    if stratify_by and stratify_by in adata.obs.columns:
        from sklearn.model_selection import train_test_split
        stratify_labels = adata.obs[stratify_by]
        sample_idx, _ = train_test_split(
            np.arange(n_cells),
            train_size=sample_size,
            stratify=stratify_labels,
            random_state=random_state
        )
    else:
        sample_idx = rng.choice(n_cells, size=sample_size, replace=False)

    return adata[sample_idx].copy()


def sample_adata_for_batch_integration(
    adata: ad.AnnData,
    batch_key: str,
    label_key: str,
    sample_size: int = 5000,
    min_cells_per_group: int = 2,
    random_state: int = 42
) -> ad.AnnData:
    """
    Sample a subset of cells optimized for batch integration evaluation.
    
    Stratifies by BOTH batch AND cell type to ensure:
    1. Each batch is proportionally represented
    2. Each cell type within each batch is represented
    3. Sufficient cells per batch-celltype combination
    
    This is the recommended sampling strategy for batch integration benchmarks.
    
    Args:
        adata: The full AnnData object to sample from
        batch_key: Column name in adata.obs for batch labels
        label_key: Column name in adata.obs for cell type labels
        sample_size: Target number of cells to sample
        min_cells_per_group: Minimum cells per batch-celltype group (for stratification)
        random_state: Seed for reproducibility
    
    Returns:
        Sampled AnnData object with proportional batch and cell type representation
    """
    n_cells = adata.n_obs
    
    # If already small enough, return copy
    if n_cells <= sample_size:
        logger.info(f"Dataset has {n_cells} cells, no subsampling needed")
        return adata.copy()
    
    rng = np.random.default_rng(random_state)
    
    # Check if both keys exist
    has_batch = batch_key and batch_key in adata.obs.columns
    has_label = label_key and label_key in adata.obs.columns
    
    if has_batch and has_label:
        # Create combined stratification key: batch_celltype
        combined_key = '_batch_celltype_stratify'
        adata.obs[combined_key] = (
            adata.obs[batch_key].astype(str) + '_' + 
            adata.obs[label_key].astype(str)
        )
        
        # Check if stratification is possible (need >= 2 samples per group for train_test_split)
        group_counts = adata.obs[combined_key].value_counts()
        small_groups = group_counts[group_counts < min_cells_per_group]
        
        if len(small_groups) > 0:
            logger.warning(
                f"Found {len(small_groups)} batch-celltype groups with < {min_cells_per_group} cells. "
                f"Falling back to batch-only stratification."
            )
            # Clean up and fall back to batch-only
            del adata.obs[combined_key]
            stratify_key = batch_key
        else:
            stratify_key = combined_key
            logger.info(f"Stratifying by batch AND cell type ({len(group_counts)} groups)")
    elif has_batch:
        stratify_key = batch_key
        logger.info(f"Stratifying by batch only")
    elif has_label:
        stratify_key = label_key
        logger.info(f"Stratifying by cell type only")
    else:
        stratify_key = None
        logger.info(f"No stratification keys available, using random sampling")
    
    # Perform sampling
    try:
        if stratify_key:
            stratify_labels = adata.obs[stratify_key]
            sample_idx, _ = train_test_split(
                np.arange(n_cells),
                train_size=sample_size,
                stratify=stratify_labels,
                random_state=random_state
            )
        else:
            sample_idx = rng.choice(n_cells, size=sample_size, replace=False)
    except ValueError as e:
        # Stratification failed (e.g., some groups too small)
        logger.warning(f"Stratified sampling failed: {e}. Using random sampling.")
        sample_idx = rng.choice(n_cells, size=sample_size, replace=False)
    
    # Clean up temporary column if created
    if '_batch_celltype_stratify' in adata.obs.columns:
        del adata.obs['_batch_celltype_stratify']
    
    sampled_adata = adata[sample_idx].copy()
    
    # Log sampling stats
    if has_batch:
        orig_batches = adata.obs[batch_key].value_counts()
        new_batches = sampled_adata.obs[batch_key].value_counts()
        logger.info(f"Subsampled {n_cells} -> {len(sample_idx)} cells")
        logger.info(f"  Original batch distribution: {dict(orig_batches)}")
        logger.info(f"  Sampled batch distribution: {dict(new_batches)}")
    
    return sampled_adata