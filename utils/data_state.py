"""
Centralized data state management for scFM_eval.

This module provides a clean, standardized way to track and manage the 
normalization/transformation state of AnnData objects, eliminating the
need for scattered detection logic and flag passing.

Key Concepts:
- DataState: Enum representing the transformation state of expression data
- Each model declares what state it expects via EXPECTED_STATE constant
- Detection logic is centralized in get_data_state()
- Models can query needs_transform() to check if transformation is needed

Usage:
    from utils.data_state import DataState, get_data_state, set_data_state, needs_transform
    
    # Check current state
    state = get_data_state(adata)
    
    # Check if model needs to apply log1p
    if needs_transform(adata, DataState.LOG1P):
        # Apply log1p during batch processing
        features_log = torch.log1p(features)
    else:
        # Data is already log1p transformed
        features_log = features
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


class DataState(str, Enum):
    """
    Standard data states for AnnData expression matrices.
    
    Using str as base allows easy serialization to/from adata.uns.
    
    States:
        RAW: Raw counts (integers, potentially large values)
        NORMALIZED: TP10K normalized counts, NOT log transformed
        LOG1P: TP10K normalized + log1p transformed
        UNKNOWN: State cannot be determined
    """
    RAW = "raw"
    NORMALIZED = "normalized"  # TP10K, no log
    LOG1P = "log1p"            # TP10K + log1p
    UNKNOWN = "unknown"


# Standard key for storing data state in adata.uns
DATA_STATE_KEY = "data_state"


def get_data_state(
    adata: "ad.AnnData",
    use_raw: bool = False,
    sample_size: int = 1000,
) -> DataState:
    """
    Get the current normalization state of AnnData.
    
    Checks in order:
    1. Explicit flag in adata.uns['data_state'] (preferred; can be set from YAML preprocessing.data_state)
    2. Scanpy convention adata.uns['log1p']
    3. Legacy preprocessing_method flag (for backwards compatibility)
    4. Distribution-based heuristics (fallback)
    
    Args:
        adata: AnnData object to check
        use_raw: If True, check adata.raw instead of adata.X
        sample_size: Number of cells to sample for heuristic detection
        
    Returns:
        DataState enum value
    """
    # Select data source
    if use_raw and adata.raw is not None:
        data_source = adata.raw
        uns_source = adata.uns  # raw doesn't have its own uns
    else:
        data_source = adata
        uns_source = adata.uns if hasattr(adata, 'uns') else {}
    
    # Check 1: Explicit data_state flag (preferred method)
    if DATA_STATE_KEY in uns_source:
        try:
            return DataState(uns_source[DATA_STATE_KEY])
        except ValueError:
            logger.warning(f"Invalid data_state value: {uns_source[DATA_STATE_KEY]}")
    
    # Check 2: Scanpy log1p convention
    if 'log1p' in uns_source:
        return DataState.LOG1P
    
    # Check 3: Legacy preprocessing_method flag (backwards compatibility)
    if 'preprocessing_method' in uns_source:
        method = str(uns_source['preprocessing_method']).lower()
        # Explicit log1p mentioned without 'no_log' or 'skip_log'
        has_log = 'log1p' in method or 'log' in method
        log_skipped = 'no_log' in method or 'skip_log' in method
        
        if has_log and not log_skipped:
            return DataState.LOG1P
        elif uns_source.get('preprocessed', False):
            # Preprocessed but not log1p
            return DataState.NORMALIZED
    
    # Check 4: Distribution-based heuristics
    return _detect_from_distribution(data_source, sample_size)


def _detect_from_distribution(
    data_source: "ad.AnnData",
    sample_size: int = 1000,
) -> DataState:
    """
    Detect data state from value distribution.
    
    Heuristics:
    - Log1p data: max value typically < 25 (log1p of large counts can reach ~22)
    - Normalized (no log): max < 100, values are floats
    - Raw counts: large integers (max > 100)
    
    Args:
        data_source: AnnData or AnnData.raw to sample from
        sample_size: Number of cells to sample
        
    Returns:
        DataState enum value
    """
    from scipy.sparse import issparse
    
    try:
        # Sample data for efficiency
        n_cells = min(sample_size, data_source.n_obs)
        sample = data_source.X[:n_cells]
        
        # Convert sparse to dense if needed
        if issparse(sample):
            sample = sample.toarray()
        
        # Calculate statistics
        max_val = np.max(sample)
        
        # Log1p data: max typically < 25 (covers DepMap, TCGA log1p ~21, etc.)
        if max_val < 25:
            logger.debug(f"Detected LOG1P state (max={max_val:.2f} < 25)")
            return DataState.LOG1P
        
        # Check for integer values (raw counts)
        sample_subset = sample[:min(1000, n_cells)]
        is_integer = np.allclose(sample_subset, np.round(sample_subset), rtol=1e-5)
        
        # Raw counts: large integers
        if max_val > 100 and is_integer:
            logger.debug(f"Detected RAW state (max={max_val:.2f} > 100, integers)")
            return DataState.RAW
        
        # Normalized: floats with moderate range
        if max_val < 100:
            logger.debug(f"Detected NORMALIZED state (max={max_val:.2f} < 100)")
            return DataState.NORMALIZED
        
        # Ambiguous - likely raw
        logger.debug(f"Uncertain state, assuming RAW (max={max_val:.2f})")
        return DataState.RAW
        
    except Exception as e:
        logger.warning(f"Error detecting data state: {e}. Returning UNKNOWN.")
        return DataState.UNKNOWN


def set_data_state(
    adata: "ad.AnnData",
    state: DataState,
    *,
    mark_preprocessed: bool = True,
) -> None:
    """
    Set the data state explicitly.
    
    This is the recommended way to mark data state after preprocessing.
    
    Args:
        adata: AnnData object to modify
        state: DataState to set
        mark_preprocessed: Also set preprocessed=True for backwards compatibility
    """
    if not hasattr(adata, 'uns') or adata.uns is None:
        adata.uns = {}
    
    adata.uns[DATA_STATE_KEY] = state.value
    
    if mark_preprocessed and state != DataState.RAW:
        adata.uns['preprocessed'] = True
        # Also set legacy flag for backwards compatibility
        adata.uns['preprocessing_method'] = state.value
    
    logger.debug(f"Set data state to {state.value}")


def needs_transform(
    adata: "ad.AnnData",
    target_state: DataState,
    use_raw: bool = False,
) -> bool:
    """
    Check if data needs transformation to reach target state.
    
    This is the key function models should use to decide whether
    to apply transformations.
    
    Args:
        adata: AnnData object
        target_state: The state the model expects
        use_raw: Whether to check adata.raw instead of adata.X
        
    Returns:
        True if transformation is needed, False if data is already in target state
        
    Example:
        # In STACK model (expects NORMALIZED, applies log1p internally)
        if needs_transform(adata, DataState.LOG1P):
            features_log = torch.log1p(features)
        else:
            features_log = features  # Already log1p
    """
    current = get_data_state(adata, use_raw=use_raw)
    return current != target_state


def ensure_state(
    adata: "ad.AnnData",
    target_state: DataState,
    target_sum: float = 1e4,
    inplace: bool = True,
) -> "ad.AnnData":
    """
    Ensure adata is in the target state, transforming if needed.
    
    This function applies the necessary transformations to reach the target state.
    Use sparingly - prefer on-the-fly transformation in model forward passes.
    
    Args:
        adata: AnnData object
        target_state: Target DataState
        target_sum: Target sum for normalization (default 10000)
        inplace: If True, modify adata in place; if False, return a copy
        
    Returns:
        AnnData in the target state
        
    Raises:
        ValueError: If transformation path is not supported
    """
    import scanpy as sc
    
    current = get_data_state(adata)
    
    if current == target_state:
        return adata
    
    if not inplace:
        adata = adata.copy()
    
    # Define valid transformation paths
    if current == DataState.RAW:
        if target_state == DataState.NORMALIZED:
            sc.pp.normalize_total(adata, target_sum=target_sum)
            set_data_state(adata, DataState.NORMALIZED)
        elif target_state == DataState.LOG1P:
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
            set_data_state(adata, DataState.LOG1P)
        else:
            raise ValueError(f"Cannot transform from {current} to {target_state}")
            
    elif current == DataState.NORMALIZED:
        if target_state == DataState.LOG1P:
            sc.pp.log1p(adata)
            set_data_state(adata, DataState.LOG1P)
        elif target_state == DataState.RAW:
            raise ValueError("Cannot reverse normalization (NORMALIZED -> RAW)")
        else:
            raise ValueError(f"Cannot transform from {current} to {target_state}")
            
    elif current == DataState.LOG1P:
        # Generally cannot reverse log1p without original data
        if target_state in (DataState.RAW, DataState.NORMALIZED):
            raise ValueError(
                f"Cannot reverse log1p transformation ({current} -> {target_state}). "
                "Consider using adata.raw if original data is preserved."
            )
    
    elif current == DataState.UNKNOWN:
        logger.warning(
            "Data state is UNKNOWN. Applying transformation but results may be incorrect."
        )
        # Attempt the transformation anyway
        if target_state == DataState.LOG1P:
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
            set_data_state(adata, DataState.LOG1P)
        elif target_state == DataState.NORMALIZED:
            sc.pp.normalize_total(adata, target_sum=target_sum)
            set_data_state(adata, DataState.NORMALIZED)
    
    logger.info(f"Transformed data from {current.value} to {target_state.value}")
    return adata


def get_state_summary(adata: "ad.AnnData", use_raw: bool = False) -> str:
    """
    Get a human-readable summary of the data state.
    
    Useful for logging and debugging.
    
    Args:
        adata: AnnData object
        use_raw: Whether to check adata.raw
        
    Returns:
        Summary string
    """
    from scipy.sparse import issparse
    
    state = get_data_state(adata, use_raw=use_raw)
    data_source = adata.raw if (use_raw and adata.raw is not None) else adata
    
    # Sample statistics
    try:
        sample = data_source.X[:min(1000, data_source.n_obs)]
        if issparse(sample):
            sample = sample.toarray()
        max_val = np.max(sample)
        mean_val = np.mean(sample)
        zero_pct = (sample == 0).sum() / sample.size * 100
    except Exception:
        max_val = mean_val = zero_pct = float('nan')
    
    return (
        f"DataState: {state.value} | "
        f"Shape: {data_source.shape} | "
        f"Max: {max_val:.2f}, Mean: {mean_val:.3f}, Zeros: {zero_pct:.1f}%"
    )


# Convenience aliases
is_log1p = lambda adata: get_data_state(adata) == DataState.LOG1P
is_normalized = lambda adata: get_data_state(adata) in (DataState.NORMALIZED, DataState.LOG1P)
is_raw = lambda adata: get_data_state(adata) == DataState.RAW


__all__ = [
    "DataState",
    "DATA_STATE_KEY",
    "get_data_state",
    "set_data_state",
    "needs_transform",
    "ensure_state",
    "get_state_summary",
    "is_log1p",
    "is_normalized",
    "is_raw",
]
