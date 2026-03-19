"""
data_loader.py

Data loading utilities for single-cell data, supporting CSV and H5AD formats with preprocessing and filtering.
"""

import os
import json
import logging
import re

import pandas as pd
import anndata as ad
import scanpy as sc

from utils.logs_ import get_logger
from utils.data_state import DataState, get_data_state, set_data_state, get_state_summary
from setup_path import BASE_PATH, DATA_PATH
from os.path import join, isabs
import numpy as np
import h5py
from typing import Dict, Any, Tuple
from scipy.sparse import issparse
import scipy.sparse as sp
from scipy.stats import median_abs_deviation


class DataLoader:
    """Base class for loading and preprocessing single-cell data."""

    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for data loading.
        """
        self.params = params
        # Resolve path: if relative, join with DATA_PATH; if absolute, use as-is
        path = params['path']
        if isabs(path):
            self.path = path
        else:
            self.path = join(DATA_PATH, path)
        self.dataset_name = os.path.basename(self.path).split(".")[0]
        self.layer = params['layer_name']
        self.label_key = params['label_key']
        self.batch_key = params['batch_key']
        self.train_test_split = params['train_test_split']
        self.cv_splits = params['cv_splits']
        self.log = get_logger()
        self.adata = None

    @staticmethod
    def validate_config(params):
        """Validate that required parameters are present.

        Args:
            params (dict): Configuration parameters.
        Raises:
            AssertionError: If required parameters are missing.
        """
        assert 'path' in params, "Missing required parameter: 'path'"

    def prepare_data(self, process_dir):
        """Prepare data for downstream analysis. To be implemented in subclasses.

        Args:
            process_dir (str): Directory for processing outputs.
        """
        pass

    def _extract_h5ad_metadata(self) -> Dict[str, Any]:
        """Extract metadata from H5AD file using h5py for efficient partial reading.
        
        Extracts metadata for obs, var, uns, obsm, varm, layers, obsp, and varp
        without loading the full data matrices.
        
        Returns:
            Dict[str, Any]: Metadata dictionary containing:
                - obs: DataFrame with cell metadata
                - var: DataFrame with gene metadata
                - uns: Unstructured metadata dictionary
                - obsm: Dictionary with keys and shapes of observation multidimensional arrays
                - varm: Dictionary with keys and shapes of variable multidimensional arrays
                - layers: Dictionary with keys and shapes of layer data
                - obsp: Dictionary with keys and shapes of observation pairwise arrays
                - varp: Dictionary with keys and shapes of variable pairwise arrays
                - shape: Tuple of (n_obs, n_vars)
        """
        self.log.info(f"Extracting metadata from H5AD file: {self.path}")
        
        metadata: Dict[str, Any] = {
            'file_path': self.path,
            'obs': None,
            'var': None,
            'uns': None,
            'obsm': {},
            'varm': {},
            'layers': {},
            'obsp': {},
            'varp': {},
            'shape': None,
            'obs_columns': [],
            'var_columns': []
        }
        
        try:
            with h5py.File(self.path, 'r') as f:
                from anndata.io import read_elem
                
                # Read obs (cell metadata) - typically much smaller than X
                if 'obs' in f:
                    obs_data = read_elem(f['obs'])
                    if isinstance(obs_data, pd.DataFrame):
                        metadata['obs'] = obs_data
                        metadata['obs_columns'] = list(obs_data.columns)
                    else:
                        self.log.warning("obs is not a DataFrame, skipping")
                
                # Read var (gene metadata)
                if 'var' in f:
                    var_data = read_elem(f['var'])
                    if isinstance(var_data, pd.DataFrame):
                        metadata['var'] = var_data
                        metadata['var_columns'] = list(var_data.columns)
                    else:
                        self.log.warning("var is not a DataFrame, skipping")
                
                # Read uns (unstructured metadata)
                if 'uns' in f:
                    metadata['uns'] = read_elem(f['uns'])
                
                # Read obsm (observations multidimensional arrays)
                if 'obsm' in f:
                    for key in f['obsm'].keys():
                        try:
                            arr = f['obsm'][key]
                            if hasattr(arr, 'shape'):
                                metadata['obsm'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading obsm['{key}']: {e}")
                
                # Read varm (variables multidimensional arrays)
                if 'varm' in f:
                    for key in f['varm'].keys():
                        try:
                            arr = f['varm'][key]
                            if hasattr(arr, 'shape'):
                                metadata['varm'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading varm['{key}']: {e}")
                
                # Read layers (layered data)
                if 'layers' in f:
                    for key in f['layers'].keys():
                        try:
                            arr = f['layers'][key]
                            if hasattr(arr, 'shape'):
                                metadata['layers'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading layers['{key}']: {e}")
                
                # Read obsp (observations pairwise arrays)
                if 'obsp' in f:
                    for key in f['obsp'].keys():
                        try:
                            arr = f['obsp'][key]
                            if hasattr(arr, 'shape'):
                                metadata['obsp'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading obsp['{key}']: {e}")
                
                # Read varp (variables pairwise arrays)
                if 'varp' in f:
                    for key in f['varp'].keys():
                        try:
                            arr = f['varp'][key]
                            if hasattr(arr, 'shape'):
                                metadata['varp'][key] = {
                                    'shape': arr.shape,
                                    'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else None
                                }
                        except Exception as e:
                            self.log.warning(f"Error reading varp['{key}']: {e}")
                
                # Get shape information without loading X
                if 'obs' in f and 'var' in f:
                    n_obs = f['obs'].attrs.get('_index', None)
                    if n_obs is None:
                        # Fallback: try to get length from obs DataFrame
                        if metadata['obs'] is not None:
                            n_obs = len(metadata['obs'])
                        else:
                            # Last resort: check shape attribute if available
                            n_obs = f.attrs.get('n_obs', None)
                    
                    n_vars = f['var'].attrs.get('_index', None)
                    if n_vars is None:
                        if metadata['var'] is not None:
                            n_vars = len(metadata['var'])
                        else:
                            n_vars = f.attrs.get('n_vars', None)
                    
                    if n_obs is not None and n_vars is not None:
                        metadata['shape'] = (n_obs, n_vars)
                    elif 'shape' in f.attrs:
                        metadata['shape'] = tuple(f.attrs['shape'])
                
                # Check preprocessing status
                is_preprocessed = False
                preprocessing_method = None
                if metadata['uns'] and isinstance(metadata['uns'], dict):
                    is_preprocessed = metadata['uns'].get('preprocessed', False)
                    preprocessing_method = metadata['uns'].get('preprocessing_method', None)
                
                # Log summary of extracted metadata
                log_parts = [
                    f"Shape: {metadata['shape']}",
                    f"Obs columns: {len(metadata['obs_columns'])}",
                    f"Var columns: {len(metadata['var_columns'])}"
                ]
                
                # Add preprocessing status
                if is_preprocessed:
                    method_str = f" ({preprocessing_method})" if preprocessing_method else ""
                    log_parts.append(f"Preprocessed: True{method_str}")
                else:
                    log_parts.append("Preprocessed: False")
                
                if metadata['obsm']:
                    log_parts.append(f"obsm keys: {list(metadata['obsm'].keys())}")
                if metadata['varm']:
                    log_parts.append(f"varm keys: {list(metadata['varm'].keys())}")
                if metadata['layers']:
                    log_parts.append(f"layers keys: {list(metadata['layers'].keys())}")
                if metadata['obsp']:
                    log_parts.append(f"obsp keys: {list(metadata['obsp'].keys())}")
                if metadata['varp']:
                    log_parts.append(f"varp keys: {list(metadata['varp'].keys())}")
                
                self.log.info(f"Metadata extracted. {', '.join(log_parts)}")
                
        except Exception as e:
            self.log.error(f"Error extracting metadata from H5AD file: {e}")
            raise
        
        return metadata

    def load(self):
        """Load data and return AnnData object. To be implemented in subclasses.

        Returns:
            ad.AnnData: Loaded data object.
        """
        print(f"Loading data from {self.path}")
        adata = ad.AnnData()
        self.adata = adata
        self.log.info(f'Data Loaded, {self.adata.X.shape}')
        self.log.info(
            f'min {np.min(self.adata.X)}, max {np.max(self.adata.X)}')
        return adata

    def extract_dataset_metadata(self) -> Dict[str, Any]:
        """Extract metadata from dataset without loading full data.
        
        Returns:
            Dict[str, Any]: Metadata dictionary with obsm keys, preprocessing status, and other info.
        """
        if hasattr(self, 'path') and self.path.endswith('.h5ad'):
            return self._extract_h5ad_metadata()
        else:
            # For non-H5AD files, return minimal metadata
            return {'obsm': {}, 'obs_columns': [], 'var_columns': [], 'uns': {}}
    
    def is_preprocessed(self) -> bool:
        """Check if dataset is already preprocessed (normalized + log1p).
        
        Returns:
            bool: True if dataset is marked as preprocessed, False otherwise.
        """
        metadata = self.extract_dataset_metadata()
        if metadata.get('uns') and isinstance(metadata['uns'], dict):
            return metadata['uns'].get('preprocessed', False)
        return False

    def _filter(self, adata, filter_dict):
        """Filter AnnData object based on filter_dict.

        Args:
            adata (ad.AnnData): AnnData object to filter.
            filter_dict (dict): Dictionary of {column: [values]} to filter by.
        Returns:
            ad.AnnData: Filtered AnnData object.
        """
        # Build combined mask to filter only once (avoids multiple slow view creations)
        mask = np.ones(adata.n_obs, dtype=bool)
        for col, values in filter_dict.items():
            mask &= adata.obs[col].isin(values).values
        
        # Return view to avoid expensive copy of large X matrix
        return adata[mask]

    def _translate_split_ids(self):
        """Translate split IDs to match data format.

        Handles mismatch where CV splits use 'Post' but data has 'On' in donor_id_pre_post.
        Translates 'BIOKEY_X_Post' -> 'BIOKEY_X_On' in split IDs to match actual data.
        """
        if not hasattr(self, 'cv_split_dict') or not hasattr(self, 'train_test_split_dict'):
            return

        # Quick check: only translate if id_column is donor_id_pre_post and splits contain '_Post'
        needs_translation = False
        if self.cv_split_dict:
            id_column = self.cv_split_dict.get('id_column', '')
            if id_column == 'donor_id_pre_post':
                # Quick check: sample first few split IDs to see if they have '_Post'
                fold1 = self.cv_split_dict.get('fold_1', {})
                sample_ids = fold1.get('train_ids', [])[
                    :3] if fold1.get('train_ids') else []
                if sample_ids and any('_Post' in str(sid) for sid in sample_ids):
                    # Check data format quickly (sample only, don't load all unique values)
                    if 'donor_id_pre_post' in self.adata.obs.columns:
                        # Sample a small subset to check format (much faster than .unique())
                        sample_data = self.adata.obs['donor_id_pre_post'].head(
                            1000).astype(str)
                        has_on_in_data = any('_On' in str(dv)
                                             for dv in sample_data.values)
                        if has_on_in_data:
                            needs_translation = True
                            self.log.info(
                                "Translating split IDs: '_Post' -> '_On' to match data format"
                            )

        if needs_translation:
            # Translate train_test_split
            if self.train_test_split_dict:
                split_dict = self.train_test_split_dict.get(
                    'train_test_split', {})
                if 'train_ids' in split_dict:
                    # Optimized: only translate IDs that contain '_Post' (in-place)
                    train_ids = split_dict['train_ids']
                    for idx, sid in enumerate(train_ids):
                        if '_Post' in str(sid):
                            train_ids[idx] = str(sid).replace('_Post', '_On')
                if 'test_ids' in split_dict:
                    # Optimized: only translate IDs that contain '_Post' (in-place)
                    test_ids = split_dict['test_ids']
                    for idx, sid in enumerate(test_ids):
                        if '_Post' in str(sid):
                            test_ids[idx] = str(sid).replace('_Post', '_On')

            # Translate CV splits
            if self.cv_split_dict:
                n_splits = self.cv_split_dict.get('n_splits', 0)
                for i in range(1, n_splits + 1):
                    fold_key = f'fold_{i}'
                    if fold_key in self.cv_split_dict:
                        fold = self.cv_split_dict[fold_key]
                        if 'train_ids' in fold:
                            # Optimized: only translate IDs that contain '_Post' (in-place)
                            train_ids = fold['train_ids']
                            for idx, sid in enumerate(train_ids):
                                if '_Post' in str(sid):
                                    train_ids[idx] = str(
                                        sid).replace('_Post', '_On')
                        if 'test_ids' in fold:
                            # Optimized: only translate IDs that contain '_Post' (in-place)
                            test_ids = fold['test_ids']
                            for idx, sid in enumerate(test_ids):
                                if '_Post' in str(sid):
                                    test_ids[idx] = str(
                                        sid).replace('_Post', '_On')

    def qc(
        self,
        min_genes: int = None,
        min_cells: int = None,
        remove_outliers: bool = False,
        mad_nmads: Dict[str, float] = None,
        pct_counts_mt_max: float = 8.0,
        use_top20: bool = False,
        **kwargs,
    ):
        """Apply quality control filters to the data.

        Two types of QC (both optional via config):

        1. **Threshold-based** (min_genes, min_cells): hard cutoffs.
           - filter_cells: drop cells with fewer than min_genes genes expressed.
           - filter_genes: drop genes expressed in fewer than min_cells cells.

        2. **MAD-based outlier removal** (remove_outliers=True): distribution-based.
           - Annotates mt/ribo/hb, computes QC metrics (total_counts, n_genes_by_counts, pct_mt).
           - Removes cells that are more than N MADs from the median on each metric
             (and optionally pct_mt > pct_counts_mt_max). Dataset-adaptive.

        Args:
            min_genes: Minimum number of genes per cell (None = skip cell filter).
            min_cells: Minimum number of cells per gene (None = skip gene filter).
            remove_outliers: If True, apply MAD-based outlier removal after threshold QC.
            mad_nmads: Optional dict of metric -> n MADs. Keys: total_counts, genes_by_counts,
                pct_counts_mt. Defaults: 5, 5, 3. Set to 0 to skip that metric.
            pct_counts_mt_max: Also remove cells with pct_counts_mt > this (percent). Default 8.
            use_top20: If True, also use pct_counts_in_top_20_genes with 5 MADs.
            **kwargs: Ignored (for forward compatibility with extra YAML keys).
        """
        self.log.info(f"obs shape before QC: {self.adata.X.shape}")

        # ----- 1. Threshold-based QC (min_genes / min_cells) -----
        if min_genes is not None or min_cells is not None:
            sc.pp.calculate_qc_metrics(
                self.adata, percent_top=None, log1p=False, inplace=True
            )
            if min_genes is not None:
                sc.pp.filter_cells(self.adata, min_genes=int(min_genes))
                self.log.info(f"After filter_cells(min_genes={min_genes}): {self.adata.shape}")
            if min_cells is not None:
                sc.pp.filter_genes(self.adata, min_cells=int(min_cells))
                self.log.info(f"After filter_genes(min_cells={min_cells}): {self.adata.shape}")

        # ----- 2. MAD-based outlier removal (optional) -----
        if remove_outliers:
            self._qc_mad_outliers(
                mad_nmads=mad_nmads or {},
                pct_counts_mt_max=pct_counts_mt_max,
                use_top20=use_top20,
            )

        self.log.info(f"obs shape after QC: {self.adata.X.shape}")

    def _qc_mad_outliers(
        self,
        mad_nmads: Dict[str, float] = None,
        pct_counts_mt_max: float = 8.0,
        use_top20: bool = False,
    ) -> None:
        """Remove cells that are MAD-based outliers on QC metrics (mt, total counts, etc.)."""
        adata = self.adata
        mad_nmads = mad_nmads or {}

        # Default n MADs (match notebook PCABaselineEvaluator)
        nmad_total = mad_nmads.get("total_counts", 5)
        nmad_genes = mad_nmads.get("genes_by_counts", 5)
        nmad_mt = mad_nmads.get("pct_counts_mt", 3)

        # Annotate gene types if not already
        if "mt" not in adata.var.columns:
            adata.var["mt"] = adata.var_names.str.startswith("MT-")
            adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
            adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]", regex=True)

        # Ensure we have log1p QC metrics (MAD uses log-scale for counts/genes)
        if "log1p_total_counts" not in adata.obs.columns:
            sc.pp.calculate_qc_metrics(
                adata,
                qc_vars=["mt", "ribo", "hb"],
                inplace=True,
                percent_top=[20] if use_top20 else None,
                log1p=True,
            )

        log_umis = np.asarray(adata.obs["log1p_total_counts"].values, dtype=float)
        log_genes = np.asarray(adata.obs["log1p_n_genes_by_counts"].values, dtype=float)
        pct_mt = np.asarray(adata.obs["pct_counts_mt"].values, dtype=float)

        def mad_outlier(x: np.ndarray, nmads: float) -> np.ndarray:
            if nmads <= 0:
                return np.zeros_like(x, dtype=bool)
            med = np.median(x)
            mad = median_abs_deviation(x)
            if mad == 0:
                return np.zeros_like(x, dtype=bool)
            return (x < med - nmads * mad) | (x > med + nmads * mad)

        umi_outliers = mad_outlier(log_umis, nmad_total)
        gene_outliers = mad_outlier(log_genes, nmad_genes)
        mt_outliers = mad_outlier(pct_mt, nmad_mt) | (pct_mt > pct_counts_mt_max)

        if use_top20 and "pct_counts_in_top_20_genes" in adata.obs.columns:
            top20 = np.asarray(adata.obs["pct_counts_in_top_20_genes"].values, dtype=float)
            top20_outliers = mad_outlier(top20, mad_nmads.get("pct_counts_in_top_20_genes", 5))
        else:
            top20_outliers = np.zeros(adata.n_obs, dtype=bool)

        qc_fail = umi_outliers | gene_outliers | mt_outliers | top20_outliers
        n_remove = int(qc_fail.sum())
        self.adata = adata[~qc_fail].copy()
        self.log.info(
            f"MAD-based outlier removal: removed {n_remove} cells "
            f"(total_counts={nmad_total}, genes_by_counts={nmad_genes}, pct_mt={nmad_mt}, "
            f"pct_mt_max={pct_counts_mt_max}). Shape after: {self.adata.shape}"
        )

    def _check_if_normalized(self, sample_size: int = 10000) -> Tuple[bool, bool]:
        """
        Check if data is already normalized/log1p transformed.
        
        Uses the centralized data_state module for detection.
        
        Args:
            sample_size: Number of cells to sample for distribution check.
            
        Returns:
            tuple: (is_normalized, is_log1p) - whether data appears normalized and/or log1p transformed.
        """
        # Use centralized detection
        state = get_data_state(self.adata, sample_size=sample_size)
        
        # Log the summary
        self.log.info(f"Data state detection: {get_state_summary(self.adata)}")
        
        # Convert to legacy tuple format for backwards compatibility
        is_log1p = (state == DataState.LOG1P)
        is_normalized = (state in (DataState.NORMALIZED, DataState.LOG1P))
        
        return is_normalized, is_log1p

    def plot_expression_distribution(self, save_dir, title_prefix="", sample_size=10000):
        """
        Plot distribution of expression values (.X) before and/or after normalization.
        
        Args:
            save_dir: Directory to save the plot
            title_prefix: Prefix for plot title
            sample_size: Number of cells to sample for plotting
        """
        from pathlib import Path
        from matplotlib import pyplot as plt
        from scipy.sparse import issparse
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample data
        n_cells = min(sample_size, self.adata.n_obs)
        if issparse(self.adata.X):
            sample = self.adata.X[:n_cells].toarray()
        else:
            sample = self.adata.X[:n_cells]
        
        # Flatten for histogram
        sample_flat = sample.flatten()
        
        # Remove zeros for better visualization (log scale)
        sample_nonzero = sample_flat[sample_flat > 0]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Histogram (linear scale, including zeros)
        axes[0].hist(sample_flat, bins=100, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Expression Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{title_prefix}Distribution (all values)')
        axes[0].axvline(np.mean(sample_flat), color='r', linestyle='--', label=f'Mean: {np.mean(sample_flat):.3f}')
        axes[0].axvline(np.median(sample_flat), color='g', linestyle='--', label=f'Median: {np.median(sample_flat):.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Histogram (log scale, non-zero values only)
        if len(sample_nonzero) > 0:
            axes[1].hist(sample_nonzero, bins=100, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Expression Value (log scale)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'{title_prefix}Distribution (non-zero values, log scale)')
            axes[1].set_xscale('log')
            axes[1].axvline(np.mean(sample_nonzero), color='r', linestyle='--', label=f'Mean: {np.mean(sample_nonzero):.3f}')
            axes[1].axvline(np.median(sample_nonzero), color='g', linestyle='--', label=f'Median: {np.median(sample_nonzero):.3f}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No non-zero values', ha='center', va='center')
            axes[1].set_title(f'{title_prefix}Distribution (non-zero values)')
        
        # Add statistics text
        stats_text = (
            f'Shape: {self.adata.shape}\n'
            f'Max: {np.max(sample_flat):.3f}\n'
            f'Mean: {np.mean(sample_flat):.3f}\n'
            f'Median: {np.median(sample_flat):.3f}\n'
            f'Zeros: {np.sum(sample_flat == 0) / len(sample_flat) * 100:.1f}%'
        )
        fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        # Clean filename: remove spaces, convert to lowercase, replace special chars
        filename_prefix = title_prefix.lower().replace(" ", "_").replace("-", "_").strip("_")
        plot_path = save_dir / f'{filename_prefix}expression_distribution.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.log.info(f'Saved expression distribution plot to {plot_path}')

    def scale(self, normalize, target_sum, apply_log1p, method=None, plots_dir=None, skip_distribution_check=False):
        """Normalize and/or log-transform the data.
        
        Args:
            normalize (bool): Whether to normalize total counts per cell.
            target_sum (float): Target sum for normalization.
            apply_log1p (bool): Whether to apply log1p transformation.
            method (str, optional): Embedding method name (e.g., 'scimilarity', 'scconcept').
                                   Used to apply method-specific preprocessing.
            plots_dir (str, optional): Directory to save distribution plots.
            skip_distribution_check (bool): If True, skip the distribution-based normalization check.
                                           Use when caller has already performed the check.
        """
        self.normalize = normalize
        self.target_sum = target_sum
        self.apply_log1p = apply_log1p
        self.preprocessing_method = method

        # Check if preprocessing is already done using centralized state detection
        current_state = get_data_state(self.adata)
        if current_state != DataState.RAW and current_state != DataState.UNKNOWN:
            self.log.info(f'Data already in {current_state.value} state, skipping preprocessing...')
            if plots_dir:
                self.plot_expression_distribution(plots_dir, title_prefix="Post-normalization ")
            return
        
        # Plot pre-normalization distribution if plots_dir provided
        if plots_dir:
            self.plot_expression_distribution(plots_dir, title_prefix="Pre-normalization ")
        
        # Check if data is already normalized based on distribution (unless caller already checked)
        if not skip_distribution_check:
            is_normalized, is_log1p_detected = self._check_if_normalized()
            
            # Skip normalization if data appears already normalized
            if self.normalize and is_normalized:
                self.log.warning(f'Data appears already normalized (max value < 50). Skipping normalization step.')
                self.normalize = False  # Don't normalize again
            
            # Skip log1p if data appears already log1p transformed
            if self.apply_log1p and is_log1p_detected:
                self.log.warning(f'Data appears already log1p transformed (max value < 15). Skipping log1p step.')
                self.apply_log1p = False  # Don't log1p again
            
            # If both steps are skipped, mark state and return
            if not self.normalize and not self.apply_log1p:
                self.log.info('Both normalization and log1p appear already done.')
                # Set appropriate state based on detection
                detected_state = get_data_state(self.adata)
                set_data_state(self.adata, detected_state)
                if plots_dir:
                    self.plot_expression_distribution(plots_dir, title_prefix="Post-normalization ")
                return

        if self.normalize:
            # Preserve raw data if it doesn't exist (important for models like STACK that prefer raw counts)
            if self.adata.raw is None:
                self.log.info('Preserving raw counts in adata.raw before normalization...')
                self.adata.raw = self.adata.copy()
            self.log.info(f'Normalizing data (target_sum={self.target_sum})... This may take a while for large datasets.')
            sc.pp.normalize_total(self.adata, target_sum=self.target_sum)
            self.log.info('Normalization complete.')
        if self.apply_log1p:
            self.log.info('Applying log1p transformation... This may take a while for large datasets.')
            sc.pp.log1p(self.adata)
            self.log.info('Log1p transformation complete.')
        
        # Set the data state based on what was applied
        if self.apply_log1p:
            final_state = DataState.LOG1P
        elif self.normalize:
            final_state = DataState.NORMALIZED
        else:
            final_state = DataState.RAW
        
        set_data_state(self.adata, final_state)
        
        self.log.info(
            f"Applied preprocessing: normalize={self.normalize}, log1p={self.apply_log1p}, "
            f"target_sum={self.target_sum}"
        )
        self.log.info(f'Data state after preprocessing: {get_state_summary(self.adata)}')
        
        # Plot post-normalization distribution if plots_dir provided
        if plots_dir:
            self.plot_expression_distribution(plots_dir, title_prefix="Post-normalization ")

    def hvg(self, n_top_genes, flavor, batch_key=None):
        """Select highly variable genes (HVGs).

        Args:
            n_top_genes (int): Number of top HVGs to select.
            flavor (str): Method for HVG selection (e.g., 'seurat').
            batch_key (str, optional): Batch key for batch-aware HVG selection.
        """
        # Ensure we have a copy, not a view, to avoid ImplicitModificationWarning
        if hasattr(self.adata, 'is_view') and self.adata.is_view:
            self.log.debug("Converting view to copy to allow modifications")
            self.adata = self.adata.copy()
        
        # Ensure batch column is categorical if batch_key is provided
        if batch_key is not None and batch_key in self.adata.obs.columns:
            if not pd.api.types.is_categorical_dtype(self.adata.obs[batch_key]):
                self.log.info(f"Converting batch column '{batch_key}' to categorical dtype")
                self.adata.obs[batch_key] = self.adata.obs[batch_key].astype('category')
        
        # Check if data needs log transformation for HVG computation
        # HVG computation (especially seurat flavor) expects log-transformed data
        _, is_log1p = self._check_if_normalized()
        if not is_log1p:
            self.log.info("Data is not log-transformed. Applying log1p transformation for HVG computation.")
            sc.pp.log1p(self.adata)
        
        sc.pp.highly_variable_genes(
            self.adata, batch_key=batch_key, flavor=flavor, subset=True, n_top_genes=n_top_genes)
        self.log.info(
            f"Applied HVG to data.X, n_top_genes = {n_top_genes}, flavor = {flavor}, batch_key = {batch_key}")
        self.log.info(f'Data shape after HVG, {self.adata.X.shape}')


    @staticmethod
    def _mygene_symbols_to_ensembl(symbols: list[str], logger=None) -> dict[str, str]:
        """Convert gene symbols to Ensembl IDs using mygene.
        
        Args:
            symbols: List of gene symbols to convert.
            logger: Optional logger instance for warnings. If None, uses module logger.
            
        Returns:
            Dict mapping symbols to Ensembl IDs (unmapped symbols map to themselves).
        """
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)
        
        try:
            import mygene
        except ImportError:
            logger.warning("mygene not installed. Install with: pip install mygene")
            return {s: s for s in symbols}
        
        mg = mygene.MyGeneInfo()
        mapping = {}
        
        for i in range(0, len(symbols), 1000):
            batch = symbols[i:i + 1000]
            try:
                results = mg.querymany(
                    batch, scopes='symbol,alias', fields='ensembl.gene',
                    species='human', returnall=False, as_dataframe=False
                )
                for r in results:
                    query = r.get('query', '')
                    ensembl = r.get('ensembl', {})
                    if isinstance(ensembl, list):
                        eid = ensembl[0].get('gene', '') if ensembl else ''
                    elif isinstance(ensembl, dict):
                        eid = ensembl.get('gene', '')
                    else:
                        eid = ''
                    mapping[query] = eid if eid else query
            except Exception as e:
                logger.warning(f"mygene batch error: {e}")
                for s in batch:
                    mapping.setdefault(s, s)
        
        return mapping

    def _mygene_ensembl_to_symbols(self, ensembl_ids: list[str]) -> dict[str, str]:
        """Convert Ensembl IDs to gene symbols using mygene.
        
        Args:
            ensembl_ids: List of Ensembl IDs to convert.
            
        Returns:
            Dict mapping Ensembl IDs to symbols (unmapped IDs map to themselves).
        """
        try:
            import mygene
        except ImportError:
            self.log.warning("mygene not installed. Install with: pip install mygene")
            return {e: e for e in ensembl_ids}
        
        mg = mygene.MyGeneInfo()
        mapping = {}
        
        for i in range(0, len(ensembl_ids), 1000):
            batch = ensembl_ids[i:i + 1000]
            try:
                results = mg.querymany(
                    batch, scopes='ensembl.gene', fields='symbol',
                    species='human', returnall=False, as_dataframe=False
                )
                for r in results:
                    eid = r.get('query', '')
                    symbol = r.get('symbol', '')
                    mapping[eid] = symbol if symbol else eid
            except Exception as e:
                self.log.warning(f"mygene batch error: {e}")
                for eid in batch:
                    mapping.setdefault(eid, eid)
        
        return mapping

    def _normalize_symbols_unique(self, symbols: list[str], normalize: bool = True) -> list[str]:
        """Normalize gene symbols and make them unique by appending __dupN suffix.
        
        Args:
            symbols: List of gene symbols.
            normalize: If True, uppercase and strip whitespace.
            
        Returns:
            List of normalized, unique symbols.
        """
        seen: dict[str, int] = {}
        result = []
        for s in symbols:
            norm = str(s).strip().upper() if normalize else str(s).strip()
            if norm in seen:
                seen[norm] += 1
                result.append(f"{norm}__dup{seen[norm]}")
            else:
                seen[norm] = 0
                result.append(norm)
        return result

    def _detect_index_type(self) -> str:
        """Detect whether var.index contains Ensembl IDs or gene symbols.
        
        Returns:
            'ensembl' if >80% of indices look like Ensembl IDs, else 'symbol'.
        """
        sample = self.adata.var.index[:min(100, len(self.adata.var))].astype(str)
        n_ensembl = sum(1 for x in sample if x.startswith(('ENSG', 'ENSMUSG')))
        return 'ensembl' if n_ensembl > len(sample) * 0.8 else 'symbol'

    def _find_ensembl_column(self) -> str | None:
        """Find a column in var that likely contains Ensembl IDs."""
        for col in self.adata.var.columns:
            if any(x in str(col).lower() for x in ['ensembl', 'ensg', 'gene_id']):
                return col
        return None

    def _is_synthetic_data(self) -> bool:
        """Detect if this is synthetic data (e.g., from CONCORD simulation).
        
        Synthetic data typically has:
        - Gene names matching patterns like "Gene_1", "batch_1_Gene_1", etc.
        - 'topology' in uns or obs (CONCORD synthetic data marker)
        
        Returns:
            True if data appears to be synthetic, False otherwise.
        """
        # Check for CONCORD synthetic data marker
        if hasattr(self.adata, 'uns') and 'topology' in self.adata.uns:
            return True
        if hasattr(self.adata, 'obs') and 'topology' in self.adata.obs.columns:
            return True
        
        # Check gene name patterns (CONCORD uses "Gene_1", "Gene_2", etc.)
        sample_genes = self.adata.var.index[:min(20, len(self.adata.var))].astype(str)
        synthetic_patterns = [
            r'^Gene_\d+$',  # "Gene_1", "Gene_2", etc.
            r'^batch_\d+_Gene_\d+$',  # "batch_1_Gene_1", etc.
            r'^cluster_Gene_\d+$',  # "cluster_Gene_1", etc.
            r'^gene_\d+$',  # "gene_1", "gene_2", etc.
        ]
        n_synthetic = sum(
            1 for g in sample_genes 
            if any(re.match(pattern, g) for pattern in synthetic_patterns)
        )
        
        # If >80% of sampled genes match synthetic patterns, it's synthetic
        return n_synthetic > len(sample_genes) * 0.8

    def ensure_both_gene_identifiers(self, gene_symbol_column: str = 'feature_name', normalize: bool = True):
        """Ensure both 'gene_id' (Ensembl) and 'gene_symbol' columns exist in var.
        
        Args:
            gene_symbol_column: Column name containing gene symbols. Default: 'feature_name'
            normalize: If True, uppercase and strip gene symbols. Default: True
        """
        if self.adata is None:
            raise ValueError("Data must be loaded first")
        
        # Convert view to copy if needed
        if getattr(self.adata, 'is_view', False):
            self.adata = self.adata.copy()
        
        index_type = self._detect_index_type()
        has_symbol_col = gene_symbol_column in self.adata.var.columns
        is_synthetic = self._is_synthetic_data()
        
        # --- Ensure 'gene_id' column (Ensembl IDs) ---
        if 'gene_id' not in self.adata.var.columns:
            if index_type == 'ensembl':
                self.adata.var['gene_id'] = self.adata.var.index.astype(str)
                self.log.info("Added 'gene_id' from var.index (Ensembl IDs)")
            elif (ensembl_col := self._find_ensembl_column()):
                self.adata.var['gene_id'] = self.adata.var[ensembl_col].astype(str)
                self.log.info(f"Added 'gene_id' from column '{ensembl_col}'")
            elif is_synthetic:
                # Synthetic data: skip Ensembl conversion, just use gene names as-is
                self.adata.var['gene_id'] = self.adata.var.index.astype(str)
                self.log.info("Synthetic data detected - skipping Ensembl ID conversion. Using gene names as gene_id.")
            else:
                # Convert symbols to Ensembl
                symbols = (self.adata.var[gene_symbol_column].astype(str).tolist()
                          if has_symbol_col else self.adata.var.index.astype(str).tolist())
                # Normalize "SYMBOL (ENTREZID)" format used by DepMap/CCLE so mygene can match
                symbols_for_query = [re.sub(r'\s*\(\d+\)\s*$', '', s).strip() or s for s in symbols]
                self.log.info("Converting gene symbols to Ensembl IDs via mygene...")
                mapping = self._mygene_symbols_to_ensembl(symbols_for_query, logger=self.log)
                ensembl_ids = [mapping.get(sq, s) for sq, s in zip(symbols_for_query, symbols)]
                n_mapped = sum(1 for e in ensembl_ids if str(e).startswith('ENSG'))
                if n_mapped > 0:
                    self.adata.var['gene_id'] = ensembl_ids
                    self.log.info(f"Added 'gene_id' ({n_mapped}/{len(ensembl_ids)} mapped)")
                else:
                    self.log.warning("Could not map any symbols to Ensembl IDs; using symbols as gene_id fallback")
                    # Fallback: use symbols so var['gene_id'] exists for downstream (e.g. scConcept)
                    self.adata.var['gene_id'] = ensembl_ids
        
        # --- Ensure 'gene_symbol' column ---
        if 'gene_symbol' not in self.adata.var.columns:
            if has_symbol_col:
                symbols = self.adata.var[gene_symbol_column].astype(str).tolist()
                self.adata.var['gene_symbol'] = self._normalize_symbols_unique(symbols, normalize)
                self.log.info(f"Added 'gene_symbol' from '{gene_symbol_column}'")
            elif index_type == 'ensembl' and not is_synthetic:
                # Only try mygene conversion for non-synthetic data
                self.log.info("Converting Ensembl IDs to gene symbols via mygene...")
                ensembl_ids = self.adata.var.index.astype(str).tolist()
                mapping = self._mygene_ensembl_to_symbols(ensembl_ids)
                symbols = [mapping.get(e, e) for e in ensembl_ids]
                self.adata.var['gene_symbol'] = self._normalize_symbols_unique(symbols, normalize)
                self.log.info("Added 'gene_symbol' from Ensembl conversion")
            else:
                # Index is already gene symbols (or synthetic data)
                symbols = self.adata.var.index.astype(str).tolist()
                # For synthetic data, don't normalize (keep original names like "Gene_1")
                self.adata.var['gene_symbol'] = self._normalize_symbols_unique(symbols, normalize=not is_synthetic)
                if is_synthetic:
                    self.log.info("Synthetic data detected - using gene names as gene_symbol (no normalization)")
                else:
                    self.log.info("Added 'gene_symbol' from var.index")
        
        self.log.info(f"Gene identifiers: gene_id={'gene_id' in self.adata.var.columns}, "
                      f"gene_symbol={'gene_symbol' in self.adata.var.columns}")

    def ensure_gene_symbols_in_var_index(self, gene_column: str = 'feature_name', normalize: bool = True):
        """Set var.index to gene symbols (for SCimilarity compatibility).
        
        Calls ensure_both_gene_identifiers() first, then sets var.index to 'gene_symbol'.
        
        Args:
            gene_column: Column name containing gene symbols. Default: 'feature_name'
            normalize: If True, uppercase and strip gene symbols. Default: True
        """
        self.ensure_both_gene_identifiers(gene_symbol_column=gene_column, normalize=normalize)
        
        if 'gene_symbol' in self.adata.var.columns:
            self.adata.var.index = self.adata.var['gene_symbol'].values
            self.log.info("Set var.index to gene symbols")
        else:
            self.log.warning("'gene_symbol' column not available, var.index unchanged")


class CSVDataLoader(DataLoader):
    """Loader for CSV-formatted single-cell data."""

    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for CSV loading.
        """
        super().__init__(params)
        self.path = params['path']

    def load(self):
        """Load data from a CSV file.

        Returns:
            ad.AnnData: Loaded data object.
        """
        self.log.info(f"Loading CSV data from {self.path}")
        df = pd.read_csv(self.path)
        labels = df['label'].values
        features = df.drop(columns=['label']).values
        adata = ad.AnnData(X=features, obs=pd.DataFrame({'label': labels}))
        self.adata = adata
        return adata


class H5ADLoader(DataLoader):
    """Loader for H5AD-formatted single-cell data."""

    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for H5AD loading.
        """
        super().__init__(params)
        # Path is already resolved in base class __init__
        self.load_raw = bool(params.get('load_raw', False))
        self.label_key = params['label_key']
        self.batch_key = params['batch_key']
        self.filter = params.get('filter', None)

    def load(self):
        """Load data from an H5AD file, with optional filtering and splitting.

        Returns:
            ad.AnnData: Loaded data object.
        """
        # Check if file exists, and if it's a concord synthetic dataset, generate it automatically
        if not os.path.exists(self.path):
            # Check if this is a concord synthetic dataset that needs to be generated
            if "cell_topology_synthetic" in self.path:
                self.log.warning(f"Dataset file not found: {self.path}")
                self.log.info("Attempting to auto-generate CONCORD synthetic dataset...")
                try:
                    self._generate_concord_dataset()
                    self.log.info(f"✅ Successfully generated dataset: {self.path}")
                except Exception as e:
                    self.log.error(f"Failed to auto-generate dataset: {e}")
                    raise FileNotFoundError(
                        f"Dataset file not found and auto-generation failed: {self.path}. "
                        f"Error: {e}"
                    )
            else:
                raise FileNotFoundError(f"Dataset file not found: {self.path}")
        
        logging.info(f"Loading H5AD data from {self.path}")
        self.log.info("Reading H5AD file (this may take a while for large files)...")
        
        adata = ad.read_h5ad(self.path)
        self.log.info(f"H5AD file loaded. Initial shape: {adata.shape}")

        # Ensure unique obs/var names to avoid "not valid obs/var names or indices" downstream
        if adata.obs_names.duplicated().any():
            self.log.warning("Duplicate obs names detected; making unique.")
            adata.obs_names_make_unique()
        if adata.var_names.duplicated().any():
            self.log.warning("Duplicate var names detected; making unique.")
            adata.var_names_make_unique()

        available = list(adata.obs.columns)
        if self.label_key not in adata.obs.columns:
            raise KeyError(
                f"Label key '{self.label_key}' not found in adata.obs. "
                f"Available columns: {available}. "
                "Check dataset config (e.g. label_key); some datasets use 'CellType' instead of 'cell_type'."
            )
        if self.batch_key not in adata.obs.columns:
            raise KeyError(
                f"Batch key '{self.batch_key}' not found in adata.obs. "
                f"Available columns: {available}."
            )
        adata.obs['label'] = adata.obs[self.label_key]
        adata.obs['batch'] = adata.obs[self.batch_key]

        if self.load_raw:
            self.log.info("Converting raw to adata...")
            adata = adata.raw.to_adata()

        # Preserve counts layer for SCimilarity (requires layers["counts"] for lognorm_counts)
        if 'counts' not in adata.layers:
            if self.layer == 'X':
                adata.layers['counts'] = adata.X
                self.log.info('Stored reference to X as counts layer')
            elif self.layer in adata.layers:
                self.log.info(f"Creating counts layer from {self.layer}...")
                adata.layers['counts'] = adata.layers[self.layer].copy()
                self.log.info(f'Created counts layer from {self.layer}')

        if self.layer != 'X':
            self.log.info(f"Switching to layer '{self.layer}'...")
            if 'original_X' not in adata.layers:
                adata.layers['original_X'] = adata.X
            adata.X = adata.layers[self.layer]

        self.log.info(f'Data Loaded, {adata.X.shape}')
        
        # Compute min/max efficiently
        try:
            if hasattr(adata.X, 'data') and len(adata.X.data) > 0:
                x_min = float(adata.X.data.min()) if adata.X.nnz > 0 else 0.0
                x_max = float(adata.X.data.max()) if adata.X.nnz > 0 else 0.0
            else:
                x_min = float(np.min(adata.X))
                x_max = float(np.max(adata.X))
            self.log.info(f'X min {x_min}, X max {x_max}')
        except Exception as e:
            self.log.info(f'X min/max computation skipped: {e}')

        if self.filter is not None:
            self.log.info(f"Applying filters: {self.filter}")
            filter_dict = {entry["column"]: entry["values"]
                           for entry in self.filter}
            adata = self._filter(adata, filter_dict)
            self.log.info(f"After filtering, shape: {adata.shape}")

        self.adata = adata

        if self.train_test_split:
            fname = os.path.join(BASE_PATH, self.train_test_split)
            with open(fname, 'r') as f:
                self.train_test_split_dict = json.load(f)
            fname = os.path.join(BASE_PATH, self.cv_splits)
            with open(fname, 'r') as f:
                self.cv_split_dict = json.load(f)

            # Fix mismatch: CV splits use 'Post' but data may have 'On' in donor_id_pre_post
            self._translate_split_ids()

        return adata
    
    def _generate_concord_dataset(self):
        """Generate a missing CONCORD synthetic dataset automatically.
        
        This method imports the generation module and calls it to create the dataset.
        Parameters match the official CONCORD benchmark notebooks exactly.
        """
        from data.concord_generator import generate_concord_dataset_by_name
        
        # Extract dataset name from filename (e.g., "concord_trajectory.h5ad" -> "trajectory")
        filename = os.path.basename(self.path)
        dataset_name = filename.replace("concord_", "").replace(".h5ad", "").lower()
        
        # Generate the dataset with benchmark-matched parameters
        # Using None for n_cells/n_genes uses the official benchmark defaults
        generate_concord_dataset_by_name(
            dataset_name=dataset_name,
            output_path=self.path,
            n_cells=None,  # Use benchmark defaults
            n_genes=None,  # Use benchmark defaults
            n_batches=2,   # Match benchmark (2 batches)
            seed=42,
        )
