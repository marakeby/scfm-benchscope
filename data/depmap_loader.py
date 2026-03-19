"""
DepMap drug response data loader.

Loads the DepMap PRISM drug screening dataset for drug response prediction tasks.
The dataset has:
- ~685K observations (cell line × drug pairs)
- ~19K genes (bulk RNA expression per cell line)
- Drug response metrics (AUC, IC50, EC50, etc.)

Special handling:
- The same cell line appears multiple times (once per drug tested)
- For embedding extraction, we need unique cell lines only
- For evaluation, we use the full dataset with all (cell line, drug) pairs
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import anndata as ad

from data.data_loader import H5ADLoader
from utils.logs_ import get_logger


logger = get_logger()


class DepMapLoader(H5ADLoader):
    """
    Loader for DepMap drug response dataset.
    
    The DepMap dataset has one row per (cell line, drug) combination.
    For embedding extraction, we provide methods to get unique cell lines.
    """
    
    # Default column names for DepMap data
    DEFAULT_CELL_LINE_COL = 'depmap_id'
    DEFAULT_DRUG_COL = 'broad_id'
    DEFAULT_TARGET_COL = 'auc'
    DEFAULT_TISSUE_COL = 'cell_line_OncotreeLineage'
    DEFAULT_MOA_COL = 'moa'
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize DepMap loader.
        
        Args:
            params: Configuration parameters including:
                - path: Path to DepMap h5ad file
                - cell_line_col: Column name for cell line IDs (default: 'depmap_id')
                - drug_col: Column name for drug IDs (default: 'broad_id')
                - target_col: Column name for response values (default: 'auc')
                - tissue_col: Column name for tissue lineage (default: 'cell_line_OncotreeLineage')
                - moa_col: Column name for mechanism of action (default: 'moa')
                - extract_unique_cell_lines: If True, subset to unique cell lines for embedding
        """
        # Set defaults for H5AD loader
        params.setdefault('layer_name', 'X')
        params.setdefault('load_raw', False)
        
        # DepMap-specific: use cell line ID as the "batch" concept
        # and tissue as the "label" concept for QC/preprocessing compatibility
        params.setdefault('label_key', params.get('tissue_col', self.DEFAULT_TISSUE_COL))
        params.setdefault('batch_key', params.get('cell_line_col', self.DEFAULT_CELL_LINE_COL))
        params.setdefault('train_test_split', None)
        params.setdefault('cv_splits', None)
        
        super().__init__(params)
        
        # DepMap-specific columns
        self.cell_line_col = params.get('cell_line_col', self.DEFAULT_CELL_LINE_COL)
        self.drug_col = params.get('drug_col', self.DEFAULT_DRUG_COL)
        self.target_col = params.get('target_col', self.DEFAULT_TARGET_COL)
        self.tissue_col = params.get('tissue_col', self.DEFAULT_TISSUE_COL)
        self.moa_col = params.get('moa_col', self.DEFAULT_MOA_COL)
        self.extract_unique = params.get('extract_unique_cell_lines', False)
        
        # Will be populated after load
        self.unique_cell_line_indices = None
        self.cell_line_to_idx = None
    
    def load(self) -> ad.AnnData:
        """
        Load DepMap data.
        
        Returns:
            AnnData with drug response data. If extract_unique_cell_lines=True,
            returns only unique cell lines for embedding extraction.
        """
        self.log.info(f"Loading DepMap data from {self.path}")
        
        # Load full data
        adata = ad.read_h5ad(self.path)
        self.log.info(f"Loaded DepMap data: {adata.shape}")
        self.log.info(f"  Cell lines: {adata.obs[self.cell_line_col].nunique()}")
        self.log.info(f"  Drugs: {adata.obs[self.drug_col].nunique()}")
        
        # Store full data reference
        self.full_adata = adata
        
        # Build cell line index mapping
        cell_lines = adata.obs[self.cell_line_col].values
        unique_cls, first_indices = np.unique(cell_lines, return_index=True)
        self.unique_cell_line_indices = first_indices
        self.cell_line_to_idx = {cl: i for i, cl in enumerate(unique_cls)}
        
        self.log.info(f"  Unique cell lines: {len(unique_cls)}")
        
        # Map standard keys
        adata.obs['label'] = adata.obs[self.tissue_col]
        adata.obs['batch'] = adata.obs[self.cell_line_col]
        
        # Apply filter if specified
        if hasattr(self, 'filter') and self.filter is not None:
            filter_dict = {entry["column"]: entry["values"] for entry in self.filter}
            adata = self._filter(adata, filter_dict)
            self.log.info(f"After filtering: {adata.shape}")
        
        if self.extract_unique:
            # Return only unique cell lines for embedding extraction
            adata = adata[first_indices].copy()
            self.log.info(f"Extracted unique cell lines: {adata.shape}")
        
        self.adata = adata
        return adata
    
    def get_unique_cell_lines_adata(self) -> ad.AnnData:
        """
        Get AnnData with only unique cell lines (for embedding extraction).
        
        Returns:
            AnnData subset with one row per unique cell line
        """
        if self.full_adata is None:
            raise ValueError("Must call load() first")
        
        return self.full_adata[self.unique_cell_line_indices].copy()
    
    def get_full_adata(self) -> ad.AnnData:
        """
        Get full AnnData with all (cell line, drug) pairs (for evaluation).
        
        Returns:
            Full AnnData with all observations
        """
        if self.full_adata is None:
            raise ValueError("Must call load() first")
        
        return self.full_adata
    
    def map_embeddings_to_full(
        self, 
        unique_embeddings: np.ndarray,
        embedding_key: str = 'X_embedding'
    ) -> None:
        """
        Map embeddings from unique cell lines to all observations.
        
        Args:
            unique_embeddings: Embeddings for unique cell lines (n_unique, dim)
            embedding_key: Key to store in obsm
        """
        if self.full_adata is None:
            raise ValueError("Must call load() first")
        
        if len(unique_embeddings) != len(self.unique_cell_line_indices):
            raise ValueError(
                f"Embedding count ({len(unique_embeddings)}) doesn't match "
                f"unique cell line count ({len(self.unique_cell_line_indices)})"
            )
        
        # Get cell line for each row in full data
        cell_lines = self.full_adata.obs[self.cell_line_col].values
        
        # Map unique cell lines to their index in unique_embeddings
        unique_cls = self.full_adata.obs.iloc[self.unique_cell_line_indices][self.cell_line_col].values
        cl_to_emb_idx = {cl: i for i, cl in enumerate(unique_cls)}
        
        # Create full embedding matrix by repeating embeddings
        indices = np.array([cl_to_emb_idx[cl] for cl in cell_lines])
        full_embeddings = unique_embeddings[indices]
        
        # Store in full adata
        self.full_adata.obsm[embedding_key] = full_embeddings
        
        # Also update current adata reference
        if self.adata is self.full_adata:
            pass  # Already updated
        elif self.extract_unique:
            self.adata.obsm[embedding_key] = unique_embeddings
        else:
            self.adata.obsm[embedding_key] = full_embeddings
        
        self.log.info(f"Mapped embeddings to full data: {full_embeddings.shape}")
    
    def get_split_statistics(self) -> pd.DataFrame:
        """
        Get statistics about the dataset for split planning.
        
        Returns:
            DataFrame with statistics per cell line and per drug
        """
        if self.full_adata is None:
            raise ValueError("Must call load() first")
        
        obs = self.full_adata.obs
        
        # Cell line statistics
        cl_stats = obs.groupby(self.cell_line_col, observed=True).agg(
            n_drugs=(self.drug_col, 'nunique'),
            auc_mean=(self.target_col, 'mean'),
            auc_std=(self.target_col, 'std'),
            tissue=(self.tissue_col, 'first'),
        )
        
        # Drug statistics
        drug_stats = obs.groupby(self.drug_col, observed=True).agg(
            n_cell_lines=(self.cell_line_col, 'nunique'),
            auc_mean=(self.target_col, 'mean'),
            auc_std=(self.target_col, 'std'),
            moa=(self.moa_col, 'first'),
        )
        
        # MoA statistics
        moa_stats = obs.groupby(self.moa_col, observed=True).agg(
            n_drugs=(self.drug_col, 'nunique'),
            n_samples=(self.target_col, 'count'),
            auc_mean=(self.target_col, 'mean'),
            auc_std=(self.target_col, 'std'),
        )
        
        return {
            'cell_line': cl_stats,
            'drug': drug_stats,
            'moa': moa_stats,
        }


class DepMapUniqueLoader(DepMapLoader):
    """
    DepMap loader that returns only unique cell lines.
    
    Use this for the embedding extraction step, then use map_embeddings_to_full()
    to propagate embeddings to all (cell line, drug) pairs for evaluation.
    """
    
    def __init__(self, params: Dict[str, Any]):
        params['extract_unique_cell_lines'] = True
        super().__init__(params)
