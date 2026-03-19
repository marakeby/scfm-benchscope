#!/usr/bin/env python
"""scConcept embedding extraction script.

This script requires a separate environment due to dependency conflicts.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_name Corpus-30M --cache_dir ./cache/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction, normalize_gene_name_for_vocab
from data.data_loader import DataLoader

import logging
logger = logging.getLogger(__name__)

try:
    from concept import scConcept
    SCCONCEPT_AVAILABLE = True
except ImportError:
    SCCONCEPT_AVAILABLE = False
    logger.error("scConcept package not found. Please install it.")


class SpeciesMismatchError(Exception):
    """Exception raised when dataset species doesn't match model requirements."""
    pass


class scConceptExtractor(BaseExtractor):
    """scConcept embedding extractor.
    
    Extracts cell embeddings using pre-trained scConcept models.
    Stores embeddings in adata.obsm['X_scConcept'].
    """
    
    def __init__(
        self,
        params: dict = None,
        model_name: str = "Corpus-30M",
        cache_dir: str = "./cache/",
        batch_size: int = 32,
        gene_id_column: str | None = None,
        repo_id: str = "theislab/scConcept",
        **kwargs
    ):
        # Handle both YAML config style (params dict) and CLI style (kwargs)
        # params must be first positional to match base class and registry calling convention
        if params is not None:
            super().__init__(params=params)
            # After super().__init__, self.params contains the merged config
            model_name = self.params.get('model_name', model_name)
            cache_dir = self.params.get('cache_dir', cache_dir)
            batch_size = self.params.get('batch_size', batch_size)
            gene_id_column = self.params.get('gene_id_column', gene_id_column)
            repo_id = self.params.get('repo_id', repo_id)
            # num_workers for DataLoader (None = auto in scConcept api)
            self.num_workers = self.params.get('num_workers', None)
        else:
            super().__init__(
                model_name=model_name,
                cache_dir=cache_dir,
                batch_size=batch_size,
                gene_id_column=gene_id_column,
                repo_id=repo_id,
                **kwargs
            )
            # For CLI style, self.params contains kwargs, extract from there
            model_name = self.params.get('model_name', model_name)
            cache_dir = self.params.get('cache_dir', cache_dir)
            batch_size = self.params.get('batch_size', batch_size)
            gene_id_column = self.params.get('gene_id_column', gene_id_column)
            repo_id = self.params.get('repo_id', repo_id)
            self.num_workers = self.params.get('num_workers', None)
        
        if not SCCONCEPT_AVAILABLE:
            raise ImportError(
                "scConcept is not available. Please install it with: "
                "pip install git+https://github.com/theislab/scConcept.git@main"
            )
        
        # Ensure model_name is a string, not a dict
        if isinstance(model_name, dict):
            raise ValueError(
                f"model_name must be a string, got dict: {model_name}. "
                f"This usually means params structure is incorrect."
            )
        
        self.model_name_str = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.gene_id_column = gene_id_column
        self.repo_id = repo_id
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scConcept instance (model will be loaded in fit_transform)
        self.concept = None
        self._model_loaded = False
        
        logger.info(
            f"scConcept extractor initialized: "
            f"model_name={self.model_name_str}, cache_dir={self.cache_dir}, "
            f"batch_size={self.batch_size}, gene_id_column={self.gene_id_column}"
        )
    
    @property
    def model_name(self) -> str:
        return f"scConcept-{self.model_name_str}"
    
    @property
    def embedding_dim(self) -> int:
        return -1  # Unknown until model is loaded
    
    def load_model(self) -> None:
        """Load scConcept model if not already loaded."""
        if self._model_loaded and self.concept is not None:
            return
        
        logger.info(f"Loading scConcept model: {self.model_name_str}")
        
        # Initialize scConcept
        self.concept = scConcept(
            repo_id=self.repo_id,
            cache_dir=str(self.cache_dir)
        )
        
        # Load model
        self.concept.load_config_and_model(model_name=self.model_name_str)
        
        self._model_loaded = True
        logger.info(f"scConcept model {self.model_name_str} loaded successfully")
    
    def _check_species(self, data) -> bool:
        """Check if dataset contains human gene IDs (required for scConcept).
        
        Args:
            data: AnnData object
            
        Returns:
            bool: True if human, False if non-human (should skip)
        """
        # Sample gene IDs to detect species
        sample_gene_ids = data.var.index.values[:100]
        is_mouse = any('ENSMUSG' in str(gid) for gid in sample_gene_ids)
        is_human = any('ENSG' in str(gid) for gid in sample_gene_ids) and not is_mouse
        
        if is_mouse:
            return False
        elif not is_human:
            # Unknown format - might still work, but warn
            logger.warning(
                f"Could not detect human Ensembl IDs (ENSG*) in gene IDs. "
                f"Sample: {sample_gene_ids[:5]}. Proceeding with caution."
            )
        return True
    
    def _convert_symbols_to_ensembl(self, adata: sc.AnnData) -> sc.AnnData:
        """Convert gene symbols to Ensembl IDs using mygene.
        
        Uses the existing _mygene_symbols_to_ensembl method from DataLoader.
        
        Args:
            adata: AnnData object with gene symbols in var.index or gene_symbol column
            
        Returns:
            AnnData with Ensembl IDs set as var.index
            
        Raises:
            ImportError: If mygene is not installed
            ValueError: If conversion fails completely
        """
        # Get gene symbols - try gene_symbol column first, then var.index
        if 'gene_symbol' in adata.var.columns:
            gene_symbols = adata.var['gene_symbol'].astype(str).tolist()
            logger.info(f"Using 'gene_symbol' column for conversion ({len(gene_symbols)} genes)")
        else:
            gene_symbols = adata.var.index.astype(str).tolist()
            logger.info(f"Using var.index for conversion ({len(gene_symbols)} genes)")
        # Normalize "SYMBOL (ENTREZ_ID)" -> "SYMBOL" for mygene lookup
        gene_symbols = [normalize_gene_name_for_vocab(s) for s in gene_symbols]
        
        # Use DataLoader's existing static method for conversion
        logger.info(f"Converting {len(gene_symbols)} gene symbols to Ensembl IDs using mygene...")
        symbol_to_ensembl = DataLoader._mygene_symbols_to_ensembl(gene_symbols, logger=logger)
        
        # Map all symbols to Ensembl IDs
        ensembl_ids = [symbol_to_ensembl.get(s, s) for s in gene_symbols]
        n_mapped = sum(1 for eid in ensembl_ids if str(eid).startswith('ENSG'))
        
        logger.info(f"Conversion complete: {n_mapped}/{len(ensembl_ids)} genes mapped to Ensembl IDs ({100*n_mapped/len(ensembl_ids):.1f}%)")
        
        if n_mapped == 0:
            raise ValueError(
                f"Failed to convert any gene symbols to Ensembl IDs. "
                f"Sample symbols: {gene_symbols[:5]}. "
                f"This dataset may not contain human gene symbols."
            )
        
        # Update var.index with Ensembl IDs
        adata.var.index = ensembl_ids

        # Ensure var index is string (handles float/NaN from gene_id or conversion)
        self._ensure_var_index_string(adata)
        # Make var names unique to avoid duplicates (unmapped genes keep original symbols)
        adata.var_names_make_unique()

        return adata

    @staticmethod
    def _ensure_var_index_string(adata: sc.AnnData) -> None:
        """Ensure var.index is string type so var_names_make_unique() does not fail on float/NaN.

        When gene_id (or var.index) contains missing values, pandas uses float dtype and NaN.
        anndata's make_index_unique does 'v + join + str(...)', which raises TypeError for float.
        """
        idx = adata.var.index
        try:
            na_mask = idx.isna() if hasattr(idx, "isna") else np.isnan(idx) if idx.dtype.kind == "f" else np.zeros(len(idx), dtype=bool)
        except (TypeError, ValueError):
            na_mask = np.zeros(len(idx), dtype=bool)
        if np.any(na_mask) or (getattr(idx.dtype, "kind", None) == "f"):
            out = np.array([f"unknown_{i}" if na_mask[i] else str(idx[i]) for i in range(len(idx))], dtype=object)
            adata.var.index = out
        else:
            adata.var.index = idx.astype(str)

    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using scConcept.
        
        Note: scConcept's model applies log1p internally during batch processing.
        If data is already normalized+log1p (preprocessed=True), this will result
        in double log1p transformation. However, scConcept's API doesn't provide
        a way to skip this, so we log a warning.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array
            
        Raises:
            SpeciesMismatchError: If dataset contains non-human gene IDs
        """
        if not self._model_loaded:
            self.load_model()
        
        logger.info(f"Extracting scConcept embeddings from AnnData with shape {adata.shape}")
        
        # Check species early - skip non-human datasets
        if not self._check_species(adata):
            sample_gene_ids = adata.var.index.values[:5]
            raise SpeciesMismatchError(
                f"Dataset contains non-human gene IDs (detected mouse: ENSMUSG*). "
                f"scConcept Corpus-30M model requires human gene IDs (ENSG*). "
                f"Sample gene IDs: {sample_gene_ids.tolist()}. "
                f"Skipping scConcept extraction for this dataset."
            )
        
        # Check if data is already log1p transformed
        # scConcept applies log1p internally, so we skip it if data is already transformed
        from utils.data_state import DataState, get_data_state
        current_state = get_data_state(adata)
        # YAML/params override (e.g. DepMap has max~17 so detected as normalized but is log-scaled)
        if isinstance(self.params.get('skip_log1p'), bool):
            skip_log1p = self.params['skip_log1p']
            logger.info(f"Using skip_log1p={skip_log1p} from config (override)")
        else:
            skip_log1p = (current_state == DataState.LOG1P)

        if skip_log1p:
            logger.info(
                f"Data state is {current_state.value}. "
                "Will skip log1p in scConcept to avoid double transformation."
            )
        else:
            logger.info(
                f"Data state is {current_state.value}. "
                "scConcept will apply log1p internally during batch processing."
            )
        
        # Use Ensembl IDs from saved column (SCConcept requires Ensembl IDs)
        gene_id_column_to_use = None
        
        if 'gene_id' in adata.var.columns:
            # Check if gene_id column actually contains Ensembl IDs
            sample_gene_ids = adata.var['gene_id'][:min(100, len(adata.var))].astype(str)
            n_ensembl = sum(1 for gid in sample_gene_ids if str(gid).startswith('ENSG'))
            
            if n_ensembl > len(sample_gene_ids) * 0.5:
                # Use gene_id column as var.index
                adata.var.index = adata.var['gene_id'].values
                self._ensure_var_index_string(adata)  # Handles float/NaN so make_unique doesn't fail
                adata.var_names_make_unique()  # Handle duplicates from unmapped genes
                gene_id_column_to_use = None  # Use var.index
                logger.info(f"Using 'gene_id' column as var.index for scConcept ({n_ensembl}/{len(sample_gene_ids)} Ensembl IDs)")
            else:
                # gene_id column exists but doesn't contain Ensembl IDs - need conversion
                logger.warning(f"'gene_id' column exists but contains few Ensembl IDs ({n_ensembl}/{len(sample_gene_ids)})")
                adata = self._convert_symbols_to_ensembl(adata)
                gene_id_column_to_use = None
        else:
            # Check if var.index already contains Ensembl IDs
            sample_indices = adata.var.index[:min(100, len(adata.var.index))].astype(str)
            n_ensembl = sum(1 for idx in sample_indices if str(idx).startswith('ENSG'))
            index_has_ensembl = n_ensembl > len(sample_indices) * 0.5
            
            if index_has_ensembl:
                # var.index already has Ensembl IDs, use it directly
                self._ensure_var_index_string(adata)  # Handles float/NaN so make_unique doesn't fail
                adata.var_names_make_unique()  # Ensure uniqueness
                gene_id_column_to_use = None
                logger.info(f"var.index contains Ensembl IDs ({n_ensembl}/{len(sample_indices)} detected), using directly")
            elif self.gene_id_column is not None:
                # User specified a custom column
                gene_id_column_to_use = self.gene_id_column
                logger.info(f"Using user-specified gene_id_column: {self.gene_id_column}")
            else:
                # No Ensembl IDs available - convert gene symbols to Ensembl IDs
                logger.info("No Ensembl IDs found. Converting gene symbols to Ensembl IDs...")
                adata = self._convert_symbols_to_ensembl(adata)
                gene_id_column_to_use = None
        
        # Extract embeddings (num_workers must be int; PyTorch DataLoader rejects None)
        num_workers = getattr(self, 'num_workers', None)
        if num_workers is None:
            num_workers = 2
        logger.info(
            f"Extracting embeddings with parameters: "
            f"batch_size={self.batch_size}, gene_id_column={gene_id_column_to_use}, skip_log1p={skip_log1p}, num_workers={num_workers}"
        )
        result = self.concept.extract_embeddings(
            adata=adata,
            batch_size=self.batch_size,
            gene_id_column=gene_id_column_to_use,
            skip_log1p=skip_log1p,
            num_workers=num_workers,
        )
        
        # Store CLS embeddings in adata.obsm (use array; avoid DataFrame index alignment on duplicate obs names)
        embeddings = result['cls_cell_emb']
        if hasattr(embeddings, 'index') and hasattr(embeddings, 'loc'):
            # DataFrame: align by obs_names then pass values so AnnData never sees duplicate index
            emb = np.asarray(embeddings.loc[adata.obs_names].values, dtype=np.float32)
        else:
            emb = np.asarray(embeddings, dtype=np.float32)
        if emb.shape[0] != adata.n_obs:
            raise ValueError(
                f"Embedding row count ({emb.shape[0]}) does not match adata.n_obs ({adata.n_obs}). "
                "Ensure unique obs names (e.g. adata.obs_names_make_unique()) before extraction."
            )
        adata.obsm['X_scConcept'] = emb

        logger.info(
            f"Extracted embeddings with shape {emb.shape} "
            f"and stored in adata.obsm['X_scConcept']"
        )

        return emb.copy()


def main():
    parser = create_argument_parser()
    
    # scConcept-specific arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Corpus-30M",
        help="scConcept model name (default: Corpus-30M)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache/",
        help="Directory for cached models (default: ./cache/)"
    )
    parser.add_argument(
        "--gene_id_column",
        type=str,
        default=None,
        help="Column name in adata.var for gene IDs (default: None, uses index)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="theislab/scConcept",
        help="HuggingFace repository ID (default: theislab/scConcept)"
    )
    
    args = parser.parse_args()
    
    extractor = scConceptExtractor(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        gene_id_column=args.gene_id_column,
        repo_id=args.repo_id,
    )
    
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
