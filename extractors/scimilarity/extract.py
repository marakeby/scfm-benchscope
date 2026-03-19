#!/usr/bin/env python
"""SCimilarity embedding extraction script.

This script can be run standalone in the base environment.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_path /path/to/scimilarity/model
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
from scipy.sparse import issparse

# Import scimilarity BEFORE modifying sys.path to avoid import shadowing
# (extractors/scimilarity/ would shadow the installed scimilarity package)
try:
    from scimilarity.cell_embedding import CellEmbedding
    from scimilarity.utils import align_dataset, lognorm_counts
    SCIMILARITY_AVAILABLE = True
except ImportError:
    SCIMILARITY_AVAILABLE = False

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction, normalize_gene_name_for_vocab

import logging
logger = logging.getLogger(__name__)

if not SCIMILARITY_AVAILABLE:
    logger.error("SCimilarity package not found. Please install it.")


class SCimilarityExtractor(BaseExtractor):
    """SCimilarity embedding extractor.
    
    Extracts single-cell embeddings using SCimilarity.
    """
    
    def __init__(
        self,
        params: dict = None,
        model_path: str = None,
        use_gpu: bool = False,
        buffer_size: int = 10000,
        **kwargs
    ):
        # Handle both YAML config style (params dict) and CLI style (kwargs)
        # params must be first positional to match base class and registry calling convention
        if params is not None:
            super().__init__(params=params)
            # After super().__init__, self.params contains the merged config
            # Extract values from self.params, falling back to function parameters
            model_path = self.params.get('model_path', model_path)
            use_gpu = self.params.get('use_gpu', use_gpu)
            buffer_size = self.params.get('buffer_size', buffer_size)
        else:
            super().__init__(
                model_path=model_path,
                use_gpu=use_gpu,
                buffer_size=buffer_size,
                **kwargs
            )
            # For CLI style, self.params contains kwargs, extract from there
            model_path = self.params.get('model_path', model_path)
            use_gpu = self.params.get('use_gpu', use_gpu)
            buffer_size = self.params.get('buffer_size', buffer_size)
        
        if not SCIMILARITY_AVAILABLE:
            raise ImportError("SCimilarity package not found.")
        
        # Ensure model_path is a string, not a dict
        if isinstance(model_path, dict):
            raise ValueError(
                f"model_path must be a string, got dict: {model_path}. "
                f"This usually means params structure is incorrect."
            )
        
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.buffer_size = buffer_size
        self.ce = None
        
        logger.info(f'SCimilarityExtractor initialized: model_path={model_path}, use_gpu={use_gpu}')
    
    @property
    def model_name(self) -> str:
        return "SCimilarity"
    
    @property
    def embedding_dim(self) -> int:
        return -1  # Unknown until model is loaded
    
    def load_model(self) -> None:
        """Load SCimilarity model."""
        if self._model_loaded:
            return
            
        logger.info(f"Loading SCimilarity model from {self.model_path}")
        self.ce = CellEmbedding(model_path=self.model_path, use_gpu=self.use_gpu)
        self._model_loaded = True
        logger.info("SCimilarity model loaded successfully")
    
    def preprocess_h5ad(self, h5ad_path_data, reorder: bool = True, norm: bool = True):
        """
        Loads and preprocesses AnnData according to SCimilarity requirements:
          - Aligns gene ordering (SCimilarity-specific, always needed)
          - Applies log-normalization (TP10K) - skips if already done centrally

        Returns:
            AnnData object with processed .X
        """
        if isinstance(h5ad_path_data, str):
            adata = sc.read_h5ad(h5ad_path_data)
        else:
            adata = h5ad_path_data

        # Use gene symbols from saved column (SCimilarity requires gene symbols in var.index)
        if 'gene_symbol' in adata.var.columns:
            adata.var.index = [
                normalize_gene_name_for_vocab(str(s)) for s in adata.var['gene_symbol'].values
            ]
        else:
            raise ValueError("gene_symbol column not found in adata.var")

        # SCimilarity's align_dataset requires non-negative X (e.g. counts). Tumor/config data
        # may have adata.X set to a normalized or scaled layer with negative values.
        X = adata.X
        x_min = float(np.min(X)) if not issparse(X) else float(X.data.min()) if X.nnz > 0 else 0.0
        if x_min < 0:
            if "counts" in adata.layers:
                counts = adata.layers["counts"]
                c_min = float(np.min(counts)) if not issparse(counts) else float(counts.data.min()) if counts.nnz > 0 else 0.0
                if c_min >= 0:
                    logger.info(
                        "adata.X has negative values; using adata.layers['counts'] for SCimilarity alignment."
                    )
                    adata.X = adata.layers["counts"].copy()
                else:
                    logger.warning(
                        "adata.X and layers['counts'] contain negative values; clipping X to 0 for alignment."
                    )
                    if issparse(X):
                        X = X.copy()
                        X.data = np.clip(X.data, 0, None)
                        adata.X = X
                    else:
                        adata.X = np.clip(np.asarray(X), 0, None)
            else:
                logger.warning(
                    "adata.X has negative values and no layers['counts']; clipping X to 0 for alignment."
                )
                if issparse(X):
                    X = X.copy()
                    X.data = np.clip(X.data, 0, None)
                    adata.X = X
                else:
                    adata.X = np.clip(np.asarray(X), 0, None)

        # align_dataset also requires layers['counts'] to be non-negative; clip if needed.
        if "counts" in adata.layers:
            counts = adata.layers["counts"]
            c_min = float(np.min(counts)) if not issparse(counts) else float(counts.data.min()) if counts.nnz > 0 else 0.0
            if c_min < 0:
                logger.warning(
                    "layers['counts'] contains negative values; clipping to 0 for SCimilarity."
                )
                if issparse(counts):
                    counts = counts.copy()
                    counts.data = np.clip(counts.data, 0, None)
                    adata.layers["counts"] = counts
                else:
                    adata.layers["counts"] = np.clip(np.asarray(counts), 0, None)

        # Do gene alignment (SCimilarity-specific requirement)
        if reorder:
            # Normalize SCimilarity's gene_order to uppercase to match our normalized dataset genes
            # This fixes case sensitivity issues (e.g., 'C9orf152' vs 'C9ORF152')
            gene_order_normalized = [str(g).strip().upper() for g in self.ce.gene_order]
            logger.info(f"Normalizing SCimilarity gene_order to uppercase for case-insensitive matching")
            adata = align_dataset(adata, gene_order_normalized)
        
        # Check if normalization already done centrally
        if norm:
            if hasattr(adata, 'uns') and adata.uns.get('preprocessed', False):
                logger.info('Normalization already done centrally, skipping lognorm_counts()')
            else:
                # Do SCimilarity's normalization (equivalent to centralized normalize + log1p)
                adata = lognorm_counts(adata)

        return adata
    
    def get_embeddings(self, adata, num_cells: int = -1, buffer_size: int = 10000):
        """
        Computes embeddings using the SCimilarity model.

        Returns:
            numpy.ndarray of shape [num_cells, latent_dim]
        """
        X = adata.X
        embeddings = self.ce.get_embeddings(X, num_cells=num_cells, buffer_size=buffer_size)
        return embeddings
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using SCimilarity.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array
        """
        if not self._model_loaded:
            self.load_model()
        
        # Always do preprocessing - it will check internally if normalization is already done
        # This ensures gene alignment is always performed (SCimilarity-specific requirement)
        logger.info('Applying SCimilarity preprocessing (gene alignment + normalization if needed)...')
        adata = self.preprocess_h5ad(adata, reorder=True, norm=True)
        
        embeddings = self.get_embeddings(adata, num_cells=-1, buffer_size=self.buffer_size)
        adata.obsm['X_scimilarity'] = embeddings
        
        logger.info(f"Extracted SCimilarity embeddings. Shape: {embeddings.shape}")
        
        return embeddings


def main():
    parser = create_argument_parser()
    
    # SCimilarity-specific arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to SCimilarity model directory"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Enable GPU"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10000,
        help="Buffer size for embedding extraction (default: 10000)"
    )
    
    args = parser.parse_args()
    
    extractor = SCimilarityExtractor(
        model_path=args.model_path,
        use_gpu=args.use_gpu,
        buffer_size=args.buffer_size,
    )
    
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
