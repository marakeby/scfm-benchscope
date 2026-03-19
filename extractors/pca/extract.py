#!/usr/bin/env python
"""PCA embedding extraction script.

This script can be run standalone in the base environment.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --n_components 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

import logging
logger = logging.getLogger(__name__)


class PCAExtractor(BaseExtractor):
    """PCA embedding extractor."""
    
    def __init__(
        self,
        params: dict | None = None,
        **kwargs,
    ):
        # Support both init styles: dict (from registry) and kwargs (from CLI)
        super().__init__(params=params, **kwargs)
        self.n_components = int(self.params.get("n_components", 50))
        self.use_hvg = bool(self.params.get("use_hvg", False))
        self.n_top_genes = int(self.params.get("n_top_genes", 2000))
        self.hvg_flavor = str(self.params.get("hvg_flavor", "seurat"))
    
    @property
    def model_name(self) -> str:
        return "PCA"
    
    @property
    def embedding_dim(self) -> int:
        return self.n_components
    
    def load_model(self) -> None:
        """PCA doesn't require model loading."""
        logger.info("PCA extractor ready (no model to load)")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using PCA.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array of shape (n_cells, n_components)
        """
        adata = adata.copy()
        
        # Optionally select highly variable genes
        if self.use_hvg:
            logger.info(f"Selecting top {self.n_top_genes} highly variable genes...")
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_top_genes,
                flavor=self.hvg_flavor,
                subset=True
            )
        
        # Scale the data
        logger.info("Scaling the data...")
        sc.pp.scale(adata, max_value=10)
        
        # Zero-variance genes become NaN after scaling — fill with 0
        import scipy.sparse as sp
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        nan_mask = np.isnan(X)
        n_nan_genes = nan_mask.any(axis=0).sum()
        if n_nan_genes > 0:
            logger.info(
                f"Filling NaN in {n_nan_genes} zero-variance genes after scaling"
            )
            X[nan_mask] = 0.0
            adata.X = X
        
        # Run PCA
        logger.info(f"Running PCA with {self.n_components} components...")
        sc.tl.pca(adata, n_comps=self.n_components)
        
        return adata.obsm['X_pca'].copy()


def main():
    parser = create_argument_parser()
    
    # PCA-specific arguments
    parser.add_argument(
        "--n_components",
        type=int,
        default=50,
        help="Number of PCA components (default: 50)"
    )
    parser.add_argument(
        "--use_hvg",
        action="store_true",
        help="Use highly variable genes before PCA"
    )
    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=2000,
        help="Number of top highly variable genes (default: 2000)"
    )
    parser.add_argument(
        "--hvg_flavor",
        type=str,
        default="seurat",
        choices=["seurat", "seurat_v3", "cell_ranger"],
        help="Flavor for highly variable gene selection (default: seurat)"
    )
    
    args = parser.parse_args()
    
    # Create extractor (CLI style — kwargs handled by BaseExtractor)
    extractor = PCAExtractor(
        n_components=args.n_components,
        use_hvg=args.use_hvg,
        n_top_genes=args.n_top_genes,
        hvg_flavor=args.hvg_flavor,
    )
    
    # Run extraction
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
