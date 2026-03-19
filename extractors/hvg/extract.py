#!/usr/bin/env python
"""HVG (Highly Variable Genes) embedding extraction script.

This script can be run standalone in the base environment.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --n_top_genes 2000 --flavor seurat
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


class HVGExtractor(BaseExtractor):
    """HVG embedding extractor (returns expression of highly variable genes)."""
    
    def __init__(
        self,
        n_top_genes: int = 2000,
        flavor: str = "seurat",
        batch_key: str | None = None,
        **kwargs
    ):
        super().__init__(
            n_top_genes=n_top_genes,
            flavor=flavor,
            batch_key=batch_key,
            **kwargs
        )
        self.n_top_genes = n_top_genes
        self.flavor = flavor
        self.batch_key = batch_key
    
    @property
    def model_name(self) -> str:
        return "HVG"
    
    @property
    def embedding_dim(self) -> int:
        return self.n_top_genes
    
    def load_model(self) -> None:
        """HVG doesn't require model loading."""
        logger.info("HVG extractor ready")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using highly variable genes.
        
        Args:
            adata: AnnData object
            
        Returns:
            Expression matrix of shape (n_cells, n_top_genes)
        """
        adata = adata.copy()
        
        logger.info(f"Selecting top {self.n_top_genes} highly variable genes...")
        sc.pp.highly_variable_genes(
            adata,
            flavor=self.flavor,
            subset=False,
            batch_key=self.batch_key,
            n_top_genes=self.n_top_genes
        )
        
        # Extract expression of highly variable genes
        hvg_mask = adata.var.highly_variable.values
        embeddings = adata.X[:, hvg_mask]
        
        # Convert to dense if sparse
        if hasattr(embeddings, 'toarray'):
            embeddings = embeddings.toarray()
        
        logger.info(f"Extracted HVG embeddings. Shape: {embeddings.shape}")
        
        return embeddings


def main():
    parser = create_argument_parser()
    
    # HVG-specific arguments
    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=2000,
        help="Number of top highly variable genes (default: 2000)"
    )
    parser.add_argument(
        "--flavor",
        type=str,
        default="seurat",
        choices=["seurat", "seurat_v3", "cell_ranger"],
        help="Flavor for highly variable gene selection (default: seurat)"
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default=None,
        help="Key in adata.obs for batch information (optional)"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = HVGExtractor(
        n_top_genes=args.n_top_genes,
        flavor=args.flavor,
        batch_key=args.batch_key,
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
