#!/usr/bin/env python
"""Baseline embedding extraction script.

This script can be run standalone in the base environment.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --baseline_type no_integration
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

# Import baseline functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "features"))
from baseline_embeddings import generate_baseline_embeddings

import logging
logger = logging.getLogger(__name__)


class BaselineExtractor(BaseExtractor):
    """Baseline embedding extractor."""
    
    def __init__(
        self,
        baseline_type: str = "no_integration",
        batch_key: str = "batch",
        label_key: str = "cell_type",
        embedding_key: str | None = "X_pca",
        random_state: int = 0,
        **kwargs
    ):
        super().__init__(
            baseline_type=baseline_type,
            batch_key=batch_key,
            label_key=label_key,
            embedding_key=embedding_key,
            random_state=random_state,
            **kwargs
        )
        self.baseline_type = baseline_type
        self.batch_key = batch_key
        self.label_key = label_key
        self.embedding_key = embedding_key
        self.random_state = random_state
    
    @property
    def model_name(self) -> str:
        return f"Baseline-{self.baseline_type}"
    
    @property
    def embedding_dim(self) -> int:
        # Most baselines use 50 dimensions (PCA-based)
        return 50
    
    def load_model(self) -> None:
        """Baseline doesn't require model loading."""
        logger.info(f"Baseline extractor ready (type: {self.baseline_type})")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract baseline embeddings.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array
        """
        logger.info(f"Generating {self.baseline_type} baseline embeddings...")
        
        embeddings = generate_baseline_embeddings(
            adata,
            baseline_type=self.baseline_type,
            batch_key=self.batch_key,
            label_key=self.label_key,
            embedding_key=self.embedding_key,
            random_state=self.random_state
        )
        
        logger.info(f"Generated baseline embeddings. Shape: {embeddings.shape}")
        
        return embeddings


def main():
    parser = create_argument_parser()
    
    # Baseline-specific arguments
    parser.add_argument(
        "--baseline_type",
        type=str,
        default="no_integration",
        choices=[
            "no_integration",
            "no_integration_batch",
            "perfect_cell_type",
            "perfect_cell_type_jittered",
            "shuffle",
            "shuffle_by_batch",
            "shuffle_by_cell_type"
        ],
        help="Type of baseline to generate (default: no_integration)"
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default="batch",
        help="Key in adata.obs for batch labels (default: batch)"
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="cell_type",
        help="Key in adata.obs for cell type labels (default: cell_type)"
    )
    parser.add_argument(
        "--embedding_key",
        type=str,
        default="X_pca",
        help="Key to use for existing embeddings (default: X_pca)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = BaselineExtractor(
        baseline_type=args.baseline_type,
        batch_key=args.batch_key,
        label_key=args.label_key,
        embedding_key=args.embedding_key,
        random_state=args.random_state,
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
