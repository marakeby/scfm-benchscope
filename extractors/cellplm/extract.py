#!/usr/bin/env python
"""CellPLM embedding extraction script.

Extract single-cell embeddings using a pretrained CellPLM model.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --pretrain_prefix CellPLM --pretrain_directory /path/to/CellPLM
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

import logging
logger = logging.getLogger(__name__)

try:
    from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
    CELLPLM_AVAILABLE = True
except ImportError:
    CELLPLM_AVAILABLE = False
    CellEmbeddingPipeline = None


class CellPLMExtractor(BaseExtractor):
    """CellPLM embedding extractor.
    
    Steps:
    1. Preprocess AnnData (normalization)
    2. Load pretrained CellPLM via CellEmbeddingPipeline
    3. Use pipeline.predict to generate cell embeddings
    """
    
    def __init__(
        self,
        pretrain_prefix: str = "CellPLM",
        pretrain_directory: str = None,
        device: str = "cuda",
        params: dict = None,
        **kwargs
    ):
        # Handle both YAML config style and CLI style
        if params is not None:
            super().__init__(params=params)
            pretrain_prefix = self.params.get('pretrain_prefix', pretrain_prefix)
            pretrain_directory = self.params.get('pretrain_directory', pretrain_directory)
            device = self.params.get('device', device)
        else:
            super().__init__(
                pretrain_prefix=pretrain_prefix,
                pretrain_directory=pretrain_directory,
                device=device,
                **kwargs
            )
        
        if not CELLPLM_AVAILABLE:
            raise ImportError("CellPLM package not found. Please install it.")
        
        self.pretrain_prefix = pretrain_prefix
        self.pretrain_directory = pretrain_directory
        self.device_str = device
        
        # Initialize pipeline
        logger.info(f"Initializing CellPLM pipeline from {pretrain_directory}")
        self.pipeline = CellEmbeddingPipeline(
            pretrain_prefix=pretrain_prefix,
            pretrain_directory=pretrain_directory
        )
        
        logger.info(f'CellPLMExtractor initialized: pretrain_prefix={pretrain_prefix}, pretrain_directory={pretrain_directory}')
    
    @property
    def model_name(self) -> str:
        return "CellPLM"
    
    @property
    def embedding_dim(self) -> int:
        return -1  # Unknown until extraction
    
    def load_model(self) -> None:
        """CellPLM model is loaded in __init__ via pipeline."""
        logger.info("CellPLM extractor ready")
        self._model_loaded = True
    
    def _preprocess(self, adata: sc.AnnData) -> sc.AnnData:
        """Preprocess data for CellPLM."""
        adata = adata.copy()
        # CellPLM expects TP10K normalized data (no log1p)
        if not (hasattr(adata, 'uns') and adata.uns.get('preprocessed', False)):
            sc.pp.normalize_total(adata, target_sum=1e4)
        return adata
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using CellPLM.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array
        """
        adata = self._preprocess(adata)
        
        logger.info(f"Extracting CellPLM embeddings for {adata.n_obs} cells")
        
        embedding = self.pipeline.predict(
            adata,
            device=self.device_str
        )
        
        embeddings = embedding.cpu().numpy()
        
        # Store in adata.obsm
        adata.obsm['X_CellPLM'] = embeddings
        
        logger.info(f"Extracted CellPLM embeddings. Shape: {embeddings.shape}")
        
        return embeddings


def main():
    parser = create_argument_parser()
    
    # CellPLM-specific arguments
    parser.add_argument(
        "--pretrain_prefix",
        type=str,
        default="CellPLM",
        help="Prefix identifying checkpoint"
    )
    parser.add_argument(
        "--pretrain_directory",
        type=str,
        required=True,
        help="Directory holding pretrained checkpoints"
    )
    
    args = parser.parse_args()
    
    extractor = CellPLMExtractor(
        pretrain_prefix=args.pretrain_prefix,
        pretrain_directory=args.pretrain_directory,
        device=args.device,
    )
    
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
