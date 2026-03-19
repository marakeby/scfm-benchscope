#!/usr/bin/env python
"""scBERT embedding extraction script.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --checkpoint_path /path/to/model.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

import logging
logger = logging.getLogger(__name__)

try:
    from scBERT.performer_pytorch.performer_pytorch import PerformerLM
    SCBERT_AVAILABLE = True
except ImportError:
    SCBERT_AVAILABLE = False
    PerformerLM = None


class scBERTExtractor(BaseExtractor):
    """scBERT embedding extractor."""
    
    def __init__(
        self,
        checkpoint_path: str = None,
        config_kwargs: dict = None,
        aggregation_method: str = "cls",
        params: dict = None,
        **kwargs
    ):
        # Handle both YAML config style and CLI style
        if params is not None:
            super().__init__(params=params)
            checkpoint_path = self.params.get('checkpoint_path', checkpoint_path)
            config_kwargs = self.params.get('config_kwargs', config_kwargs)
            aggregation_method = self.params.get('aggregation_method', aggregation_method)
        else:
            super().__init__(
                checkpoint_path=checkpoint_path,
                config_kwargs=config_kwargs,
                aggregation_method=aggregation_method,
                **kwargs
            )
        
        if not SCBERT_AVAILABLE:
            raise ImportError("scBERT package not found. Please install it.")
        
        self.checkpoint_path = checkpoint_path
        self.config_kwargs = config_kwargs or {
            'num_tokens': 7,
            'dim': 200,
            'depth': 6,
            'heads': 10,
            'max_seq_len': 16906,
            'gene2vec_path': '../data/gene2vec_16906.npy'
        }
        self.aggregation_method = aggregation_method
        self.model = None
    
    @property
    def model_name(self) -> str:
        return "scBERT"
    
    @property
    def embedding_dim(self) -> int:
        return self.config_kwargs.get('dim', 200)
    
    def load_model(self) -> None:
        """Load scBERT model from checkpoint."""
        if self._model_loaded:
            return
        
        logger.info(f"Loading scBERT model from {self.checkpoint_path}")
        
        self.model = PerformerLM(**self.config_kwargs)
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        self._model_loaded = True
        logger.info("scBERT model loaded successfully")
    
    def _preprocess(self, adata: sc.AnnData) -> sc.AnnData:
        """Preprocess data for scBERT."""
        adata = adata.copy()
        if not (hasattr(adata, 'uns') and adata.uns.get('preprocessed', False)):
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        return adata
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using scBERT.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array
        """
        if not self._model_loaded:
            self.load_model()
        
        adata = self._preprocess(adata)
        
        # Get expression matrix
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = torch.tensor(X, dtype=torch.float32)
        
        logger.info(f"Extracting scBERT embeddings for {X.shape[0]} cells")
        
        with torch.no_grad():
            outputs = self.model(X, return_encodings=True)
            token_embs = outputs
        
        # Aggregate token embeddings
        if self.aggregation_method == 'cls':
            embeddings = token_embs[:, -1, :].numpy()
        elif self.aggregation_method == 'mean':
            embeddings = token_embs.mean(dim=1).numpy()
        elif self.aggregation_method == 'sum':
            embeddings = token_embs.sum(dim=1).numpy()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        logger.info(f"Extracted scBERT embeddings: shape {embeddings.shape}")
        
        return embeddings


def main():
    parser = create_argument_parser()
    
    # scBERT-specific arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to scBERT checkpoint (.pt file)"
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="cls",
        choices=["cls", "mean", "sum"],
        help="Token aggregation method (default: cls)"
    )
    
    args = parser.parse_args()
    
    extractor = scBERTExtractor(
        checkpoint_path=args.checkpoint_path,
        aggregation_method=args.aggregation_method,
    )
    
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
