#!/usr/bin/env python
"""Base extraction script template.

All model-specific extractors should follow this interface.
"""

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_gene_name_for_vocab(gene_name: str, uppercase: bool = False) -> str:
    """Strip parenthetical suffix (e.g. 'TSPAN6 (7105)' -> 'TSPAN6') for vocab matching.

    Many pipelines (e.g. mygene) store symbols as 'SYMBOL (ENTREZ_ID)'.
    Model vocabs typically expect plain symbols.

    Args:
        gene_name: Raw gene identifier from var.index or gene_symbol column.
        uppercase: If True, return uppercase (for case-insensitive vocab match).

    Returns:
        Normalized symbol suitable for vocab / symbol lookup.
    """
    s = str(gene_name).strip()
    if " (" in s and s.endswith(")"):
        s = s.split(" (", 1)[0].strip()
    return s.upper() if uppercase else s


def normalize_adata_var_gene_names(adata: sc.AnnData) -> None:
    """Normalize gene names in adata.var in place (strip 'SYMBOL (ID)' -> 'SYMBOL').

    Updates var.index and, if present, 'gene_symbol' / 'gene_symbols' columns.
    Call this at the start of extract_embeddings when the model expects plain symbols.
    """
    import pandas as pd
    # Normalize var.index
    adata.var.index = pd.Index(
        [normalize_gene_name_for_vocab(str(i)) for i in adata.var.index],
        name=adata.var.index.name,
    )
    for col in ("gene_symbol", "gene_symbols", "feature_name"):
        if col in adata.var.columns:
            adata.var[col] = adata.var[col].astype(str).apply(normalize_gene_name_for_vocab)
    if adata.raw is not None and hasattr(adata.raw, "var"):
        raw_var = adata.raw.var
        raw_var.index = pd.Index(
            [normalize_gene_name_for_vocab(str(i)) for i in raw_var.index],
            name=raw_var.index.name,
        )
        for col in ("gene_symbol", "gene_symbols", "feature_name"):
            if col in raw_var.columns:
                adata.raw.var[col] = raw_var[col].astype(str).apply(normalize_gene_name_for_vocab)


class BaseExtractor(ABC):
    """Base class for all embedding extractors.
    
    Supports two initialization styles:
    1. YAML config style: __init__(params={'method': '...', 'params': {...}})
       - Used by run_exp.py
    2. CLI style: __init__(**kwargs)
       - Used by run_extractor.py and extract.py scripts
    
    Also supports both interfaces:
    1. EmbeddingExtractor interface: fit_transform(data_loader)
       - Used by run_exp.py
    2. BaseExtractor interface: load_model() + extract_embeddings(adata)
       - Used by run_extractor.py
    """
    
    def __init__(self, params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize extractor with model-specific parameters.
        
        Args:
            params: YAML config format dict with 'method' and 'params' keys
            **kwargs: Direct parameter assignment (CLI style)
        """
        # Support both initialization styles
        if params is not None:
            # YAML config style: params = {'method': '...', 'params': {...}}
            self.method = params.get('method', 'unknown')
            self.params = params.get('params', {})
            # Merge any top-level params that aren't in nested params
            for key, value in params.items():
                if key not in ['method', 'params']:
                    self.params[key] = value
        else:
            # CLI style: direct kwargs
            self.params = kwargs
            self.method = kwargs.get('method', 'unknown')
        
        self.model = None
        self._model_loaded = False
        
    def load_model(self) -> None:
        """Load the pre-trained model.
        
        Default implementation does nothing. Override in subclasses that need model loading.
        For models that train on data (like scVI, CONCORD), this can be a no-op.
        """
        if not self._model_loaded:
            logger.info(f"{self.model_name} extractor ready")
            self._model_loaded = True
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings from AnnData object.
        
        Default implementation raises NotImplementedError. Subclasses should either:
        1. Override this method, OR
        2. Override fit_transform() (which will be called by BaseExtractor.fit_transform)
        
        Args:
            adata: AnnData object with expression data
            
        Returns:
            Embeddings array of shape (n_cells, embedding_dim)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either extract_embeddings() or fit_transform()"
        )
    
    def fit_transform(self, data_loader) -> np.ndarray:
        """Extract embeddings using EmbeddingExtractor interface.
        
        This method bridges the EmbeddingExtractor interface (used by run_exp.py)
        with the BaseExtractor interface (used by run_extractor.py).
        
        If a subclass overrides fit_transform() directly, that will be used.
        Otherwise, this default implementation calls load_model() and extract_embeddings().
        
        Args:
            data_loader: DataLoader object with .adata attribute
            
        Returns:
            Embeddings array of shape (n_cells, embedding_dim)
        """
        # Check if subclass has overridden fit_transform
        # If so, call the parent's fit_transform (which will call the subclass version)
        # Actually, if subclass overrides, Python will call the subclass version directly
        
        # Get AnnData from data_loader
        adata = data_loader.adata
        
        # Load model if not already loaded
        if not self._model_loaded:
            self.load_model()
            self._model_loaded = True
        
        # Try to extract embeddings using extract_embeddings()
        # If it raises NotImplementedError, the subclass must have overridden fit_transform
        try:
            embeddings = self.extract_embeddings(adata)
            return embeddings
        except NotImplementedError:
            # Subclass must have overridden fit_transform, but we're being called
            # This shouldn't happen, but handle gracefully
            raise RuntimeError(
                f"{self.__class__.__name__} must implement either extract_embeddings() "
                "or override fit_transform() completely"
            )
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.method
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (if known before extraction)."""
        return -1  # Unknown until extraction
    
    @staticmethod
    def validate_config(params: dict[str, Any]) -> None:
        """Validate configuration parameters.
        
        Args:
            params: Configuration parameters
            
        Raises:
            AssertionError: If required parameters are missing
        """
        assert 'method' in params, "Missing required parameter: 'method'"


def create_argument_parser() -> argparse.ArgumentParser:
    """Create base argument parser with common arguments."""
    parser = argparse.ArgumentParser(
        description="Extract cell embeddings from single-cell data"
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input AnnData file (.h5ad)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output embeddings file (.npy)"
    )
    
    # Optional data arguments
    parser.add_argument(
        "--label_key",
        type=str,
        default=None,
        help="Key in adata.obs for cell labels"
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default=None,
        help="Key in adata.obs for batch information"
    )
    parser.add_argument(
        "--gene_name_col",
        type=str,
        default=None,
        help="Column in adata.var containing gene names (if not var_names)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    
    # Output arguments
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="Save metadata JSON alongside embeddings"
    )
    
    return parser


def run_extraction(
    extractor: BaseExtractor,
    input_path: str,
    output_path: str,
    save_metadata: bool = True,
    **kwargs: Any
) -> None:
    """Run the extraction pipeline.
    
    Args:
        extractor: Initialized extractor instance
        input_path: Path to input h5ad file
        output_path: Path to output npy file
        save_metadata: Whether to save metadata JSON
        **kwargs: Additional arguments passed to extractor
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Store original input path in extractor (for models that need file paths)
    extractor.input_path = str(input_path)
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    adata = sc.read_h5ad(input_path, backed='r')  # Use backed mode to keep file reference
    # Set filename so extractors can use original path
    adata.filename = str(input_path)
    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Load model
    logger.info("Loading model...")
    extractor.load_model()
    logger.info("Model loaded successfully")
    
    # Extract embeddings
    logger.info("Extracting embeddings...")
    embeddings = extractor.extract_embeddings(adata)
    logger.info(f"Extracted embeddings with shape {embeddings.shape}")
    
    # Save embeddings
    logger.info(f"Saving embeddings to {output_path}")
    np.save(output_path, embeddings)
    
    # Save metadata
    if save_metadata:
        metadata_path = output_path.with_suffix(".json")
        metadata = {
            "model_name": extractor.model_name,
            "input_file": str(input_path),
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "parameters": extractor.params,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("Extraction complete!")


if __name__ == "__main__":
    print("This is a base module. Use model-specific extract.py scripts.")
    sys.exit(1)
