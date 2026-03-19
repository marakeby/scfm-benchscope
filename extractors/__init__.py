"""
Extractors module for scFM_eval.

This module provides a unified interface for all embedding extractors.

Usage:
    from extractors import get_extractor, ExtractorRegistry
    
    # Get an extractor instance
    extractor = get_extractor("pca", {"n_components": 50})
    embeddings = extractor.fit_transform(data_loader)
    
    # Or use registry directly
    from extractors.registry import ExtractorRegistry
    embeddings = ExtractorRegistry.extract_embeddings(
        model_name="pca",
        adata_path="/path/to/data.h5ad",
    )

Available models:
    Foundation Models (separate env):
        - stack, scgpt, geneformer, state, uce, aido, scconcept, scfoundation
    
    Foundation Models (base env):
        - scimilarity, laplacian_ae, cellplm, cellfm, scbert
    
    Trainable Models:
        - concord, scvi
    
    Baseline Methods:
        - pca, hvg, baseline
"""

from .registry import (
    ExtractorRegistry,
    get_extractor,
    extract_embeddings,
    MODEL_REGISTRY,
)
from .base_extract import BaseExtractor

__all__ = [
    # Main API
    "get_extractor",
    "extract_embeddings",
    "ExtractorRegistry",
    "MODEL_REGISTRY",
    # Base class
    "BaseExtractor",
]


def list_models() -> list[str]:
    """List all available model names."""
    return ExtractorRegistry.list_models()


def get_model_info(model_name: str) -> dict:
    """Get metadata for a model."""
    return ExtractorRegistry.get_model_info(model_name)
