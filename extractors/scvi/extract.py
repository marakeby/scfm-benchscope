#!/usr/bin/env python
"""scVI embedding extraction script.

This script can be run standalone in the base environment.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --batch_key batch --n_layers 2 --n_latent 10
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import scvi

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

import logging
logger = logging.getLogger(__name__)

# Set seed for reproducibility
scvi.settings.seed = 9627


class scVIExtractor(BaseExtractor):
    """scVI embedding extractor."""
    
    def __init__(
        self,
        batch_key: str = None,
        n_layers: int = 2,
        n_latent: int = 10,
        gene_likelihood: str = "zinb",
        max_epochs: int = 400,
        batch_size: int = 128,
        layer_name: str | None = None,
        hvg_params: dict | None = None,
        params: dict = None,
        **kwargs
    ):
        # Handle both YAML config style and CLI style
        if params is not None:
            super().__init__(params=params)
            batch_key = self.params.get('batch_key', batch_key)
            n_layers = self.params.get('n_layers', n_layers)
            n_latent = self.params.get('n_latent', n_latent)
            gene_likelihood = self.params.get('gene_likelihood', gene_likelihood)
            max_epochs = self.params.get('max_epochs', max_epochs)
            batch_size = self.params.get('batch_size', batch_size)
            layer_name = self.params.get('layer_name', layer_name)
            hvg_params = self.params.get('hvg_params', hvg_params)
        else:
            super().__init__(
                batch_key=batch_key,
                n_layers=n_layers,
                n_latent=n_latent,
                gene_likelihood=gene_likelihood,
                max_epochs=max_epochs,
                batch_size=batch_size,
                layer_name=layer_name,
                hvg_params=hvg_params,
                **kwargs
            )
        
        if not batch_key:
            raise ValueError("Missing required parameter 'batch_key'")
        
        self.batch_key = batch_key
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.gene_likelihood = gene_likelihood
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.layer_name = layer_name
        self.hvg_params = hvg_params
        
        logger.info(f'scVIEmbeddingExtractor initialized: batch_key={batch_key}, n_layers={n_layers}, n_latent={n_latent}')
    
    @property
    def model_name(self) -> str:
        return "scVI"
    
    @property
    def embedding_dim(self) -> int:
        return self.n_latent
    
    def load_model(self) -> None:
        """scVI doesn't require pre-loading (trains on data)."""
        logger.info("scVI extractor ready (will train on data)")
        self._model_loaded = True
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using scVI.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array of shape (n_cells, n_latent)
        """
        adata = adata.copy()
        
        # Optionally select highly variable genes
        if self.hvg_params:
            n_top_genes = self.hvg_params.get('n_top_genes', 2000)
            flavor = self.hvg_params.get('flavor', 'seurat')
            logger.info(f"Selecting top {n_top_genes} highly variable genes...")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor, subset=True)
        
        # Setup AnnData for scVI
        scvi.model.SCVI.setup_anndata(adata, layer=self.layer_name, batch_key=self.batch_key)
        
        # Create and train model
        logger.info(f"Training scVI model (n_layers={self.n_layers}, n_latent={self.n_latent})...")
        model = scvi.model.SCVI(
            adata,
            n_layers=self.n_layers,
            n_latent=self.n_latent,
            gene_likelihood=self.gene_likelihood
        )
        
        logger.info(f"Training for {self.max_epochs} epochs...")
        model.train(max_epochs=self.max_epochs, batch_size=self.batch_size)
        
        # Extract latent representation
        logger.info("Extracting embeddings...")
        embeddings = model.get_latent_representation()
        
        # Store in adata.obsm
        adata.obsm["X_scVI"] = embeddings
        
        logger.info(f"Extracted scVI embeddings. Shape: {embeddings.shape}")
        
        return embeddings


def main():
    parser = create_argument_parser()
    
    # scVI-specific arguments
    parser.add_argument(
        "--batch_key",
        type=str,
        required=True,
        help="Key in adata.obs for batch information"
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2)"
    )
    parser.add_argument(
        "--n_latent",
        type=int,
        default=10,
        help="Latent dimension (default: 10)"
    )
    parser.add_argument(
        "--gene_likelihood",
        type=str,
        default="zinb",
        choices=["zinb", "nb", "poisson"],
        help="Gene likelihood distribution (default: zinb)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=400,
        help="Maximum training epochs (default: 400)"
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default=None,
        help="Layer name in adata.layers to use (default: None, uses adata.X)"
    )
    
    args = parser.parse_args()
    
    extractor = scVIExtractor(
        batch_key=args.batch_key,
        n_layers=args.n_layers,
        n_latent=args.n_latent,
        gene_likelihood=args.gene_likelihood,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        layer_name=args.layer_name,
    )
    
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
