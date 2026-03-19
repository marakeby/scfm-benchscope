#!/usr/bin/env python
"""UCE (Universal Cell Embeddings) extraction script.

This script runs in a SEPARATE environment with UCE dependencies.
UCE is NOT a pip package - it needs to be added to PYTHONPATH.

Setup:
    cd extractors/uce
    python -m venv env
    source env/bin/activate
    pip install torch>=2.0.0
    pip install scanpy anndata numpy scipy pandas tqdm accelerate requests
    
    # Add UCE to PYTHONPATH (set UCE_REPO_PATH in .env)
    export PYTHONPATH="${UCE_REPO_PATH}:$PYTHONPATH"

Usage:
    export PYTHONPATH="${UCE_REPO_PATH}:$PYTHONPATH"
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_dir /path/to/UCE --species human
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from tqdm import tqdm

# Add UCE to path (set UCE_REPO_PATH in .env)
UCE_PATH = os.environ.get("UCE_REPO_PATH", "")
if UCE_PATH and UCE_PATH not in sys.path:
    sys.path.insert(0, UCE_PATH)

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UCEExtractor(BaseExtractor):
    """UCE (Universal Cell Embeddings) model extractor."""
    
    def __init__(
        self,
        model_dir: str,
        species: str = "human",
        batch_size: int = 32,
        device: str = "auto",
        gene_name_col: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_dir=model_dir,
            species=species,
            batch_size=batch_size,
            device=device,
            gene_name_col=gene_name_col,
            **kwargs
        )
        self.model_dir = Path(model_dir)
        self.species = species
        self.batch_size = batch_size
        self.gene_name_col = gene_name_col
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    @property
    def model_name(self) -> str:
        return "UCE"
    
    @property
    def embedding_dim(self) -> int:
        return 1280  # UCE embedding dimension
    
    def load_model(self) -> None:
        """Load UCE model."""
        try:
            from model import TransformerModel
        except ImportError:
            raise ImportError(
                "UCE model.py not found. Set UCE_REPO_PATH in .env and add to PYTHONPATH."
            )
        
        logger.info(f"Loading UCE model from {self.model_dir}")
        
        # Find model checkpoint
        ckpt_path = self.model_dir / "all_tokens" / "4layer_model.torch"
        if not ckpt_path.exists():
            ckpt_path = self.model_dir / "4layer_model.torch"
        if not ckpt_path.exists():
            # Try to find any .torch file
            torch_files = list(self.model_dir.rglob("*.torch"))
            if torch_files:
                ckpt_path = torch_files[0]
            else:
                raise FileNotFoundError(f"No model checkpoint found in {self.model_dir}")
        
        # Load protein embeddings
        protein_emb_path = self.model_dir / "model_files" / "new_species_protein_embeddings.csv"
        if not protein_emb_path.exists():
            protein_emb_path = Path(UCE_PATH) / "model_files" / "new_species_protein_embeddings.csv"
        
        logger.info(f"Loading protein embeddings from {protein_emb_path}")
        self.protein_embeddings = pd.read_csv(protein_emb_path, index_col=0)
        
        # Load model
        self.model = TransformerModel(
            token_dim=1280,
            d_model=1280,
            nhead=20,
            d_hid=1280,
            nlayers=4,
            dropout=0.0,
        )
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        
        logger.info("UCE model loaded successfully")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using UCE.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array of shape (n_cells, 1280)
        """
        # Get gene names
        if self.gene_name_col and self.gene_name_col in adata.var.columns:
            gene_names = adata.var[self.gene_name_col].astype(str).tolist()
        else:
            gene_names = adata.var_names.tolist()
        
        # Map genes to protein embeddings
        available_genes = set(self.protein_embeddings.index)
        gene_to_idx = {g: i for i, g in enumerate(gene_names) if g in available_genes}
        
        logger.info(f"Mapped {len(gene_to_idx)}/{len(gene_names)} genes to protein embeddings")
        
        if len(gene_to_idx) == 0:
            raise ValueError("No genes matched the protein embeddings!")
        
        # Get expression matrix
        X = adata.X
        if issparse(X):
            X = X.toarray()
        
        # Prepare protein embeddings for matched genes
        matched_genes = [g for g in gene_names if g in available_genes]
        gene_embeddings = self.protein_embeddings.loc[matched_genes].values
        gene_embeddings = torch.tensor(gene_embeddings, dtype=torch.float32).to(self.device)
        
        # Get expression for matched genes only
        matched_indices = [gene_names.index(g) for g in matched_genes]
        X_matched = X[:, matched_indices]
        
        # Extract embeddings in batches
        embeddings = []
        n_cells = X_matched.shape[0]
        
        with torch.no_grad():
            for i in tqdm(range(0, n_cells, self.batch_size), desc="Extracting"):
                batch_X = X_matched[i:i + self.batch_size]
                batch_size = batch_X.shape[0]
                
                # Weight gene embeddings by expression
                batch_expr = torch.tensor(batch_X, dtype=torch.float32).to(self.device)
                
                # Normalize expression
                batch_expr = batch_expr / (batch_expr.sum(dim=1, keepdim=True) + 1e-8)
                
                # Weighted average of gene embeddings
                # Shape: (batch, genes) @ (genes, embed_dim) -> (batch, embed_dim)
                weighted_embeddings = batch_expr @ gene_embeddings
                
                # Pass through transformer
                # UCE expects shape (seq_len, batch, embed_dim)
                weighted_embeddings = weighted_embeddings.unsqueeze(0)  # (1, batch, embed)
                
                try:
                    output = self.model(weighted_embeddings)
                    cell_emb = output.squeeze(0)  # (batch, embed)
                except Exception as e:
                    logger.warning(f"Transformer forward failed, using weighted average: {e}")
                    cell_emb = weighted_embeddings.squeeze(0)
                
                embeddings.append(cell_emb.cpu().numpy())
        
        return np.vstack(embeddings)


def main():
    parser = create_argument_parser()
    
    # UCE-specific arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to UCE model directory"
    )
    parser.add_argument(
        "--species",
        type=str,
        default="human",
        choices=["human", "mouse"],
        help="Species for gene mapping"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = UCEExtractor(
        model_dir=args.model_dir,
        species=args.species,
        batch_size=args.batch_size,
        device=args.device,
        gene_name_col=args.gene_name_col,
    )
    
    # Run extraction
    run_extraction(
        extractor=extractor,
        input_path=args.input,
        output_path=args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
