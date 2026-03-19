#!/usr/bin/env python
"""Geneformer embedding extraction script.

This script runs in a SEPARATE environment with Geneformer dependencies.

Setup:
    cd extractors/geneformer
    python -m venv env
    source env/bin/activate
    pip install torch>=2.0.0
    pip install transformers>=4.30.0,<4.52.0 "huggingface-hub>=0.14.1,<1.0"
    pip install scanpy anndata numpy scipy pandas tqdm datasets
    pip install -e ${GENEFORMER_REPO_PATH}   # set GENEFORMER_REPO_PATH in .env

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_dir /path/to/Geneformer --model_name Geneformer-V2-104M
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch
from tqdm import trange

# Add parent to path for base module (extractors/)
_extractors_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_extractors_dir))

# Ensure the real Geneformer package is found (set GENEFORMER_REPO_PATH in .env)
_GENEFORMER_REPO = os.environ.get("GENEFORMER_REPO_PATH")
if _GENEFORMER_REPO and Path(_GENEFORMER_REPO).exists():
    sys.path.insert(0, _GENEFORMER_REPO)
from base_extract import (
    BaseExtractor,
    create_argument_parser,
    run_extraction,
    normalize_adata_var_gene_names,
    normalize_gene_name_for_vocab,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeneformerExtractor(BaseExtractor):
    """Geneformer model embedding extractor."""
    
    def __init__(
        self,
        model_dir: str,
        model_name: str = "Geneformer-V2-104M",
        batch_size: int = 32,
        max_input_size: int = 2048,
        device: str = "auto",
        gene_name_col: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_dir=model_dir,
            model_name=model_name,
            batch_size=batch_size,
            max_input_size=max_input_size,
            device=device,
            gene_name_col=gene_name_col,
            **kwargs
        )
        self.model_dir = Path(model_dir)
        self._model_name_str = model_name
        self.batch_size = batch_size
        self.max_input_size = max_input_size
        self.gene_name_col = gene_name_col
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Find actual model directory
        self._find_model_dir()
    
    def _find_model_dir(self):
        """Find the actual model directory with config.json."""
        candidates = [
            self.model_dir / self._model_name_str,  # model_dir/Geneformer-V2-104M/
            self.model_dir / self._model_name_str / self._model_name_str,  # nested
            self.model_dir,  # flat
        ]
        
        for path in candidates:
            if (path / "config.json").exists():
                self.model_path = path
                logger.info(f"Found Geneformer model at: {self.model_path}")
                return
        
        raise FileNotFoundError(
            f"Could not find Geneformer config.json in {self.model_dir}"
        )
    
    @property
    def model_name(self) -> str:
        return "Geneformer"
    
    @property
    def embedding_dim(self) -> int:
        return 512  # Default Geneformer embedding dimension
    
    def load_model(self) -> None:
        """Load Geneformer model."""
        from transformers import BertForMaskedLM
        
        logger.info(f"Loading Geneformer from {self.model_path}")
        
        self.model = BertForMaskedLM.from_pretrained(
            self.model_path,
            output_hidden_states=True,
            output_attentions=False,
        )
        self.model.eval()
        self.model.to(self.device)
        
        logger.info("Geneformer model loaded successfully")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using Geneformer.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array of shape (n_cells, embedding_dim)
        """
        from datasets import Dataset

        # Load into memory if backed (e.g. from run_extraction temp h5ad)
        if getattr(adata, "isbacked", False):
            adata = adata.to_memory()
        # Prepare adata once (normalize gene names)
        adata_copy = adata.copy()
        normalize_adata_var_gene_names(adata_copy)
        if self.gene_name_col and self.gene_name_col in adata_copy.var.columns:
            adata_copy.var_names = [
                normalize_gene_name_for_vocab(s)
                for s in adata_copy.var[self.gene_name_col].astype(str)
            ]

        tokenized_dataset = None
        try:
            from geneformer import TranscriptomeTokenizer
            logger.info("Tokenizing data (TranscriptomeTokenizer)...")
            with tempfile.TemporaryDirectory() as tmpdir:
                h5ad_path = Path(tmpdir) / "data.h5ad"
                adata_copy.write_h5ad(h5ad_path)
                tokenizer = TranscriptomeTokenizer(custom_attr_name_dict=None, nproc=4)
                tokenized_path = Path(tmpdir) / "tokenized"
                tokenized_path.mkdir()
                tokenizer.tokenize_data(
                    str(h5ad_path.parent), str(tokenized_path), file_format="h5ad",
                )
                tokenized_dataset = Dataset.load_from_disk(str(tokenized_path / "data.dataset"))
        except Exception as e:
            logger.warning(f"TranscriptomeTokenizer failed: {e}. Using manual tokenization.")
            tokenized_dataset = self._manual_tokenize(adata_copy)

        logger.info("Extracting embeddings...")
        embeddings = self._extract_from_dataset(tokenized_dataset)
        return embeddings
    
    def _manual_tokenize(self, adata: sc.AnnData):
        """Manual tokenization fallback (no geneformer package import)."""
        import pickle
        from datasets import Dataset

        # Resolve token dictionary from repo path so we never import geneformer
        dict_path = Path(__file__).parent.parent.parent / "features" / "geneformer" / "token_dictionary.pkl"
        if not dict_path.exists():
            dict_path = _GENEFORMER_REPO / "geneformer" / "token_dictionary_gc104M.pkl"
        if not dict_path.exists():
            try:
                from geneformer import perturber_utils
                dict_path = Path(perturber_utils.__file__).parent / "token_dictionary.pkl"
            except Exception:
                pass
        if not dict_path.exists():
            raise FileNotFoundError(
                "Geneformer token dictionary not found. Tried "
                f"features/geneformer/token_dictionary.pkl and "
                f"{_GENEFORMER_REPO / 'geneformer' / 'token_dictionary_gc104M.pkl'}"
            )
        with open(dict_path, "rb") as f:
            token_dict = pickle.load(f)

        # Rank genes by expression and take top genes per cell
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        
        # Tokenize each cell
        tokenized_cells = []
        for i in range(X.shape[0]):
            cell_expr = X[i]
            # Get non-zero genes sorted by expression
            nonzero_idx = np.where(cell_expr > 0)[0]
            if len(nonzero_idx) == 0:
                tokenized_cells.append({"input_ids": [0], "length": 1})
                continue
            
            sorted_idx = nonzero_idx[np.argsort(-cell_expr[nonzero_idx])]
            
            # Map to tokens
            tokens = []
            for idx in sorted_idx[:self.max_input_size]:
                gene = adata.var_names[idx]
                if gene in token_dict:
                    tokens.append(token_dict[gene])
            
            if len(tokens) == 0:
                tokens = [0]
            
            tokenized_cells.append({
                "input_ids": tokens,
                "length": len(tokens),
            })
        
        return Dataset.from_list(tokenized_cells)
    
    def _extract_from_dataset(self, dataset) -> np.ndarray:
        """Extract embeddings from tokenized dataset."""
        embeddings = []
        
        for i in trange(0, len(dataset), self.batch_size, desc="Extracting"):
            batch = dataset[i:i + self.batch_size]
            
            # Pad batch
            input_ids = batch["input_ids"]
            max_len = max(len(x) for x in input_ids)
            
            padded_ids = []
            attention_mask = []
            for ids in input_ids:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [0] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
            
            # Convert to tensors
            input_tensor = torch.tensor(padded_ids, dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    attention_mask=mask_tensor,
                )
                
                # Get CLS token embedding or mean pool
                hidden_states = outputs.hidden_states[-1]  # Last layer
                
                # Mean pool over non-padded tokens
                mask_expanded = mask_tensor.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                cell_emb = sum_embeddings / sum_mask
                
                embeddings.append(cell_emb.cpu().numpy())
        
        return np.vstack(embeddings)


def main():
    parser = create_argument_parser()
    
    # Geneformer-specific arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to Geneformer model directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Geneformer-V2-104M",
        help="Model name within directory"
    )
    parser.add_argument(
        "--max_input_size",
        type=int,
        default=2048,
        help="Maximum number of genes per cell"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = GeneformerExtractor(
        model_dir=args.model_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_input_size=args.max_input_size,
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
