#!/usr/bin/env python
"""scGPT embedding extraction script.

This script runs in a SEPARATE environment with scGPT dependencies.

Setup:
    cd extractors/scgpt
    python -m venv env
    source env/bin/activate
    # Install PyTorch with CUDA 12.8 support (for systems with CUDA 12.8+)
    # For CUDA 11.8, use: --index-url https://download.pytorch.org/whl/cu118
    # For CUDA 12.1, use: --index-url https://download.pytorch.org/whl/cu121
    pip install torch torchvision torchtext --index-url https://download.pytorch.org/whl/cu128
    pip install scanpy anndata numpy scipy pandas tqdm safetensors
    pip install -e ${SCGPT_REPO_PATH}   # set SCGPT_REPO_PATH in .env

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_dir /path/to/scGPT --model_name scGPT_human
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from tqdm import tqdm

# IMPORTANT: Add scGPT source directory FIRST (before extractors/) to avoid name collision.
# Set SCGPT_REPO_PATH in .env to your scGPT repo path.
SCGPT_SRC_PATH = os.environ.get("SCGPT_REPO_PATH", "")
if SCGPT_SRC_PATH and SCGPT_SRC_PATH in sys.path:
    sys.path.remove(SCGPT_SRC_PATH)
if SCGPT_SRC_PATH:
    sys.path.insert(0, SCGPT_SRC_PATH)

# Add parent to path for base module (this adds extractors/ to path, at position 1)
sys.path.insert(1, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction, normalize_gene_name_for_vocab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class scGPTExtractor(BaseExtractor):
    """scGPT model embedding extractor."""
    
    def __init__(
        self,
        model_dir: str,
        model_name: str = "scGPT_human",
        batch_size: int = 64,
        n_bins: int = 51,
        max_seq_len: int = 1200,
        device: str = "auto",
        gene_name_col: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_dir=model_dir,
            model_name=model_name,
            batch_size=batch_size,
            n_bins=n_bins,
            max_seq_len=max_seq_len,
            device=device,
            gene_name_col=gene_name_col,
            **kwargs
        )
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_bins = n_bins
        self.max_seq_len = max_seq_len
        self.gene_name_col = gene_name_col
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Will be loaded
        self.vocab = None
        self.gene_ids = None
    
    @property
    def model_name(self) -> str:
        return "scGPT"
    
    @model_name.setter
    def model_name(self, value):
        self._model_name = value
    
    @property
    def embedding_dim(self) -> int:
        return 512  # scGPT embedding dimension
    
    def _find_model_files(self):
        """Find model files in various directory structures."""
        base = self.model_dir
        name = self._model_name
        
        # Try different structures - prefer subdirectory if model_name is specified
        candidates = [
            base / name,  # model_dir/scGPT_human/
            base,  # model_dir/ (flat structure)
        ]
        
        for path in candidates:
            # Check for HuggingFace format (requires both config.json and vocab.json)
            if (path / "config.json").exists():
                vocab_path = path / "vocab.json"
                # If vocab.json doesn't exist here, check subdirectory
                if not vocab_path.exists() and name and (base / name / "vocab.json").exists():
                    vocab_path = base / name / "vocab.json"
                if vocab_path.exists():
                    return path, "huggingface"
            # Check for old format
            if (path / "args.json").exists():
                return path, "old"
        
        raise FileNotFoundError(
            f"Could not find scGPT model files in {base}. "
            f"Expected either config.json (HF format) or args.json (old format)"
        )
    
    def load_model(self) -> None:
        """Load scGPT model."""
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        
        model_path, format_type = self._find_model_files()
        logger.info(f"Loading scGPT from {model_path} (format: {format_type})")
        
        if format_type == "huggingface":
            # Load from HuggingFace format
            config_file = model_path / "config.json"
            with open(config_file) as f:
                config = json.load(f)
            
            # Load vocab
            vocab_file = model_path / "vocab.json"
            self.vocab = GeneVocab.from_file(vocab_file)
            
            # Create model
            self.model = TransformerModel(
                ntoken=len(self.vocab),
                d_model=config.get("d_model", 512),
                nhead=config.get("nhead", 8),
                d_hid=config.get("d_hid", 512),
                nlayers=config.get("nlayers", 12),
                nlayers_cls=config.get("nlayers_cls", 3),
                n_cls=1,
                vocab=self.vocab,
                dropout=config.get("dropout", 0.2),
                pad_token=config.get("pad_token", "<pad>"),
                pad_value=config.get("pad_value", -2),
                do_mvc=config.get("do_mvc", True),
                do_dab=config.get("do_dab", False),
                use_batch_labels=config.get("use_batch_labels", False),
                domain_spec_batchnorm=config.get("domain_spec_batchnorm", False),
                explicit_zero_prob=config.get("explicit_zero_prob", False),
                use_fast_transformer=config.get("use_fast_transformer", False),
            )
            
            # Load weights
            if (model_path / "model.safetensors").exists():
                from safetensors.torch import load_file
                state_dict = load_file(model_path / "model.safetensors")
            elif (model_path / "pytorch_model.bin").exists():
                state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
            else:
                raise FileNotFoundError("No model weights found")
            
            self.model.load_state_dict(state_dict, strict=False)
            
        else:
            # Old format
            args_file = model_path / "args.json"
            with open(args_file) as f:
                args = json.load(f)
            
            vocab_file = model_path / "vocab.json"
            self.vocab = GeneVocab.from_file(vocab_file)
            
            self.model = TransformerModel(
                ntoken=len(self.vocab),
                d_model=args.get("embsize", 512),
                nhead=args.get("nheads", 8),
                d_hid=args.get("d_hid", 512),
                nlayers=args.get("nlayers", 12),
                nlayers_cls=args.get("n_layers_cls", 3),
                n_cls=1,
                vocab=self.vocab,
                dropout=args.get("dropout", 0.2),
                pad_token="<pad>",
                pad_value=-2,
                do_mvc=True,
                do_dab=False,
            )
            
            model_file = model_path / "best_model.pt"
            state_dict = torch.load(model_file, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()
        self.model.to(self.device)
        
        # Build gene ID mapping (normalize to uppercase for case-insensitive matching)
        self.gene_ids = {g.upper().strip(): i for i, g in enumerate(self.vocab.get_stoi().keys())}
        
        logger.info(f"scGPT model loaded with vocab size {len(self.vocab)}")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using scGPT.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array of shape (n_cells, 512)
        """
        # Get gene names - try to use gene symbols if available
        # scGPT vocab uses gene symbols (normalized to uppercase), not Ensembl IDs
        logger.info(f"Available var columns: {list(adata.var.columns)}")
        logger.info(f"var.index sample (first 5): {list(adata.var.index[:5])}")
        
        if self.gene_name_col and self.gene_name_col in adata.var.columns:
            gene_names = adata.var[self.gene_name_col].astype(str).tolist()
            logger.info(f"Using specified gene_name_col: '{self.gene_name_col}'")
        else:
            # Try common gene symbol column names (scGPT vocab uses gene symbols, not Ensembl IDs)
            gene_symbol_cols = ['gene_symbol', 'gene_name', 'feature_name', 'Symbol']
            gene_names = None
            for col in gene_symbol_cols:
                if col in adata.var.columns:
                    gene_names = adata.var[col].astype(str).tolist()
                    logger.info(f"Using '{col}' column for gene mapping (found {len(gene_names)} genes)")
                    break
            
            # Fallback to var_names if no symbol column found
            if gene_names is None:
                gene_names = adata.var_names.tolist()
                logger.warning(
                    f"No gene symbol column found (tried: {gene_symbol_cols}). "
                    f"Using var_names - this may result in poor gene matching. "
                    f"Make sure ensure_both_gene_identifiers() was called before extraction."
                )
        
        # Normalize gene names for vocab (strip "SYMBOL (ID)" -> "SYMBOL", uppercase)
        gene_names = [normalize_gene_name_for_vocab(g, uppercase=True) for g in gene_names]
        logger.info(f"Gene names sample (first 5, normalized): {gene_names[:5]}")
        
        # Map genes to vocab indices
        gene_ids = []
        valid_gene_indices = []
        for i, gene in enumerate(gene_names):
            if gene in self.gene_ids:
                gene_ids.append(self.gene_ids[gene])
                valid_gene_indices.append(i)
        
        logger.info(f"Mapped {len(gene_ids)}/{len(gene_names)} genes to vocab")
        
        if len(gene_ids) == 0:
            raise ValueError("No genes matched the vocabulary!")
        
        # Get expression matrix (keep sparse for memory efficiency)
        # Handle backed mode by loading all data first
        X = adata.X[:]  # This loads data from backed mode
        
        # Subset to valid genes (works on both sparse and dense)
        X = X[:, valid_gene_indices]
        gene_ids = np.array(gene_ids)
        
        is_sparse = issparse(X)
        logger.info(f"Expression matrix shape: {X.shape}, sparse: {is_sparse}")
        
        # Use scGPT's tokenize_and_pad_batch for proper tokenization
        from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch
        
        # Extract embeddings in batches
        embeddings = []
        n_cells = X.shape[0]
        pad_token = "<pad>"
        pad_value = -2  # scGPT default pad value
        
        logger.info(f"Extracting embeddings with max_seq_len={self.max_seq_len}")
        
        with torch.no_grad():
            for i in tqdm(range(0, n_cells, self.batch_size), desc="Extracting"):
                batch_X = X[i:i + self.batch_size]
                
                # Convert batch to dense if sparse
                if issparse(batch_X):
                    batch_X = batch_X.toarray()
                batch_X = np.asarray(batch_X, dtype=np.float32)
                
                # Use scGPT's tokenization (keeps only non-zero genes, truncates to max_seq_len)
                tokenized = tokenize_and_pad_batch(
                    batch_X,
                    gene_ids,
                    max_len=self.max_seq_len,
                    vocab=self.vocab,
                    pad_token=pad_token,
                    pad_value=pad_value,
                    append_cls=True,
                    include_zero_gene=False,
                )
                
                genes_tensor = tokenized["genes"].to(self.device)
                values_tensor = tokenized["values"].to(self.device)
                
                # Create padding mask (True where padded)
                src_key_padding_mask = genes_tensor.eq(self.vocab[pad_token])
                
                # Get embeddings
                try:
                    output = self.model._encode(
                        genes_tensor,
                        values_tensor,
                        src_key_padding_mask=src_key_padding_mask,
                    )
                    # Average pool over non-padded genes for cell embedding
                    mask = ~src_key_padding_mask
                    mask_expanded = mask.unsqueeze(-1).float()
                    cell_emb = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                except Exception as e:
                    logger.warning(f"Using encode_batch fallback: {e}")
                    # Fallback: use encode_batch if available
                    if hasattr(self.model, 'encode_batch'):
                        cell_emb = self.model.encode_batch(
                            genes_tensor,
                            values_tensor,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_size=genes_tensor.shape[0],
                        )
                        if isinstance(cell_emb, np.ndarray):
                            cell_emb = torch.from_numpy(cell_emb)
                    else:
                        raise RuntimeError(f"Failed to extract embeddings: {e}")
                
                embeddings.append(cell_emb.cpu().numpy())
        
        return np.vstack(embeddings)


def main():
    parser = create_argument_parser()
    
    # scGPT-specific arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to scGPT model directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="scGPT_human",
        help="Model name within directory"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=51,
        help="Number of bins for expression binning"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1200,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = scGPTExtractor(
        model_dir=args.model_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
        max_seq_len=args.max_seq_len,
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
