#!/usr/bin/env python
"""scFoundation embedding extraction script.

This script can be run standalone with the scFoundation environment.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_path /path/to/models.ckpt \
        --gene_index /path/to/OS_scRNA_gene_index.19264.tsv
    Or: --gene_index_path (used when invoked from run_exp subprocess)
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from tqdm import tqdm

# Add extractors parent (base_extract) and repo root (for features.*)
_extractors_dir = Path(__file__).resolve().parent.parent
_repo_root = _extractors_dir.parent
sys.path.insert(0, str(_extractors_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from base_extract import (
    BaseExtractor,
    create_argument_parser,
    run_extraction,
    normalize_gene_name_for_vocab,
)

import logging
logger = logging.getLogger(__name__)


class scFoundationExtractor(BaseExtractor):
    """scFoundation model embedding extractor."""

    def __init__(
        self,
        params: Optional[dict[str, Any]] = None,
        model_path: Optional[str] = None,
        gene_index_path: Optional[str] = None,
        ckpt_name: str = "gene",
        output_type: str = "cell",
        pool_type: str = "all",
        batch_size: int = 16,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        # Support registry/YAML config: single dict with (nested) params
        if params is not None and isinstance(params, dict):
            flat = params.get("params", params)
            model_path = model_path or flat.get("model_path")
            gene_index_path = gene_index_path or flat.get("gene_index_path")
            ckpt_name = flat.get("ckpt_name", ckpt_name)
            output_type = flat.get("output_type", output_type)
            pool_type = flat.get("pool_type", pool_type)
            batch_size = flat.get("batch_size", batch_size)
            device = flat.get("device", device)
        if not model_path or not gene_index_path:
            raise ValueError(
                "scFoundation requires model_path and gene_index_path "
                "(set in YAML params or registry default_params)."
            )
        super().__init__(
            params=params,
            model_path=model_path,
            gene_index_path=gene_index_path,
            ckpt_name=ckpt_name,
            output_type=output_type,
            pool_type=pool_type,
            batch_size=batch_size,
            device=device,
            **kwargs,
        )
        self.model_path = model_path
        self.gene_index_path = gene_index_path
        self.ckpt_name = ckpt_name
        self.output_type = output_type
        self.pool_type = pool_type
        self.batch_size = batch_size

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.gene_list = self._load_gene_list()
    
    def _load_gene_list(self) -> list:
        """Load gene list from TSV file."""
        gene_list_df = pd.read_csv(self.gene_index_path, header=0, delimiter='\t')
        return list(gene_list_df['gene_name'])
    
    @property
    def model_name(self) -> str:
        return "scFoundation"
    
    @property
    def embedding_dim(self) -> int:
        # Cell checkpoint pools 4 encoder outputs: 4 * encoder hidden_dim (e.g. 768 -> 3072)
        if getattr(self, "_embed_dim", None) is not None:
            return self._embed_dim
        return 3072  # default for cell (4 * 768)

    def load_model(self) -> None:
        """Load scFoundation model (cell checkpoint for cell embeddings)."""
        from features.scfoundation.pretrainmodels import select_model
        from features.old.load import load_model_frommmf

        # For cell embeddings use 'cell' checkpoint; for gene use 'gene'
        ckpt_key = "cell" if self.output_type == "cell" else self.ckpt_name
        logger.info(f"Loading scFoundation model from {self.model_path} (key={ckpt_key!r})")
        model_out, config = load_model_frommmf(self.model_path, ckpt_key)
        self.model = model_out
        self._config = config
        enc_dim = config.get("encoder", {}).get("hidden_dim", 768)
        self._embed_dim = 4 * enc_dim if self.output_type == "cell" and self.pool_type == "all" else enc_dim
        self.model.eval()
        self.model.to(self.device)
        logger.info("scFoundation model loaded successfully")

    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract cell embeddings using scFoundation encoder + pool (same as get_embedding.py)."""
        from features.old.load import gatherData

        # Align genes to model order and ensure (n_cells, 19264)
        X_aligned = self._align_genes(adata)
        n_cells = X_aligned.shape[0]
        pad_token_id = self._config["pad_token_id"]

        # Build input (n_cells, 19266): 19264 genes + [resolution, log10(totalcount)] per cell
        totalcounts = np.array(X_aligned.sum(axis=1), dtype=np.float32).reshape(-1, 1)
        totalcounts = np.clip(totalcounts, 1e-10, None)
        log_total = np.log10(totalcounts)
        extra = np.hstack([np.full((n_cells, 1), 4.0), log_total])  # tgthighres 't4'
        X_input = np.hstack([X_aligned, extra]).astype(np.float32)

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, n_cells, self.batch_size), desc="Extracting"):
                batch = X_input[i : i + self.batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                value_labels = batch_tensor > 0
                data_gene_ids = torch.arange(
                    batch_tensor.shape[1], device=self.device
                ).unsqueeze(0).expand(batch_tensor.shape[0], -1)

                x, x_padding = gatherData(batch_tensor, value_labels, pad_token_id)
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pad_token_id)

                x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                position_emb = self.model.pos_emb(position_gene_ids)
                x = x + position_emb
                geneemb = self.model.encoder(x, x_padding)

                seq_len = geneemb.shape[1]
                enc_dim = geneemb.shape[2]
                if self.pool_type == "all":
                    if seq_len >= 2:
                        e1 = geneemb[:, -1, :]
                        e2 = geneemb[:, -2, :]
                        e3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                        e4 = torch.mean(geneemb[:, :-2, :], dim=1)
                    elif seq_len == 1:
                        single = geneemb[:, 0, :]
                        e1 = e2 = e3 = e4 = single
                    else:
                        zero = torch.zeros(geneemb.shape[0], enc_dim, device=geneemb.device, dtype=geneemb.dtype)
                        e1 = e2 = e3 = e4 = zero
                    emb = torch.cat([e1, e2, e3, e4], dim=1)
                else:
                    emb, _ = torch.max(geneemb, dim=1)
                embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)
    
    def _align_genes(self, adata: sc.AnnData) -> np.ndarray:
        """Align adata genes to model's gene list."""
        # Get expression matrix (ensure dense: sparse .X or layer can break column assignment)
        X = adata.X
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Get gene names from adata (normalize "SYMBOL (ID)" -> "SYMBOL" for vocab match)
        adata_genes = list(adata.var_names)
        gene_to_idx = {
            normalize_gene_name_for_vocab(g): i for i, g in enumerate(adata_genes)
        }

        # Align to model's gene list (normalize model gene names for lookup)
        X_aligned = np.zeros((X.shape[0], len(self.gene_list)), dtype=np.float32)
        for i, gene in enumerate(self.gene_list):
            key = normalize_gene_name_for_vocab(gene)
            if key in gene_to_idx:
                col = X[:, gene_to_idx[key]]
                # Ensure 1D (sparse column slice can be (n, 1) or matrix)
                X_aligned[:, i] = np.asarray(col).ravel()
        
        n_matched = sum(
            normalize_gene_name_for_vocab(g) in gene_to_idx for g in self.gene_list
        )
        logger.info(f"Aligned {n_matched}/{len(self.gene_list)} genes")
        
        return X_aligned


def main():
    parser = create_argument_parser()
    
    # scFoundation-specific arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to scFoundation model checkpoint"
    )
    parser.add_argument(
        "--gene_index",
        type=str,
        default=None,
        help="Path to gene index TSV file (alias for --gene_index_path)",
    )
    parser.add_argument(
        "--gene_index_path",
        type=str,
        default=None,
        help="Path to gene index TSV file (used by run_exp subprocess)",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="gene",
        help="Checkpoint key in .ckpt file (e.g. gene, cell, rde)"
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="cell",
        choices=["cell", "gene"],
        help="Output type"
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        default="all",
        choices=["all", "max"],
        help="Pooling type for cell embeddings"
    )
    
    args = parser.parse_args()
    gene_index_path = args.gene_index_path or args.gene_index
    if not gene_index_path:
        parser.error("Either --gene_index or --gene_index_path is required")

    # Create extractor
    extractor = scFoundationExtractor(
        model_path=args.model_path,
        gene_index_path=gene_index_path,
        ckpt_name=args.ckpt_name,
        output_type=args.output_type,
        pool_type=args.pool_type,
        batch_size=args.batch_size,
        device=args.device,
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
