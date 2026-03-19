#!/usr/bin/env python
"""STACK embedding extraction script.

This script can be run standalone with the STACK environment.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --checkpoint_path /path/to/bc_large_aligned.ckpt \
        --genelist_path /path/to/basecount_1000per_15000max.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction, normalize_adata_var_gene_names

import logging
logger = logging.getLogger(__name__)


class STACKExtractor(BaseExtractor):
    """STACK model embedding extractor."""

    def __init__(
        self,
        checkpoint_path: str | dict | None = None,
        genelist_path: str | None = None,
        batch_size: int = 32,
        device: str = "auto",
        gene_name_col: str | None = None,
        **kwargs: object,
    ) -> None:
        # Registry style: single params dict as first argument (from scFM_eval get_extractor)
        if isinstance(checkpoint_path, dict) and genelist_path is None:
            params = checkpoint_path
            checkpoint_path = params.get("checkpoint_path")
            genelist_path = params.get("genelist_path")
            batch_size = params.get("batch_size", batch_size)
            device = params.get("device", device)
            gene_name_col = params.get("gene_name_col", gene_name_col)
            extra = {
                k: v
                for k, v in params.items()
                if k
                not in (
                    "checkpoint_path",
                    "genelist_path",
                    "batch_size",
                    "device",
                    "gene_name_col",
                    "method",
                )
            }
            kwargs = {**extra, **kwargs}

        if checkpoint_path is None or genelist_path is None:
            raise ValueError(
                "STACKExtractor requires checkpoint_path and genelist_path "
                "(pass them as keyword args or inside a params dict)."
            )

        super().__init__(
            params=None,
            checkpoint_path=checkpoint_path,
            genelist_path=genelist_path,
            batch_size=batch_size,
            device=device,
            gene_name_col=gene_name_col,
            **kwargs,
        )
        self.checkpoint_path = checkpoint_path
        self.genelist_path = genelist_path
        self.batch_size = batch_size
        self.gene_name_col = gene_name_col

        # Determine device
        import torch

        if device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
        self.device = torch.device(self.device_str)
    
    @property
    def model_name(self) -> str:
        return "STACK"
    
    @property
    def embedding_dim(self) -> int:
        return 1600  # STACK embedding dimension
    
    def _ensure_stack_on_path(self) -> None:
        """Prepend Stack repo src to sys.path so ``stack.model_loading`` resolves to the
        real Stack package rather than the ``extractors/stack/`` directory inside scFM_eval."""
        import importlib
        import os

        # Always try to inject the repo path first -- the extractors/stack/ dir shadows
        # the real stack package, so we must put the repo ahead of it on sys.path.
        repo_root = os.environ.get("STACK_REPO_ROOT")
        if repo_root:
            src_dir = Path(repo_root) / "src"
            if src_dir.is_dir() and (src_dir / "stack").is_dir():
                src_str = str(src_dir)
                if src_str not in sys.path:
                    sys.path.insert(0, src_str)
                    # Force Python to re-discover the 'stack' package from the new path
                    if "stack" in sys.modules:
                        del sys.modules["stack"]
                    importlib.invalidate_caches()

        # Verify the import works
        try:
            from stack.model_loading import load_model_from_checkpoint  # noqa: F401
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(
                "Cannot import 'stack.model_loading'. "
                "Set STACK_REPO_ROOT to the Stack repo root "
                "(e.g. export STACK_REPO_ROOT=/path/to/Bio_FMs/RNA/stack)."
            ) from exc

    def load_model(self) -> None:
        """Load STACK model from checkpoint."""
        self._ensure_stack_on_path()
        from stack.model_loading import load_model_from_checkpoint

        logger.info(f"Loading STACK from {self.checkpoint_path}")
        self.model = load_model_from_checkpoint(
            self.checkpoint_path,
            strict=False
        )
        self.model.eval()
        
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        
        logger.info(f"STACK model loaded successfully on {self.device}")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using STACK.
        
        Args:
            adata: AnnData object (should have gene_symbol column from ensure_both_gene_identifiers())
            
        Returns:
            Embeddings array of shape (n_cells, 1600)
        """
        normalize_adata_var_gene_names(adata)
        # STACK can accept AnnData object directly (no need to write temp file!)
        # This is much faster than writing/reading large h5ad files
        
        # CRITICAL: STACK uses raw.var by default, but raw.var may not have gene symbols
        # Ensure raw.var has gene symbols if main var has them but raw.var doesn't
        if adata.raw is not None:
            # Check if raw.var is missing gene symbol columns that exist in main var
            raw_var = adata.raw.var
            main_var = adata.var
            
            # Check for gene_symbol (singular) or gene_symbols (plural) in main var
            gene_symbol_col = None
            if 'gene_symbol' in main_var.columns:
                gene_symbol_col = 'gene_symbol'
            elif 'gene_symbols' in main_var.columns:
                gene_symbol_col = 'gene_symbols'
            
            # If main var has gene symbols but raw.var doesn't, copy them over
            if gene_symbol_col is not None and gene_symbol_col not in raw_var.columns:
                logger.info(f"Copying '{gene_symbol_col}' from main var to raw.var for STACK compatibility")
                try:
                    # Try direct copy if indices match
                    if len(raw_var) == len(main_var) and (raw_var.index == main_var.index).all():
                        adata.raw.var[gene_symbol_col] = main_var[gene_symbol_col].values
                        logger.info(f"Successfully copied '{gene_symbol_col}' to raw.var (indices matched)")
                    else:
                        # Need to align by index - use reindex to handle mismatches
                        aligned_symbols = main_var.reindex(raw_var.index)[gene_symbol_col]
                        if aligned_symbols.notna().any():
                            adata.raw.var[gene_symbol_col] = aligned_symbols.values
                            n_matched = aligned_symbols.notna().sum()
                            logger.info(f"Successfully copied '{gene_symbol_col}' to raw.var ({n_matched}/{len(raw_var)} genes matched)")
                        else:
                            logger.warning(f"Could not align '{gene_symbol_col}' from main var to raw.var (no matching indices)")
                except Exception as e:
                    logger.warning(f"Failed to copy '{gene_symbol_col}' to raw.var: {e}. STACK will try to use main var or index.")
        
        # Use gene_symbol column if available, otherwise fall back to feature_name
        # Check both singular and plural forms, and check both main var and raw.var
        gene_name_col = self.gene_name_col
        if gene_name_col is None:
            # Priority: check raw.var first (since STACK uses raw by default), then main var
            var_to_check = adata.raw.var if adata.raw is not None else adata.var
            
            if 'gene_symbol' in var_to_check.columns:
                gene_name_col = 'gene_symbol'
                logger.info("Using 'gene_symbol' column for gene names (auto-detected)")
            elif 'gene_symbols' in var_to_check.columns:
                gene_name_col = 'gene_symbols'
                logger.info("Using 'gene_symbols' column for gene names (auto-detected)")
            elif 'gene_symbol' in adata.var.columns:
                gene_name_col = 'gene_symbol'
                logger.info("Using 'gene_symbol' column from main var for gene names (auto-detected)")
            elif 'gene_symbols' in adata.var.columns:
                gene_name_col = 'gene_symbols'
                logger.info("Using 'gene_symbols' column from main var for gene names (auto-detected)")
            elif 'feature_name' in var_to_check.columns:
                gene_name_col = 'feature_name'
                logger.info("Using 'feature_name' column for gene names (auto-detected)")
            else:
                logger.warning("No gene symbol column found, STACK will use var.index (may fail if index has Ensembl IDs)")
        
        # Pass AnnData object directly - STACK's TestSamplerDataset accepts either path or AnnData
        embeddings, _ = self.model.get_latent_representation(
            adata_path=adata,  # Can be AnnData object or file path
            genelist_path=self.genelist_path,
            batch_size=self.batch_size,
            gene_name_col=gene_name_col,
            show_progress=True,
        )
        return embeddings


def main():
    parser = create_argument_parser()
    
    # STACK-specific arguments
    # Note: Using checkpoint_path and genelist_path to match registry parameter names
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to STACK checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--genelist_path",
        type=str,
        required=True,
        help="Path to STACK gene list file (.pkl)"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = STACKExtractor(
        checkpoint_path=args.checkpoint_path,
        genelist_path=args.genelist_path,
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
