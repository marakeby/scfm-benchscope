#!/usr/bin/env python
"""STATE embedding extraction script.

This script runs in a SEPARATE environment with STATE (arc-state) dependencies.

Data requirements (STATE loader):
    STATE accepts either raw (integer) counts or log1p-transformed counts in adata.X
    and auto-detects which (max > 35 or expm1(X).sum() > 5e6 → raw, else log1p).
    This extractor only replaces NaN/Inf in X with 0 so the loader does not crash.

Setup:
    cd extractors/state
    python -m venv env
    source env/bin/activate
    pip install torch>=2.0.0
    pip install scipy>=1.15.0  # STATE requires newer scipy
    pip install scanpy anndata numpy pandas tqdm transformers peft hydra-core wandb
    pip install -e ${STATE_REPO_PATH}   # set STATE_REPO_PATH in .env to your STATE repo path

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_dir /path/to/STATE_SE
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse, csr_matrix

# Add parent to path for base module
_extractors_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_extractors_dir))
# Add repo root so we can use TEMP_PATH (avoids filling /tmp on compute nodes)
_repo_root = _extractors_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def _get_temp_dir() -> Path:
    """Use TEMP_PATH (network storage) if available, else TMPDIR/cwd to avoid filling /tmp."""
    try:
        from setup_path import TEMP_PATH
        return Path(TEMP_PATH)
    except ImportError:
        pass
    for env in ("TMPDIR", "TEMP", "TMP"):
        if os.environ.get(env):
            return Path(os.environ[env])
    return Path.cwd()

from base_extract import BaseExtractor, create_argument_parser, run_extraction, normalize_adata_var_gene_names


def _prepare_adata_for_state(adata: sc.AnnData) -> sc.AnnData:  # type: ignore[name-defined]
    """Prepare a copy of adata for STATE: replace NaN/Inf in X with 0 so the loader does not crash."""
    # Backed (on-disk) AnnData does not allow .copy() without a filename; load into memory first
    adata_work = adata.to_memory() if getattr(adata, "isbacked", False) else adata
    adata_out = adata_work.copy()
    X = adata_out.X
    if issparse(X):
        data = np.array(X.data, dtype=np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        adata_out.X = csr_matrix((data, X.indices, X.indptr), shape=X.shape)
    else:
        X_flat = np.asarray(X, dtype=np.float64)
        X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        adata_out.X = X_flat
    return adata_out


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class STATEExtractor(BaseExtractor):
    """STATE SE (State Embedding) model extractor."""
    
    def __init__(
        self,
        model_dir: str,
        checkpoint_name: Optional[str] = None,
        batch_size: int = 32,
        device: str = "auto",
        gene_name_col: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_dir=model_dir,
            checkpoint_name=checkpoint_name,
            batch_size=batch_size,
            device=device,
            gene_name_col=gene_name_col,
            **kwargs
        )
        self.model_dir = Path(model_dir)
        self.checkpoint_name = checkpoint_name
        self.batch_size = batch_size
        self.gene_name_col = gene_name_col
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Find checkpoint
        self._find_checkpoint()
    
    def _find_checkpoint(self):
        """Find the checkpoint file in the model directory."""
        ckpt_patterns = [
            "*.ckpt",
            "**/*.ckpt",
            "epoch*.ckpt",
        ]
        
        checkpoints = []
        for pattern in ckpt_patterns:
            checkpoints.extend(self.model_dir.glob(pattern))
        
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint files found in {self.model_dir}"
            )
        
        # Prefer specific checkpoint if provided
        if self.checkpoint_name:
            for ckpt in checkpoints:
                if self.checkpoint_name in ckpt.name:
                    self.checkpoint_path = ckpt
                    logger.info(f"Using checkpoint: {self.checkpoint_path}")
                    return
        
        # Otherwise prefer epoch16 or latest
        for ckpt in checkpoints:
            if "epoch16" in ckpt.name:
                self.checkpoint_path = ckpt
                logger.info(f"Using checkpoint: {self.checkpoint_path}")
                return
        
        # Use first found
        self.checkpoint_path = sorted(checkpoints)[-1]
        logger.info(f"Using checkpoint: {self.checkpoint_path}")
    
    @property
    def model_name(self) -> str:
        return "STATE"
    
    @property
    def embedding_dim(self) -> int:
        return 512  # STATE embedding dimension
    
    def load_model(self) -> None:
        """Load STATE model."""
        from state.emb.inference import Inference
        from omegaconf import OmegaConf
        
        logger.info(f"Loading STATE from {self.checkpoint_path}")
        
        # Load config from model directory if available
        config_path = self.model_dir / "config.yaml"
        cfg = None
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            logger.info(f"Loaded config from {config_path}")
        
        # Load protein embeddings from local file if available
        protein_embeds = None
        protein_embeddings_path = self.model_dir / "protein_embeddings.pt"
        if protein_embeddings_path.exists():
            logger.info(f"Loading protein embeddings from {protein_embeddings_path}")
            protein_embeds = torch.load(protein_embeddings_path, map_location="cpu", weights_only=False)
        
        # Initialize Inference (same interface as features/state_extractor.py)
        self.model = Inference(cfg=cfg, protein_embeds=protein_embeds)
        self.model.load_model(str(self.checkpoint_path))
        
        logger.info("STATE model loaded successfully")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using STATE.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array of shape (n_cells, embedding_dim)
        """
        normalize_adata_var_gene_names(adata)
        adata_to_write = _prepare_adata_for_state(adata)
        # STATE requires an h5ad file path. Always write the NaN/Inf-cleaned adata to a temp
        # file so STATE's loader never sees NaNs (avoids "cannot convert float NaN to integer").
        temp_dir = _get_temp_dir()
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".h5ad", delete=False, dir=str(temp_dir)
        )
        adata_path = temp_file.name
        adata_to_write.write_h5ad(adata_path)
        logger.info(f"Saved adata to temporary file: {adata_path}")

        try:
            # Use temporary output file on same filesystem (avoids /tmp filling on nodes)
            with tempfile.NamedTemporaryFile(
                suffix=".h5ad", delete=False, dir=str(temp_dir)
            ) as tmp_file:
                output_path = tmp_file.name
            
            # Run inference using encode_adata (same as features/state_extractor.py)
            embeddings = self.model.encode_adata(
                input_adata_path=adata_path,
                output_adata_path=output_path,
                emb_key="X_state",
                batch_size=self.batch_size,
            )
            
            # Clean up temp output file
            if Path(output_path).exists():
                Path(output_path).unlink()
            
            return embeddings
            
        finally:
            if adata_path and Path(adata_path).exists():
                Path(adata_path).unlink(missing_ok=True)


def main():
    parser = create_argument_parser()
    
    # STATE-specific arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to STATE model directory"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Specific checkpoint name to use (e.g., 'epoch16')"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = STATEExtractor(
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint_name,
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
