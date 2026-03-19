"""STACK (StateICL) model extractor for single-cell RNA-seq data.

STACK is a large-scale encoder-decoder foundation model for single-cell biology
that uses tabular attention architecture. This extractor loads the pretrained
STACK model and extracts cell embeddings.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger
from utils.data_state import DataState, get_data_state

logger = get_logger()


class STACKExtractor(EmbeddingExtractor):
    """Extractor for STACK (StateICL) model embeddings.
    
    STACK uses a tabular attention architecture and requires:
    - Checkpoint file (.ckpt)
    - Gene list file (.pkl) used during training
    """

    def __init__(self, params: dict) -> None:
        """Initialize STACK extractor.
        
        Args:
            params: Dictionary containing:
                - method: Method name (required by base class)
                - checkpoint_path: Path to STACK checkpoint (.ckpt file)
                - genelist_path: Path to gene list pickle file (.pkl)
                - batch_size: Batch size for inference (default: 32)
                - gene_name_col: Optional column name for gene symbols in adata.var
                - device: Device to use ('cuda', 'cpu', or 'auto')
        """
        # Add method if not present (required by base class)
        if "method" not in params:
            params["method"] = "STACK"
        super().__init__(params)
        self.logger = get_logger()
        
        # Base class extracts params.get('params', {}), so merge with top-level params
        # This handles both YAML format (nested) and direct format (flat)
        merged_params = {**params, **self.params}
        self.params = merged_params
        
        self.logger.info(f"STACKExtractor initialized with params: {self.params}")

        # Required parameters
        self.checkpoint_path = Path(self.params.get("checkpoint_path"))
        self.genelist_path = Path(self.params.get("genelist_path"))
        self.batch_size = self.params.get("batch_size", 32)
        self.gene_name_col = self.params.get("gene_name_col", None)
        self.device_str = self.params.get("device", "auto")

        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"STACK checkpoint not found: {self.checkpoint_path}"
            )
        if not self.genelist_path.exists():
            raise FileNotFoundError(
                f"STACK gene list not found: {self.genelist_path}"
            )

        # Set device
        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        self.model = None
        self.logger.info(f"Using device: {self.device}")

    def _load_model(self) -> None:
        """Load STACK model from checkpoint."""
        try:
            from stack.model_loading import load_model_from_checkpoint
        except ImportError as e:
            raise ImportError(
                "STACK package not installed. Install with: pip install arc-stack"
            ) from e

        self.logger.info(f"Loading STACK model from {self.checkpoint_path}")
        try:
            self.model = load_model_from_checkpoint(
                str(self.checkpoint_path),
                device=self.device,
                strict=False,  # Allow missing/extra keys
            )
            self.model.eval()
            self.logger.info("STACK model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading STACK model: {e}")
            # Try with strict=False explicitly
            try:
                import torch
                from stack.models.core import StateICLModel
                
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                if "hyper_parameters" in checkpoint:
                    model_config = checkpoint["hyper_parameters"].get("model_config", {})
                    self.model = StateICLModel(**model_config)
                    state_dict = checkpoint.get("state_dict", {})
                    # Remove 'model.' prefix if present
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith("model."):
                            new_state_dict[key[6:]] = value
                        else:
                            new_state_dict[key] = value
                    self.model.load_state_dict(new_state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    self.logger.info("STACK model loaded successfully (manual loading)")
                else:
                    raise
            except Exception as e2:
                self.logger.error(f"Manual loading also failed: {e2}")
                raise

    def fit_transform(self, loader) -> np.ndarray:
        """Extract cell embeddings using STACK model.
        
        Args:
            loader: Data loader object with adata attribute
            
        Returns:
            Cell embeddings array of shape (n_cells, embedding_dim)
        """
        if self.model is None:
            self._load_model()

        # Get AnnData object
        adata = loader.adata
        # Use loader.path (resolved path) or save adata temporarily
        adata_path = getattr(loader, 'path', None)
        
        # If path is not available, save adata temporarily
        if adata_path is None or not Path(adata_path).exists():
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            adata.write(temp_path)
            adata_path = temp_path
            cleanup_temp = True
        else:
            cleanup_temp = False

        # Log data state - STACK applies log1p internally if needed
        data_state = get_data_state(adata)
        self.logger.info(
            f"Extracting STACK embeddings for {adata.n_obs} cells, {adata.n_vars} genes. "
            f"Data state: {data_state.value}"
        )

        # Use STACK's get_latent_representation method (auto-detects log1p state)
        try:
            embeddings, dataset_embeddings = self.model.get_latent_representation(
                adata_path=str(adata_path),
                genelist_path=str(self.genelist_path),
                gene_name_col=self.gene_name_col,
                batch_size=self.batch_size,
                show_progress=True,
                num_workers=4,
            )
            
            if cleanup_temp:
                Path(temp_path).unlink()
        except Exception as e:
            self.logger.error(f"Error extracting STACK embeddings: {e}")
            if cleanup_temp and Path(temp_path).exists():
                Path(temp_path).unlink()
            raise

        self.logger.info(
            f"Extracted STACK embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}"
        )

        return embeddings.astype(np.float32)
