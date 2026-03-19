"""STATE (State Embedding) model extractor for single-cell RNA-seq data.

STATE SE (State Embedding) is a foundation model for single-cell data that
generates universal cell embeddings. This extractor loads the pretrained STATE
model and extracts cell embeddings.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger

logger = get_logger()


class STATEExtractor(EmbeddingExtractor):
    """Extractor for STATE SE (State Embedding) model embeddings.
    
    STATE SE requires:
    - Model folder containing checkpoint files
    - Optional config file (can be extracted from checkpoint)
    """

    def __init__(self, params: dict) -> None:
        """Initialize STATE extractor.
        
        Args:
            params: Dictionary containing:
                - method: Method name (required by base class)
                - model_folder: Path to STATE model folder (contains checkpoint)
                - batch_size: Batch size for inference (default: 32)
                - config_path: Optional path to config YAML file
                - device: Device to use ('cuda', 'cpu', or 'auto')
        """
        # Add method if not present (required by base class)
        if "method" not in params:
            params["method"] = "STATE"
        super().__init__(params)
        self.logger = get_logger()
        
        # Base class extracts params.get('params', {}), so merge with top-level params
        merged_params = {**params, **self.params}
        self.params = merged_params
        
        self.logger.info(f"STATEExtractor initialized with params: {self.params}")

        # Required parameters
        self.model_folder = Path(self.params.get("model_folder"))
        self.batch_size = self.params.get("batch_size", 32)
        self.config_path = self.params.get("config_path", None)
        self.device_str = self.params.get("device", "auto")

        # Validate model folder
        if not self.model_folder.exists():
            raise FileNotFoundError(
                f"STATE model folder not found: {self.model_folder}"
            )

        # Find checkpoint file
        # Allow explicit checkpoint path override
        if "checkpoint_path" in self.params:
            self.checkpoint_path = Path(self.params["checkpoint_path"])
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Specified checkpoint not found: {self.checkpoint_path}"
                )
        else:
            checkpoint_files = list(self.model_folder.glob("*.ckpt"))
            if not checkpoint_files:
                raise FileNotFoundError(
                    f"No checkpoint file found in {self.model_folder}"
                )
            # Prefer epoch16 over epoch4 if available
            epoch16_files = [f for f in checkpoint_files if "epoch16" in f.name]
            if epoch16_files:
                self.checkpoint_path = epoch16_files[0]
            else:
                self.checkpoint_path = checkpoint_files[0]
        
        self.logger.info(f"Using checkpoint: {self.checkpoint_path}")

        # Set device
        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        self.inference = None
        self.logger.info(f"Using device: {self.device}")

    def _load_model(self) -> None:
        """Load STATE model from checkpoint."""
        try:
            from state.emb.inference import Inference
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError(
                "STATE package not installed. Install with: pip install arc-state"
            ) from e

        self.logger.info(f"Loading STATE model from {self.checkpoint_path}")

        # Load protein embeddings from local file if available
        protein_embeds = None
        protein_embeddings_path = self.model_folder / "protein_embeddings.pt"
        if protein_embeddings_path.exists():
            self.logger.info(f"Loading protein embeddings from {protein_embeddings_path}")
            protein_embeds = torch.load(protein_embeddings_path, map_location="cpu", weights_only=False)
        else:
            self.logger.warning(
                f"Protein embeddings file not found at {protein_embeddings_path}. "
                "Will attempt to load from config or checkpoint."
            )

        # Initialize inference object with config
        cfg = None
        if self.config_path:
            cfg = OmegaConf.load(self.config_path)
        else:
            # Try to load config from model folder
            config_file = self.model_folder / "config.yaml"
            if config_file.exists():
                self.logger.info(f"Loading config from {config_file}")
                cfg = OmegaConf.load(config_file)
                # Update embedding paths in config to point to local files if they exist
                if cfg and "embeddings" in cfg:
                    for emb_key in cfg["embeddings"]:
                        if emb_key != "current" and isinstance(cfg["embeddings"][emb_key], dict):
                            if protein_embeddings_path.exists():
                                cfg["embeddings"][emb_key]["all_embeddings"] = str(protein_embeddings_path)
                                self.logger.info(
                                    f"Updated config embedding path to {protein_embeddings_path}"
                                )

        # Initialize inference with protein embeddings (will be used if provided)
        self.inference = Inference(cfg=cfg, protein_embeds=protein_embeds)
        self.inference.load_model(str(self.checkpoint_path))
        self.logger.info("STATE model loaded successfully")

    def fit_transform(self, loader) -> np.ndarray:
        """Extract cell embeddings using STATE model.
        
        Args:
            loader: Data loader object with adata attribute
            
        Returns:
            Cell embeddings array of shape (n_cells, embedding_dim)
        """
        if self.inference is None:
            self._load_model()

        # Get AnnData object
        adata = loader.adata
        # Use loader.path (resolved path) or save adata temporarily
        adata_path = getattr(loader, 'path', None)
        
        # If path is not available, save adata temporarily
        if adata_path is None or not Path(adata_path).exists():
            temp_file = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            adata.write(temp_path)
            adata_path = temp_path
            cleanup_input = True
        else:
            cleanup_input = False

        self.logger.info(
            f"Extracting STATE embeddings for {adata.n_obs} cells, {adata.n_vars} genes"
        )

        # Use temporary output file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            # Use STATE's encode_adata method
            self.inference.encode_adata(
                input_adata_path=str(adata_path),
                output_adata_path=output_path,
                emb_key="X_state",
                batch_size=self.batch_size,
            )

            # Read embeddings from output file
            import anndata as ad
            output_adata = ad.read_h5ad(output_path)
            embeddings = output_adata.obsm["X_state"].astype(np.float32)

            self.logger.info(
                f"Extracted STATE embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}"
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error extracting STATE embeddings: {e}")
            raise
        finally:
            # Clean up temporary files
            if Path(output_path).exists():
                Path(output_path).unlink()
            if cleanup_input and Path(adata_path).exists():
                Path(adata_path).unlink()
