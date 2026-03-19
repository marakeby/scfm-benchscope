"""CellFM model extractor for single-cell RNA-seq data.

CellFM is a large-scale foundation model pre-trained on transcriptomics of
100 million human cells using a retention-based architecture (MAE Autobin).
This extractor loads the pretrained CellFM model and extracts cell embeddings.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Optional imports - will fail gracefully if not available
try:
    from perturblab.model.cellfm import CellFMModel
    PERTURBLAB_AVAILABLE = True
except ImportError:
    PERTURBLAB_AVAILABLE = False
    CellFMModel = None

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger

logger = get_logger()


class CellFMExtractor(EmbeddingExtractor):
    """Extractor for CellFM model embeddings.
    
    CellFM requires:
    - Model checkpoint directory containing config.json and model.pt
    """

    def __init__(self, params: dict) -> None:
        """Initialize CellFM extractor.
        
        Args:
            params: Dictionary containing:
                - method: Method name (required by base class)
                - model_path: Path to CellFM model directory (contains model.pt and config.json)
                - model_name: Model name ('80m' or '800m', default: '80m')
                - batch_size: Batch size for inference (default: 32)
                - return_cls_token: Whether to return CLS token embeddings (default: True)
                - device: Device to use ('cuda', 'cpu', or 'auto')
        """
        # Add method if not present (required by base class)
        if "method" not in params:
            params["method"] = "CellFM"
        super().__init__(params)
        self.logger = get_logger()
        
        # Base class extracts params.get('params', {}), so merge with top-level params
        merged_params = {**params, **self.params}
        self.params = merged_params
        
        self.logger.info(f"CellFMExtractor initialized with params: {self.params}")

        # Required parameters
        self.model_path = Path(self.params.get("model_path"))
        self.model_name = self.params.get("model_name", "80m")
        self.batch_size = self.params.get("batch_size", 32)
        self.return_cls_token = self.params.get("return_cls_token", True)
        self.device_str = self.params.get("device", "auto")

        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"CellFM model path not found: {self.model_path}")

        # Check for model files
        model_file = self.model_path / "model.pt"
        config_file = self.model_path / "config.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"CellFM model.pt not found in {self.model_path}")
        if not config_file.exists():
            self.logger.warning(f"CellFM config.json not found in {self.model_path}")

        # Set device
        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        self.model = None
        self.logger.info(f"Using device: {self.device}")

    def _load_model(self) -> None:
        """Load CellFM model from checkpoint."""
        if not PERTURBLAB_AVAILABLE:
            raise ImportError(
                "CellFM package not installed. Install with: pip install perturblab"
            )

        self.logger.info(f"Loading CellFM model from {self.model_path}")

        # Try to load from local path first
        try:
            self.model = CellFMModel.from_pretrained(str(self.model_path))
        except Exception:
            # Fallback: try loading model name
            try:
                model_name = f"cellfm-{self.model_name}"
                self.model = CellFMModel.from_pretrained(model_name)
            except Exception as e:
                # Last resort: manual loading
                self.logger.warning(
                    f"Could not load via from_pretrained, attempting manual load: {e}"
                )
                # This would require knowing the exact model architecture
                raise ImportError(
                    f"Could not load CellFM model. Please ensure perturblab is properly installed."
                )

        self.model.eval()
        self.model.to(self.device)
        self.logger.info("CellFM model loaded successfully")

    def fit_transform(self, loader) -> np.ndarray:
        """Extract cell embeddings using CellFM model.
        
        Args:
            loader: Data loader object with adata attribute
            
        Returns:
            Cell embeddings array of shape (n_cells, embedding_dim)
        """
        if self.model is None:
            self._load_model()

        # Get AnnData object
        adata = loader.adata

        self.logger.info(
            f"Extracting CellFM embeddings for {adata.n_obs} cells, {adata.n_vars} genes"
        )

        try:
            # Prepare data using CellFM's preprocessing
            adata_processed = self.model.prepare_data(adata.copy())

            # Get embeddings
            embeddings_dict = self.model.predict_embeddings(
                adata_processed,
                batch_size=self.batch_size,
                return_cls_token=self.return_cls_token,
            )

            # Extract cell embeddings
            if "cell_embeddings" in embeddings_dict:
                embeddings = embeddings_dict["cell_embeddings"]
            elif "embeddings" in embeddings_dict:
                embeddings = embeddings_dict["embeddings"]
            else:
                # Take first value if structure is different
                embeddings = list(embeddings_dict.values())[0]

            # Ensure numpy array
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()

            embeddings = embeddings.astype(np.float32)

            self.logger.info(
                f"Extracted CellFM embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}"
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error extracting CellFM embeddings: {e}")
            raise
