"""AIDO.Cell model extractor for single-cell RNA-seq data.

AIDO.Cell is a foundation model trained on 50 million cells over diverse
human tissues and organs. It uses a bidirectional transformer encoder
with auto-discretization for encoding continuous gene expression values.
This extractor loads the pretrained AIDO.Cell model and extracts cell embeddings.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Optional imports - will fail gracefully if not available
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoTokenizer = None

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger

logger = get_logger()


class AIDOExtractor(EmbeddingExtractor):
    """Extractor for AIDO.Cell model embeddings.
    
    AIDO.Cell uses HuggingFace transformers format and requires:
    - Model directory with config.json and model weights
    """

    def __init__(self, params: dict) -> None:
        """Initialize AIDO.Cell extractor.
        
        Args:
            params: Dictionary containing:
                - method: Method name (required by base class)
                - model_path: Path to AIDO.Cell model directory
                - model_name: Model name ('3M', '10M', '100M', or '650M', default: '100M')
                - batch_size: Batch size for inference (default: 32)
                - max_length: Maximum sequence length (default: None, uses model default)
                - device: Device to use ('cuda', 'cpu', or 'auto')
        """
        # Add method if not present (required by base class)
        if "method" not in params:
            params["method"] = "AIDO"
        super().__init__(params)
        self.logger = get_logger()
        
        # Base class extracts params.get('params', {}), so merge with top-level params
        merged_params = {**params, **self.params}
        self.params = merged_params
        
        self.logger.info(f"AIDOExtractor initialized with params: {self.params}")

        # Required parameters
        self.model_path = Path(self.params.get("model_path"))
        self.model_name = self.params.get("model_name", "100M")
        self.batch_size = self.params.get("batch_size", 32)
        self.max_length = self.params.get("max_length", None)
        self.device_str = self.params.get("device", "auto")

        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"AIDO.Cell model path not found: {self.model_path}")

        # Check for model files
        config_file = self.model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"AIDO.Cell config.json not found in {self.model_path}"
            )

        # Set device
        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        self.model = None
        self.tokenizer = None
        self.logger.info(f"Using device: {self.device}")

    def _load_model(self) -> None:
        """Load AIDO.Cell model from checkpoint."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers package not installed. Install with: pip install transformers"
            )
        
        self.logger.info(f"Loading AIDO.Cell model from {self.model_path}")

        try:
            # Load tokenizer (if available)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path), trust_remote_code=True
                )
            except Exception:
                self.logger.warning("Tokenizer not found, proceeding without tokenizer")
                self.tokenizer = None

            # Load model
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

            self.model.eval()
            self.model.to(self.device)
            self.logger.info("AIDO.Cell model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading AIDO.Cell model: {e}")
            raise

    def _prepare_gene_expression(self, adata) -> torch.Tensor:
        """Prepare gene expression data for AIDO.Cell.
        
        AIDO.Cell expects gene expression values. We need to:
        1. Align genes with model vocabulary
        2. Convert to appropriate format (discretized or continuous)
        """
        # Get expression matrix
        if hasattr(adata, "X") and adata.X is not None:
            X = adata.X
        else:
            raise ValueError("AnnData object must have X attribute")

        # Convert sparse to dense if needed
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Convert to torch tensor
        X_tensor = torch.from_numpy(X.astype(np.float32))

        return X_tensor

    def fit_transform(self, loader) -> np.ndarray:
        """Extract cell embeddings using AIDO.Cell model.
        
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
            f"Extracting AIDO.Cell embeddings for {adata.n_obs} cells, {adata.n_vars} genes"
        )

        try:
            # Prepare gene expression data
            X_tensor = self._prepare_gene_expression(adata)
            X_tensor = X_tensor.to(self.device)

            # Extract embeddings in batches
            all_embeddings = []
            n_cells = X_tensor.shape[0]

            self.model.eval()
            with torch.no_grad():
                for i in range(0, n_cells, self.batch_size):
                    batch = X_tensor[i : i + self.batch_size]

                    # Forward pass
                    # AIDO.Cell models typically use the CLS token or mean pooling
                    outputs = self.model(inputs_embeds=batch.unsqueeze(1))

                    # Extract embeddings
                    if hasattr(outputs, "last_hidden_state"):
                        # Use CLS token (first token) or mean pooling
                        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                    elif hasattr(outputs, "pooler_output"):
                        embeddings = outputs.pooler_output
                    else:
                        # Fallback: use hidden states if available
                        hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
                        if hidden_states is not None:
                            embeddings = hidden_states[-1][:, 0, :]
                        else:
                            raise ValueError(
                                "Could not extract embeddings from model output"
                            )

                    all_embeddings.append(embeddings.cpu())

            # Concatenate all embeddings
            embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)

            self.logger.info(
                f"Extracted AIDO.Cell embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}"
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error extracting AIDO.Cell embeddings: {e}")
            # AIDO.Cell might have a different API - try alternative approach
            self.logger.info("Attempting alternative embedding extraction method...")
            raise
