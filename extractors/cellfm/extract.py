#!/usr/bin/env python
"""CellFM embedding extraction script.

CellFM is a large-scale foundation model pre-trained on transcriptomics of
100 million human cells using a retention-based architecture.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_path /path/to/cellfm --model_name 80m
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

import logging
logger = logging.getLogger(__name__)

# Optional imports
try:
    from perturblab.model.cellfm import CellFMModel
    PERTURBLAB_AVAILABLE = True
except ImportError:
    PERTURBLAB_AVAILABLE = False
    CellFMModel = None


class CellFMExtractor(BaseExtractor):
    """CellFM embedding extractor.
    
    CellFM requires:
    - Model checkpoint directory containing config.json and model.pt
    """

    def __init__(
        self,
        model_path: str = None,
        model_name: str = "80m",
        batch_size: int = 32,
        return_cls_token: bool = True,
        device: str = "auto",
        params: dict = None,
        **kwargs
    ):
        # Handle both YAML config style and CLI style
        if params is not None:
            super().__init__(params=params)
            model_path = self.params.get('model_path', model_path)
            model_name = self.params.get('model_name', model_name)
            batch_size = self.params.get('batch_size', batch_size)
            return_cls_token = self.params.get('return_cls_token', return_cls_token)
            device = self.params.get('device', device)
        else:
            super().__init__(
                model_path=model_path,
                model_name=model_name,
                batch_size=batch_size,
                return_cls_token=return_cls_token,
                device=device,
                **kwargs
            )
        
        # Required parameters
        self.model_path = Path(model_path) if model_path else None
        self.model_name_str = model_name
        self.batch_size = batch_size
        self.return_cls_token = return_cls_token
        
        # Validate model path
        if self.model_path and not self.model_path.exists():
            raise FileNotFoundError(f"CellFM model path not found: {self.model_path}")

        # Check for model files
        if self.model_path:
            model_file = self.model_path / "model.pt"
            config_file = self.model_path / "config.json"
            
            if not model_file.exists():
                raise FileNotFoundError(f"CellFM model.pt not found in {self.model_path}")
            if not config_file.exists():
                logger.warning(f"CellFM config.json not found in {self.model_path}")

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        logger.info(f"Using device: {self.device}")

    @property
    def model_name(self) -> str:
        return "CellFM"
    
    @property
    def embedding_dim(self) -> int:
        return -1  # Unknown until model is loaded
    
    def load_model(self) -> None:
        """Load CellFM model from checkpoint."""
        if self._model_loaded:
            return
            
        if not PERTURBLAB_AVAILABLE:
            raise ImportError(
                "CellFM package not installed. Install with: pip install perturblab"
            )

        logger.info(f"Loading CellFM model from {self.model_path}")

        # Try to load from local path first
        try:
            self.model = CellFMModel.from_pretrained(str(self.model_path))
        except Exception:
            # Fallback: try loading model name
            try:
                model_name = f"cellfm-{self.model_name_str}"
                self.model = CellFMModel.from_pretrained(model_name)
            except Exception as e:
                raise ImportError(
                    f"Could not load CellFM model. Please ensure perturblab is properly installed: {e}"
                )

        self.model.eval()
        self.model.to(self.device)
        self._model_loaded = True
        logger.info("CellFM model loaded successfully")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using CellFM.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array
        """
        if not self._model_loaded:
            self.load_model()

        logger.info(
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

            logger.info(
                f"Extracted CellFM embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}"
            )

            return embeddings

        except Exception as e:
            logger.error(f"Error extracting CellFM embeddings: {e}")
            raise


def main():
    parser = create_argument_parser()
    
    # CellFM-specific arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to CellFM model directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="80m",
        choices=["80m", "800m"],
        help="Model name (default: 80m)"
    )
    parser.add_argument(
        "--return_cls_token",
        action="store_true",
        default=True,
        help="Return CLS token embeddings"
    )
    
    args = parser.parse_args()
    
    extractor = CellFMExtractor(
        model_path=args.model_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        return_cls_token=args.return_cls_token,
        device=args.device,
    )
    
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
