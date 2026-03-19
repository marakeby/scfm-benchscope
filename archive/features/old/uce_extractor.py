"""UCE (Universal Cell Embeddings) model extractor for single-cell RNA-seq data.

UCE is a foundation model that generates universal cell representations across
different cell types, tissues, and species. This extractor loads the pretrained
UCE model and extracts cell embeddings.
"""

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Optional imports - will fail gracefully if not available
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger

logger = get_logger()


class UCEExtractor(EmbeddingExtractor):
    """Extractor for UCE (Universal Cell Embeddings) model embeddings.
    
    UCE requires:
    - Model checkpoint file (.torch or .pt)
    - Token file with protein embeddings
    - Species-specific files (chromosome mapping, offsets, protein embeddings)
    """

    def __init__(self, params: dict) -> None:
        """Initialize UCE extractor.
        
        Args:
            params: Dictionary containing:
                - method: Method name (required by base class)
                - model_path: Path to UCE model checkpoint
                - token_file: Path to token embeddings file (all_tokens.torch)
                - spec_chrom_csv_path: Path to species chromosome CSV
                - protein_embeddings_dir: Directory with protein embedding files
                - offset_pkl_path: Path to species offsets pickle file
                - species: Species name (default: 'human')
                - batch_size: Batch size for inference (default: 25)
                - pad_length: Sequence padding length (default: 1536)
                - sample_size: Number of genes to sample (default: 1024)
                - device: Device to use ('cuda', 'cpu', or 'auto')
        """
        # Add method if not present (required by base class)
        if "method" not in params:
            params["method"] = "UCE"
        super().__init__(params)
        self.logger = get_logger()
        
        # Base class extracts params.get('params', {}), so merge with top-level params
        merged_params = {**params, **self.params}
        self.params = merged_params
        
        self.logger.info(f"UCEExtractor initialized with params: {self.params}")

        # Required parameters
        self.model_path = Path(self.params.get("model_path"))
        self.token_file = self.params.get("token_file")
        self.spec_chrom_csv_path = self.params.get("spec_chrom_csv_path")
        self.protein_embeddings_dir = self.params.get("protein_embeddings_dir")
        self.offset_pkl_path = self.params.get("offset_pkl_path")
        self.species = self.params.get("species", "human")
        self.batch_size = self.params.get("batch_size", 25)
        self.pad_length = self.params.get("pad_length", 1536)
        self.sample_size = self.params.get("sample_size", 1024)
        self.device_str = self.params.get("device", "auto")

        # Validate paths
        if not self.model_path.exists():
            raise FileNotFoundError(f"UCE model not found: {self.model_path}")

        # Set device
        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        self.model = None
        self.accelerator = None
        self.logger.info(f"Using device: {self.device}")

    def _load_model(self) -> None:
        """Load UCE model from checkpoint."""
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "accelerate package not installed. Install with: pip install accelerate"
            )
        
        try:
            import sys
            # Add UCE to path if needed
            uce_path = Path(__file__).parent.parent.parent.parent / "Bio_FMs" / "RNA" / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            
            from model import TransformerModel
        except ImportError:
            raise ImportError(
                "UCE model module not found. Please ensure UCE repository is available."
            )

        self.logger.info(f"Loading UCE model from {self.model_path}")

        # Initialize accelerator
        self.accelerator = Accelerator()

        # Load model configuration (UCE uses specific architecture)
        # Default UCE-100M parameters
        self.model = TransformerModel(
            token_dim=5120,
            d_model=5120,
            nhead=8,
            d_hid=5120,
            nlayers=4,
            output_dim=1280,
            dropout=0.05,
        )

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        self.logger.info("UCE model loaded successfully")

    def fit_transform(self, loader) -> np.ndarray:
        """Extract cell embeddings using UCE model.
        
        Args:
            loader: Data loader object with adata attribute
            
        Returns:
            Cell embeddings array of shape (n_cells, embedding_dim)
        """
        if self.model is None:
            self._load_model()

        # Get AnnData object
        adata = loader.adata
        adata_path = loader.data_path

        self.logger.info(
            f"Extracting UCE embeddings for {adata.n_obs} cells, {adata.n_vars} genes"
        )

        try:
            # Use UCE's evaluation pipeline
            import sys
            uce_path = Path(__file__).parent.parent.parent.parent / "Bio_FMs" / "RNA" / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))

            from evaluate import AnndataProcessor

            # Create processor with required arguments
            class Args:
                def __init__(self):
                    self.adata_path = str(adata_path)
                    self.dir = str(tempfile.mkdtemp())
                    self.species = self.species
                    self.filter = True
                    self.skip = False
                    self.model_loc = str(self.model_path)
                    self.batch_size = self.batch_size
                    self.pad_length = self.pad_length
                    self.pad_token_idx = 0
                    self.chrom_token_left_idx = 1
                    self.chrom_token_right_idx = 2
                    self.cls_token_idx = 3
                    self.CHROM_TOKEN_OFFSET = 143574
                    self.sample_size = self.sample_size
                    self.CXG = True
                    self.nlayers = 4
                    self.output_dim = 1280
                    self.d_hid = 5120
                    self.token_dim = 5120
                    self.multi_gpu = False
                    self.spec_chrom_csv_path = self.spec_chrom_csv_path
                    self.token_file = self.token_file
                    self.protein_embeddings_dir = self.protein_embeddings_dir
                    self.offset_pkl_path = self.offset_pkl_path

            args = Args()
            processor = AnndataProcessor(args, self.accelerator)
            processor.preprocess_anndata()
            processor.generate_idxs()
            embeddings = processor.run_evaluation()

            self.logger.info(
                f"Extracted UCE embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}"
            )

            return embeddings.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error extracting UCE embeddings: {e}")
            raise
