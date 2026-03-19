#!/usr/bin/env python
"""AIDO.Cell embedding extraction script.

This script requires a separate environment due to dependency conflicts.

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --model_dir /path/to/aido/model
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

import logging
logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("transformers package not found. Please install it.")


class AIDOExtractor(BaseExtractor):
    """AIDO.Cell embedding extractor."""
    
    def __init__(
        self,
        model_dir: str,
        model_name: str = "100M",
        batch_size: int = 32,
        max_length: int | None = None,
        device: str = "auto",
        **kwargs
    ):
        super().__init__(
            model_dir=model_dir,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            **kwargs
        )
        self.model_dir = Path(model_dir)
        self.model_name_str = model_name  # Store as model_name_str to avoid conflict with property
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Determine device
        import torch
        if device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
        self.device = torch.device(self.device_str)
        
        self.model = None
        self.tokenizer = None
    
    @property
    def model_name(self) -> str:
        return f"AIDO.Cell-{self.model_name_str}"
    
    @property
    def embedding_dim(self) -> int:
        return -1  # Unknown until model is loaded
    
    def load_model(self) -> None:
        """Load AIDO.Cell model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package not found.")
        
        logger.info(f"Loading AIDO.Cell model from {self.model_dir}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModel.from_pretrained(str(self.model_dir))
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"AIDO.Cell model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load AIDO.Cell model: {e}")
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Extract embeddings using AIDO.Cell.
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        
        logger.info("Extracting AIDO.Cell embeddings...")
        
        # AIDO.Cell expects specific preprocessing
        # This is a simplified version - may need adjustment based on actual AIDO requirements
        X = adata.X
        
        # Convert to dense if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Convert to numpy if not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Process in batches
        embeddings_list = []
        n_cells = X.shape[0]
        
        with torch.no_grad():
            for i in range(0, n_cells, self.batch_size):
                batch_end = min(i + self.batch_size, n_cells)
                batch_X = X[i:batch_end]
                
                # Convert to tensor
                batch_tensor = torch.FloatTensor(batch_X).to(self.device)
                
                # Get embeddings (simplified - actual implementation may differ)
                outputs = self.model(batch_tensor)
                
                # Extract CLS token or mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output.cpu().numpy()
                else:
                    # Fallback: use first output
                    embeddings = outputs[0].mean(dim=1).cpu().numpy()
                
                embeddings_list.append(embeddings)
        
        embeddings = np.vstack(embeddings_list)
        
        logger.info(f"Extracted AIDO.Cell embeddings. Shape: {embeddings.shape}")
        
        return embeddings


def main():
    parser = create_argument_parser()
    
    # AIDO-specific arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to AIDO.Cell model directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="100M",
        choices=["3M", "10M", "100M", "650M"],
        help="Model name (default: 100M)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (default: None, uses model default)"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = AIDOExtractor(
        model_dir=args.model_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    
    # Run extraction
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
