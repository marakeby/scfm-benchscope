"""LaplacianAE embedding extractor for scFM_eval framework.

Extracts embeddings using pre-trained Laplacian-regularized autoencoder models.
"""
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch
from scimilarity.utils import align_dataset
from utils.logs_ import get_logger
from features.extractor import EmbeddingExtractor

logger = get_logger()

# Add SCFM_meta to path to import model classes
SCFM_META_DIR = Path("/lotterlab/users/riccardo/ML_BIO/SCFM_meta")
if str(SCFM_META_DIR) not in sys.path:
    sys.path.insert(0, str(SCFM_META_DIR))

try:
    from src.lightning_models import LaplacianAutoEncoder
    from src.models import Encoder
    LAPLACIAN_AE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LaplacianAE not available: {e}")
    LAPLACIAN_AE_AVAILABLE = False


class LaplacianAEExtractor(EmbeddingExtractor):
    """
    Extractor for LaplacianAE embeddings.
    
    Extracts cell embeddings using pre-trained Laplacian-regularized autoencoder models.
    The model expects:
    - Full expression data aligned to gene_order.tsv
    - Data passed through a fully connected network (encoder)
    
    Stores embeddings in adata.obsm['X_laplacian_ae'].
    """
    
    def __init__(self, params):
        """
        Initialize LaplacianAE extractor.
        
        Args:
            params: Dictionary containing:
                - method: 'laplacian_ae'
                - params: Dictionary with:
                    - checkpoint_path: Path to model checkpoint (.ckpt file)
                    - gene_order_file: Path to gene order file (TSV, one gene per line)
                    - use_gpu: Whether to use GPU (default: True)
                    - batch_size: Batch size for embedding extraction (default: 1000)
        """
        super().__init__(params)
        logger.info(f'LaplacianAEExtractor ({self.params})')
        
        if not LAPLACIAN_AE_AVAILABLE:
            raise ImportError(
                "LaplacianAE is not available. Please ensure SCFM_meta is in your path."
            )
        
        # Extract parameters with defaults
        self.checkpoint_path = self.params.get('checkpoint_path')
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required in params")
        
        self.gene_order_file = self.params.get(
            'gene_order_file',
            '/lotterlab/datasets/VCC/MODEL_CHECKPOINTS/SC_FM_repo_checkpoints/SCSimilarity/model_v1.1/gene_order.tsv'
        )
        self.use_gpu = self.params.get('use_gpu', True)
        self.batch_size = self.params.get('batch_size', 1000)
        
        # Load gene order
        logger.info(f"Loading gene order from: {self.gene_order_file}")
        with open(self.gene_order_file, 'r') as f:
            self.gene_order = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(self.gene_order)} genes from gene order file")
        
        # Initialize model (will be loaded in fit_transform)
        self.model = None
        self.encoder = None
        self._model_loaded = False
        
        logger.info(
            f"LaplacianAE extractor initialized: "
            f"checkpoint_path={self.checkpoint_path}, "
            f"gene_order_file={self.gene_order_file}, "
            f"use_gpu={self.use_gpu}, batch_size={self.batch_size}"
        )
    
    def _load_model(self):
        """
        Load LaplacianAE model from checkpoint.
        """
        if self._model_loaded and self.encoder is not None:
            return
        
        logger.info(f"Loading LaplacianAE model from checkpoint: {self.checkpoint_path}")
        
        # Check if checkpoint file exists
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract hyperparameters from checkpoint
        # PyTorch Lightning stores hyperparameters in different places
        hparams = None
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
        elif 'hparams' in checkpoint:
            hparams = checkpoint['hparams']
        elif hasattr(checkpoint, 'hyper_parameters'):
            hparams = checkpoint.hyper_parameters
        
        if hparams:
            input_dim = hparams.get('input_dim')
            embed_dim = hparams.get('embed_dim', 64)
            hidden_dims = hparams.get('hidden_dims', [1024, 256])
            dropout = hparams.get('dropout', 0.0)
        else:
            # Fallback: try to infer from model state dict
            logger.warning("No hyperparameters found in checkpoint, inferring from state dict")
            # Try to infer input_dim from first layer weight shape
            state_dict = checkpoint.get('state_dict', checkpoint)
            # Look for encoder.net.0.weight (first linear layer)
            encoder_key = None
            for key in state_dict.keys():
                if 'encoder.net.0.weight' in key or 'model.encoder.net.0.weight' in key:
                    encoder_key = key
                    break
            
            if encoder_key:
                input_dim = state_dict[encoder_key].shape[1]
                embed_dim = 64  # default
                hidden_dims = [1024, 256]  # default
                dropout = 0.0
                logger.info(f"Inferred input_dim={input_dim} from state dict")
            else:
                raise ValueError(
                    "Could not infer model architecture from checkpoint. "
                    "Please ensure checkpoint contains hyperparameters or state dict with encoder layers."
                )
        
        logger.info(
            f"Model architecture: input_dim={input_dim}, embed_dim={embed_dim}, "
            f"hidden_dims={hidden_dims}, dropout={dropout}"
        )
        
        # Create model instance
        self.model = LaplacianAutoEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        # Load state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (PyTorch Lightning adds this)
            new_state_dict = {}
            for key, value in state_dict.items():
                # Handle different key formats: 'model.encoder...' or 'encoder...'
                new_key = key
                if key.startswith('model.model.'):
                    new_key = key.replace('model.model.', 'model.')
                elif key.startswith('model.'):
                    new_key = key.replace('model.', '')
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            # Assume checkpoint is just the state dict
            self.model.load_state_dict(checkpoint, strict=False)
        
        # Extract encoder for inference
        self.encoder = self.model.model.encoder
        
        # Set to eval mode
        self.encoder.eval()
        
        # Move to GPU if requested
        if self.use_gpu and torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            logger.info("Model moved to GPU")
        else:
            if self.use_gpu:
                logger.warning("GPU requested but not available, using CPU")
            logger.info("Model on CPU")
        
        self._model_loaded = True
        logger.info("LaplacianAE model loaded successfully")
    
    def preprocess_h5ad(self, h5ad_path_data, reorder: bool = True):
        """
        Loads and preprocesses AnnData according to LaplacianAE requirements:
          - Aligns gene ordering to gene_order.tsv
          - Assumes data is already normalized (TP10K + log1p)
        
        Args:
            h5ad_path_data: Path to h5ad file or AnnData object
            reorder: Whether to reorder genes (default: True)
        
        Returns:
            AnnData object with processed .X
        """
        if isinstance(h5ad_path_data, str):
            adata = sc.read_h5ad(h5ad_path_data)
        else:
            adata = h5ad_path_data
        
        # Always do gene alignment (LaplacianAE requirement)
        if reorder:
            logger.info(f"Aligning genes to target order ({len(self.gene_order)} genes)...")
            adata = align_dataset(adata, self.gene_order)
            logger.info(f"Gene alignment complete. Final shape: {adata.shape}")
        
        return adata
    
    def get_embeddings(self, adata, num_cells: int = -1):
        """
        Computes embeddings using the LaplacianAE encoder.
        
        Args:
            adata: AnnData object with aligned gene expression
            num_cells: Number of cells to process (-1 for all)
        
        Returns:
            numpy.ndarray of shape [num_cells, embed_dim]
        """
        if not self._model_loaded:
            self._load_model()
        
        # Extract expression matrix
        X = adata.X
        
        # Convert to dense if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Convert to numpy if not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Limit number of cells if specified
        if num_cells > 0 and num_cells < X.shape[0]:
            X = X[:num_cells]
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Move to GPU if available
        if self.use_gpu and torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
        
        # Extract embeddings in batches
        embeddings_list = []
        n_cells = X_tensor.shape[0]
        
        logger.info(f"Extracting embeddings for {n_cells} cells in batches of {self.batch_size}...")
        
        with torch.no_grad():
            for i in range(0, n_cells, self.batch_size):
                batch_end = min(i + self.batch_size, n_cells)
                batch_X = X_tensor[i:batch_end]
                
                # Forward pass through encoder
                batch_embeddings = self.encoder(batch_X)
                
                # Move to CPU and convert to numpy
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings_list.append(batch_embeddings)
                
                if (i // self.batch_size + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_end}/{n_cells} cells...")
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings_list)
        
        logger.info(f"Extracted embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    def fit_transform(self, data_loader):
        """
        Extract embeddings from AnnData using LaplacianAE.
        
        Args:
            data_loader: Data loader object with adata attribute
            
        Returns:
            numpy.ndarray: Cell embeddings of shape (n_cells, embedding_dim)
        """
        if not LAPLACIAN_AE_AVAILABLE:
            raise ImportError("LaplacianAE is not available")
        
        data = data_loader.adata
        logger.info(f"Extracting LaplacianAE embeddings from AnnData with shape {data.shape}")
        
        # Preprocess (align genes)
        logger.info('Applying LaplacianAE preprocessing (gene alignment)...')
        data = self.preprocess_h5ad(data, reorder=True)
        
        # Extract embeddings
        embeddings = self.get_embeddings(data, num_cells=-1)
        
        # Store in adata.obsm
        data.obsm['X_laplacian_ae'] = embeddings
        
        logger.info(
            f"Extracted embeddings with shape {embeddings.shape} "
            f"and stored in adata.obsm['X_laplacian_ae']"
        )
        
        return embeddings.copy()
