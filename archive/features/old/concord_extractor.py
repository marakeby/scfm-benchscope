"""
CONCORD embedding extractor.

Trains CONCORD on the dataset and extracts embeddings.
Similar to scVI, CONCORD trains on the data rather than using pre-trained checkpoints.
"""
import scanpy as sc
import logging
from pathlib import Path
from typing import Optional

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger

logger = get_logger()

try:
    from concord import Concord
except ImportError:
    logger.error(
        "CONCORD package not found. Please install it or add it to your path. "
        "If using editable install, ensure it's in your environment."
    )
    raise


class ConcordExtractor(EmbeddingExtractor):
    """
    Extract embeddings using CONCORD (Contrastive learning for single-cell data).
    
    CONCORD trains a contrastive learning model on the dataset to learn embeddings
    that preserve biological signal while removing batch effects.
    
    Unlike pre-trained models (scGPT, Geneformer), CONCORD trains from scratch
    on each dataset, similar to scVI.
    """
    
    def __init__(self, params: dict):
        """
        Initialize CONCORD extractor.
        
        Args:
            params: Configuration parameters including:
                - batch_key: Column name for batch/domain information (required)
                - save_dir: Directory to save model checkpoints (optional)
                - train_frac: Fraction of data to use for training (default: 1.0)
                - n_epochs: Number of training epochs (default: 15)
                - latent_dim: Latent dimension for embeddings (default: 100)
                - batch_size: Training batch size (default: 256)
                - lr: Learning rate (default: 1e-2)
                - domain_key: Alias for batch_key (optional)
                - class_key: Column name for class labels (optional, for classifier)
                - encoder_dims: Encoder architecture (default: [1000])
                - decoder_dims: Decoder architecture (default: [1000])
                - use_decoder: Whether to use decoder (default: False)
                - clr_temperature: Contrastive loss temperature (default: 0.4)
                - clr_beta: NT-Xent loss beta parameter (default: 1.0)
                - p_intra_knn: Probability of sampling from k-NN (default: 0.0)
                - p_intra_domain: Probability of sampling within domain (default: 1.0)
                - normalize_total: Whether to normalize (default: False, assumes pre-normalized)
                - log1p: Whether to apply log1p (default: False, assumes pre-normalized)
                - And other CONCORD parameters (see Concord class for full list)
        """
        super().__init__(params)
        logger.info(f'ConcordExtractor initialized with params: {self.params}')
        
        # Required parameters
        self.batch_key = self.params.get('batch_key') or self.params.get('domain_key')
        if not self.batch_key:
            raise ValueError(
                "Missing required parameter 'batch_key' or 'domain_key'. "
                "CONCORD requires batch/domain information for training."
            )
        
        # Training parameters
        self.train_frac = self.params.get('train_frac', 1.0)
        self.n_epochs = self.params.get('n_epochs', 15)
        self.latent_dim = self.params.get('latent_dim', 100)
        self.batch_size = self.params.get('batch_size', 256)
        self.lr = self.params.get('lr', 1e-2)
        
        # Model architecture
        self.encoder_dims = self.params.get('encoder_dims', [1000])
        self.decoder_dims = self.params.get('decoder_dims', [1000])
        self.use_decoder = self.params.get('use_decoder', False)
        
        # Contrastive learning parameters
        self.clr_temperature = self.params.get('clr_temperature', 0.4)
        self.clr_beta = self.params.get('clr_beta', 1.0)
        self.p_intra_knn = self.params.get('p_intra_knn', 0.0)
        self.p_intra_domain = self.params.get('p_intra_domain', 1.0)
        
        # Preprocessing flags (typically data is pre-normalized by the pipeline)
        self.normalize_total = self.params.get('normalize_total', False)
        self.log1p = self.params.get('log1p', False)
        
        # Save directory for model checkpoints
        self.save_dir = self.params.get('save_dir', None)
        if self.save_dir:
            self.save_dir = Path(self.save_dir) / 'concord_model'
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"CONCORD model will be saved to: {self.save_dir}")
        else:
            logger.warning("save_dir not provided. Model will not be saved.")
        
        # Other optional parameters
        self.class_key = self.params.get('class_key', None)
        self.seed = self.params.get('seed', 0)
        self.device = self.params.get('device', None)  # Will use CUDA if available
        
        # Output key for embeddings
        self.output_key = self.params.get('output_key', 'X_concord')
    
    @staticmethod
    def validate_config(params: dict) -> None:
        """
        Validate configuration parameters.
        
        Args:
            params: Configuration parameters
            
        Raises:
            AssertionError: If required parameters are missing
        """
        assert 'params' in params and params is not None, "Missing 'params' in parameters"
        params_dict = params.get('params', {})
        assert 'batch_key' in params_dict or 'domain_key' in params_dict, \
            "Missing required parameter 'batch_key' or 'domain_key'"
    
    def fit_transform(self, data_loader) -> 'np.ndarray':
        """
        Train CONCORD model and extract embeddings.
        
        Args:
            data_loader: DataLoader object containing the AnnData object
            
        Returns:
            numpy.ndarray: Cell embeddings of shape (n_cells, latent_dim)
        """
        adata = data_loader.adata
        
        logger.info(f"Training CONCORD on dataset with shape: {adata.shape}")
        logger.info(f"Batch key: {self.batch_key}, Train fraction: {self.train_frac}")
        
        # Check if data is already preprocessed
        # CONCORD expects normalized data, but can handle normalization itself
        if hasattr(adata, 'uns') and adata.uns.get('preprocessed', False):
            logger.info("Data is already preprocessed. CONCORD will use it as-is.")
            # Set CONCORD to skip normalization
            normalize_total = False
            log1p = False
        else:
            logger.info("Data appears not preprocessed. CONCORD will normalize if needed.")
            normalize_total = self.normalize_total
            log1p = self.log1p
        
        # Prepare CONCORD parameters
        concord_params = {
            'domain_key': self.batch_key,  # CONCORD uses 'domain_key'
            'train_frac': self.train_frac,
            'n_epochs': self.n_epochs,
            'latent_dim': self.latent_dim,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'encoder_dims': self.encoder_dims,
            'decoder_dims': self.decoder_dims,
            'use_decoder': self.use_decoder,
            'clr_temperature': self.clr_temperature,
            'clr_beta': self.clr_beta,
            'p_intra_knn': self.p_intra_knn,
            'p_intra_domain': self.p_intra_domain,
            'normalize_total': normalize_total,
            'log1p': log1p,
            'seed': self.seed,
        }
        
        # Add optional parameters if provided
        if self.class_key:
            concord_params['class_key'] = self.class_key
            concord_params['use_classifier'] = True
        
        if self.device:
            concord_params['device'] = self.device
        
        # Add any other CONCORD parameters that were provided
        concord_param_names = {
            'schedule_ratio', 'element_mask_prob', 'feature_mask_prob',
            'domain_embedding_dim', 'covariate_embedding_dims',
            'decoder_final_activation', 'decoder_weight', 'clr_weight',
            'classifier_weight', 'unlabeled_class', 'use_importance_mask',
            'importance_penalty_weight', 'importance_penalty_type',
            'dropout_prob', 'norm_type', 'knn_warmup_epochs',
            'sampler_knn', 'sampler_emb', 'sampler_domain_minibatch_strategy',
            'domain_coverage', 'dist_metric', 'use_faiss', 'use_ivf',
            'ivf_nprobe', 'preload_dense', 'num_workers', 'chunked', 'chunk_size'
        }
        
        for param_name in concord_param_names:
            if param_name in self.params:
                concord_params[param_name] = self.params[param_name]
        
        # Initialize CONCORD
        logger.info("Initializing CONCORD model...")
        concord = Concord(
            adata=adata,
            save_dir=str(self.save_dir) if self.save_dir else None,
            copy_adata=False,  # Work directly on the adata object
            verbose=True,
            **concord_params
        )
        
        # Train the model
        logger.info(f"Training CONCORD for {self.n_epochs} epochs...")
        concord.fit_transform(
            output_key=self.output_key,
            return_decoded=False,
            return_class=False,
            return_class_prob=False,
            save_model=(self.save_dir is not None)
        )
        
        # Extract embeddings from adata
        if self.output_key not in adata.obsm:
            raise ValueError(
                f"CONCORD embeddings not found in adata.obsm['{self.output_key}']. "
                "Check CONCORD training logs for errors."
            )
        
        embeddings = adata.obsm[self.output_key].copy()
        logger.info(f"CONCORD embeddings extracted. Shape: {embeddings.shape}")
        
        return embeddings
