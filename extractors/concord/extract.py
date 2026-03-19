#!/usr/bin/env python
"""CONCORD embedding extraction script.

Trains CONCORD on the dataset and extracts embeddings.
EXACTLY replicates how CONCORD trains in its official benchmark repository.

This extractor follows CONCORD's official training pipeline:
1. Feature selection using concord.ul.select_features() (or HVG)
2. Normalization + log1p (or uses pre-normalized data)
3. Training with official default hyperparameters
4. Checkpoint saving with config for reproducibility

Usage:
    python extract.py --input data.h5ad --output embeddings.npy \
        --batch_key batch --n_epochs 15 --latent_dim 100
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import scanpy as sc

# Add parent to path for base module
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_extract import BaseExtractor, create_argument_parser, run_extraction

import logging
logger = logging.getLogger(__name__)

try:
    from concord import Concord
    import concord.ul as ccd_ul  # CONCORD utilities for feature selection
    CONCORD_AVAILABLE = True
except ImportError:
    CONCORD_AVAILABLE = False
    logger.warning("CONCORD package not found. Please install it.")


class ConcordExtractor(BaseExtractor):
    """CONCORD embedding extractor - EXACT REPLICATION of official benchmark.
    
    Extract embeddings using CONCORD (Contrastive learning for single-cell data).
    This extractor replicates EXACTLY how CONCORD trains in its official GitHub repo.
    
    Key differences from other FM extractors:
    - CONCORD trains from scratch on each dataset (no pre-trained weights)
    - Uses feature selection via concord.ul.select_features() or HVG
    - Supports both training new models and loading saved checkpoints
    
    Supported Variants (matching CONCORD's benchmark pipeline):
    - 'concord_hcl': Hard Contrastive Learning (default) - p_intra_knn=0.0, clr_beta=1.0
    - 'concord_knn': k-NN sampling - p_intra_knn=0.3, clr_beta=0.0
    - 'contrastive': Naive contrastive (no batch correction) - no batch_key, clr_beta=0.0
    
    Training Pipeline (matches CONCORD's official benchmarks):
    1. Feature selection: select_features() or sc.pp.highly_variable_genes()
    2. Preprocessing: normalize_total + log1p (if not already done)
    3. Training: Contrastive learning with masking augmentation
    4. Saving: Model checkpoint + config for reproducibility
    """
    
    # Variant configurations (matching pipeline_integration.py)
    VARIANT_CONFIGS = {
        'concord_hcl': {'p_intra_knn': 0.0, 'clr_beta': 1.0},
        'concord_knn': {'p_intra_knn': 0.3, 'clr_beta': 0.0},
        'contrastive': {'p_intra_knn': 0.0, 'clr_beta': 0.0, 'ignore_batch': True},
        'concord': {'p_intra_knn': 0.0, 'clr_beta': 1.0},  # Default = concord_hcl
    }
    
    # Default hyperparameters from CONCORD's official benchmarks
    CONCORD_DEFAULT_PARAMS = {
        'n_epochs': 15,
        'latent_dim': 100,
        'batch_size': 256,
        'lr': 1e-2,
        'train_frac': 1.0,
        'encoder_dims': [1000],
        'decoder_dims': [1000],
        'use_decoder': False,
        'clr_temperature': 0.4,
        'clr_beta': 1.0,  # Hard negative mining (set to 0 to disable)
        'clr_weight': 1.0,
        'p_intra_knn': 0.0,  # k-NN sampling (set > 0 to enable)
        'p_intra_domain': 1.0,
        'element_mask_prob': 0.4,  # Masking augmentation
        'feature_mask_prob': 0.3,
        'dropout_prob': 0.0,
        'norm_type': 'layer_norm',
        'schedule_ratio': 0.97,
        'knn_warmup_epochs': 2,
        'normalize_total': False,  # Usually data is pre-normalized
        'log1p': False,
        'seed': 0,
    }
    
    # Parameters that must be numeric (YAML can parse scientific notation as strings)
    NUMERIC_PARAMS = {
        'n_epochs': int,
        'latent_dim': int,
        'batch_size': int,
        'lr': float,
        'train_frac': float,
        'clr_temperature': float,
        'clr_beta': float,
        'clr_weight': float,
        'p_intra_knn': float,
        'p_intra_domain': float,
        'element_mask_prob': float,
        'feature_mask_prob': float,
        'dropout_prob': float,
        'schedule_ratio': float,
        'knn_warmup_epochs': int,
        'seed': int,
    }
    
    def __init__(
        self,
        params: dict | None = None,
        batch_key: str | None = None,
        domain_key: str | None = None,
        n_epochs: int = 15,
        latent_dim: int = 100,
        batch_size: int = 256,
        lr: float = 1e-2,
        train_frac: float = 1.0,
        save_dir: str | None = None,
        output_key: str = "X_concord",
        pretrained_model: str | None = None,  # Path to load pre-trained model
        n_top_genes: int | None = None,  # Feature selection: number of HVGs
        use_concord_feature_selection: bool = False,  # Use concord.ul.select_features()
        input_feature: List[str] | None = None,  # Explicit feature list
        **kwargs
    ):
        """Initialize CONCORD extractor.
        
        Args:
            params: YAML config dict (alternative to direct args)
            batch_key: Key in adata.obs for batch/domain information
            domain_key: Alias for batch_key (CONCORD naming)
            n_epochs: Number of training epochs
            latent_dim: Latent dimension size
            batch_size: Training batch size
            lr: Learning rate
            train_frac: Fraction of data to use for training
            save_dir: Directory to save model checkpoints
            output_key: Key for storing embeddings in adata.obsm
            pretrained_model: Path to pre-trained model checkpoint
            n_top_genes: Number of highly variable genes to select
            use_concord_feature_selection: Use CONCORD's feature selection
            input_feature: Explicit list of features to use
            **kwargs: Additional CONCORD parameters
            
        Variant Support:
            Set `variant` in params to use predefined configurations:
            - 'concord_hcl': Hard Contrastive Learning (p_intra_knn=0, clr_beta=1)
            - 'concord_knn': k-NN sampling (p_intra_knn=0.3, clr_beta=0)
            - 'contrastive': Naive contrastive, no batch (clr_beta=0)
        """
        # Support both initialization styles
        if params is not None:
            # YAML config style: extract from params dict
            super().__init__(params=params)
            self.batch_key = self.params.get('batch_key') or self.params.get('domain_key')
            if not self.batch_key:
                logger.warning("No batch_key or domain_key provided - CONCORD will treat all cells as single batch")
                self.batch_key = None
            
            # Training parameters (with CONCORD defaults)
            # Apply type conversion to handle YAML parsing issues (e.g., '1e-2' as string)
            for key, default_val in self.CONCORD_DEFAULT_PARAMS.items():
                raw_value = self.params.get(key, default_val)
                setattr(self, key, self._convert_param_type(key, raw_value))
            
            # Apply variant-specific config if specified
            variant = self.params.get('variant', None)
            self._apply_variant_config(variant)
            
            # Feature selection
            self.n_top_genes = self.params.get('n_top_genes', None)
            self.use_concord_feature_selection = self.params.get('use_concord_feature_selection', False)
            self.input_feature = self.params.get('input_feature', None)
            
            # Model loading
            self.pretrained_model = self.params.get('pretrained_model', None)
            
            # Output key
            self.output_key = self.params.get('output_key', 'X_concord')
            
            # Device
            self.device = self.params.get('device', None)
            
            # Class key for supervised mode
            self.class_key = self.params.get('class_key', None)
            
            # Save dir
            save_dir = self.params.get('save_dir', None)
        else:
            # CLI style: direct parameters
            super().__init__(
                params=None,
                batch_key=batch_key or domain_key,
                n_epochs=n_epochs,
                latent_dim=latent_dim,
                batch_size=batch_size,
                lr=lr,
                train_frac=train_frac,
                save_dir=save_dir,
                output_key=output_key,
                **kwargs
            )
            self.batch_key = batch_key or domain_key or kwargs.get('batch_key')
            
            # Set defaults from CONCORD_DEFAULT_PARAMS, override with kwargs
            # Apply type conversion to handle potential string values
            for key, default_val in self.CONCORD_DEFAULT_PARAMS.items():
                raw_value = kwargs.get(key, default_val)
                setattr(self, key, self._convert_param_type(key, raw_value))
            
            # Apply variant-specific config if specified
            variant = kwargs.get('variant', None)
            self._apply_variant_config(variant)
            
            # Override explicit args (with type conversion)
            self.n_epochs = self._convert_param_type('n_epochs', n_epochs)
            self.latent_dim = self._convert_param_type('latent_dim', latent_dim)
            self.batch_size = self._convert_param_type('batch_size', batch_size)
            self.lr = self._convert_param_type('lr', lr)
            self.train_frac = self._convert_param_type('train_frac', train_frac)
            self.output_key = output_key
            
            # Feature selection
            self.n_top_genes = n_top_genes
            self.use_concord_feature_selection = use_concord_feature_selection
            self.input_feature = input_feature
            
            # Model loading
            self.pretrained_model = pretrained_model
            
            # Device
            self.device = kwargs.get('device', None)
            
            # Class key
            self.class_key = kwargs.get('class_key', None)
        
        # Save directory for model checkpoints
        if save_dir:
            self.save_dir = Path(save_dir) / 'concord_model'
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"CONCORD model will be saved to: {self.save_dir}")
        else:
            self.save_dir = None
            logger.warning("save_dir not provided. Model checkpoints will not be saved.")
        
        # Store the trained Concord object for potential reuse
        self._concord_model = None
        
        # Store variant name for logging
        self._variant = getattr(self, '_variant', 'concord')
    
    def _convert_param_type(self, key: str, value):
        """Convert parameter to correct type (YAML may parse scientific notation as string).
        
        Args:
            key: Parameter name
            value: Parameter value (may be string from YAML)
            
        Returns:
            Value converted to the correct type
        """
        if value is None:
            return value
        
        if key in self.NUMERIC_PARAMS:
            target_type = self.NUMERIC_PARAMS[key]
            if not isinstance(value, target_type):
                try:
                    converted = target_type(value)
                    if str(value) != str(converted):
                        logger.debug(f"Converted {key}: {value!r} ({type(value).__name__}) -> {converted} ({target_type.__name__})")
                    return converted
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert {key}={value!r} to {target_type.__name__}: {e}")
                    return value
        return value
    
    def _apply_variant_config(self, variant: str | None) -> None:
        """Apply variant-specific configuration.
        
        Args:
            variant: One of 'concord_hcl', 'concord_knn', 'contrastive', or None
        """
        if variant is None:
            return
        
        variant = variant.lower()
        if variant not in self.VARIANT_CONFIGS:
            logger.warning(f"Unknown variant '{variant}'. Using default config.")
            return
        
        config = self.VARIANT_CONFIGS[variant]
        self._variant = variant
        
        logger.info(f"Applying CONCORD variant config: {variant}")
        
        for key, value in config.items():
            if key == 'ignore_batch':
                if value:
                    logger.info(f"  {variant}: Ignoring batch key (contrastive mode)")
                    self.batch_key = None
            else:
                logger.info(f"  {variant}: Setting {key}={value}")
                setattr(self, key, value)
    
    @property
    def model_name(self) -> str:
        return f"CONCORD-{self._variant}" if hasattr(self, '_variant') else "CONCORD"
    
    @property
    def embedding_dim(self) -> int:
        return self.latent_dim
    
    def load_model(self) -> None:
        """Load pre-trained CONCORD model if specified, otherwise prepare for training."""
        if not CONCORD_AVAILABLE:
            raise ImportError("CONCORD package not found. Please install it.")
        
        if self.pretrained_model and not self._model_loaded:
            pretrained_path = Path(self.pretrained_model)
            if pretrained_path.exists():
                logger.info(f"Loading pre-trained CONCORD model from: {pretrained_path}")
                self._concord_model = Concord.load(str(pretrained_path))
                logger.info("Pre-trained CONCORD model loaded successfully")
            else:
                logger.warning(f"Pre-trained model not found at {pretrained_path}. Will train from scratch.")
        
        if not self._model_loaded:
            logger.info("CONCORD extractor ready")
            self._model_loaded = True
    
    def _select_features(self, adata: sc.AnnData) -> Optional[List[str]]:
        """Select features using CONCORD's official methods.
        
        Order of priority:
        1. Explicit input_feature list (if provided)
        2. use_concord_feature_selection (uses concord.ul.select_features)
        3. n_top_genes HVGs (uses sc.pp.highly_variable_genes)
        4. Use existing HVGs if available
        5. Use all features (CONCORD recommends feature selection)
        
        Returns:
            List of feature names or None (use all features)
        """
        if self.input_feature is not None:
            logger.info(f"Using explicit feature list: {len(self.input_feature)} features")
            return self.input_feature
        
        if self.use_concord_feature_selection and CONCORD_AVAILABLE:
            try:
                logger.info("Using CONCORD's feature selection (concord.ul.select_features)")
                # CONCORD's official feature selection
                selected_genes = ccd_ul.select_features(
                    adata,
                    n_top_genes=self.n_top_genes or 4000,
                    batch_key=self.batch_key,
                )
                logger.info(f"CONCORD selected {len(selected_genes)} features")
                return selected_genes
            except Exception as e:
                logger.warning(f"CONCORD feature selection failed: {e}. Falling back to HVG.")
        
        if self.n_top_genes is not None:
            logger.info(f"Selecting {self.n_top_genes} highly variable genes (seurat_v3)")
            # Use CONCORD's recommended HVG method
            if 'highly_variable' not in adata.var.columns:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=self.n_top_genes,
                    flavor='seurat_v3',
                    batch_key=self.batch_key if self.batch_key in adata.obs else None,
                    subset=False,
                )
            hvgs = adata.var_names[adata.var['highly_variable']].tolist()
            logger.info(f"Selected {len(hvgs)} HVGs")
            return hvgs
        
        if 'highly_variable' in adata.var.columns:
            hvgs = adata.var_names[adata.var['highly_variable']].tolist()
            logger.info(f"Using existing HVGs: {len(hvgs)} features")
            return hvgs
        
        logger.warning(
            "No feature selection applied. CONCORD recommends selecting features. "
            "Set n_top_genes or use_concord_feature_selection=True for best results."
        )
        return None
    
    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Train CONCORD model and extract embeddings.
        
        Follows CONCORD's official benchmark pipeline:
        1. Feature selection (if configured)
        2. Preprocessing check
        3. Training or loading pre-trained model
        4. Embedding extraction
        
        Args:
            adata: AnnData object
            
        Returns:
            Embeddings array of shape (n_cells, latent_dim)
        """
        if not CONCORD_AVAILABLE:
            raise ImportError("CONCORD package not found.")
        
        logger.info(f"Processing CONCORD on dataset with shape: {adata.shape}")
        logger.info(f"Batch key: {self.batch_key}, Train fraction: {self.train_frac}")
        
        # ================================================================
        # Step 1: Feature Selection (CONCORD official method)
        # ================================================================
        selected_features = self._select_features(adata)
        
        # ================================================================
        # Step 2: Preprocessing Check
        # ================================================================
        # Check if data is already preprocessed
        is_preprocessed = (
            hasattr(adata, 'uns') and 
            adata.uns.get('preprocessed', False)
        )
        
        # Check if data looks normalized (CONCORD expects log-normalized data)
        if hasattr(adata, 'X') and adata.X is not None:
            from scipy import sparse
            X_sample = adata.X[:100] if sparse.issparse(adata.X) else adata.X[:100]
            if sparse.issparse(X_sample):
                X_sample = X_sample.toarray()
            max_val = np.max(X_sample)
            
            # If max > 20, data is likely not log-transformed
            if max_val > 20:
                logger.warning(
                    f"Data max value is {max_val:.1f}. CONCORD expects log-normalized data. "
                    "Consider setting normalize_total=True and log1p=True."
                )
        
        if is_preprocessed:
            logger.info("Data is already preprocessed. CONCORD will use it as-is.")
            normalize_total = False
            log1p = False
        else:
            normalize_total = self.normalize_total
            log1p = self.log1p
        
        # ================================================================
        # Step 3: Load Pre-trained or Train New Model
        # ================================================================
        
        if self._concord_model is not None:
            # Use pre-trained model
            logger.info("Using pre-trained CONCORD model for embedding extraction")
            self._concord_model.predict_adata(
                adata,
                output_key=self.output_key,
                return_decoded=False,
                domain_key=self.batch_key,
            )
        else:
            # Train new model
            # Prepare CONCORD parameters (EXACT match to official benchmarks)
            concord_params = {
                'domain_key': self.batch_key,
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
                'clr_weight': self.clr_weight,
                'p_intra_knn': self.p_intra_knn,
                'p_intra_domain': self.p_intra_domain,
                'element_mask_prob': self.element_mask_prob,
                'feature_mask_prob': self.feature_mask_prob,
                'dropout_prob': self.dropout_prob,
                'norm_type': self.norm_type,
                'schedule_ratio': self.schedule_ratio,
                'knn_warmup_epochs': self.knn_warmup_epochs,
                'normalize_total': normalize_total,
                'log1p': log1p,
                'seed': self.seed,
                'input_feature': selected_features,  # Use selected features
            }
            
            # Add optional parameters
            if self.class_key:
                concord_params['class_key'] = self.class_key
                concord_params['use_classifier'] = True
            
            if self.device:
                concord_params['device'] = self.device
            
            # Add any other CONCORD parameters from params dict
            additional_params = {
                'domain_embedding_dim', 'covariate_embedding_dims',
                'decoder_final_activation', 'decoder_weight',
                'classifier_weight', 'unlabeled_class', 'use_importance_mask',
                'importance_penalty_weight', 'importance_penalty_type',
                'sampler_knn', 'sampler_emb', 'sampler_domain_minibatch_strategy',
                'domain_coverage', 'dist_metric', 'use_faiss', 'use_ivf',
                'ivf_nprobe', 'preload_dense', 'num_workers', 'chunked', 'chunk_size'
            }
            
            if hasattr(self, 'params') and self.params:
                for param_name in additional_params:
                    if param_name in self.params:
                        concord_params[param_name] = self.params[param_name]
            
            # Initialize CONCORD
            logger.info("Initializing CONCORD model...")
            logger.info(f"CONCORD params: n_epochs={self.n_epochs}, latent_dim={self.latent_dim}, "
                       f"clr_beta={self.clr_beta}, clr_temp={self.clr_temperature}")
            
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
            
            # Store for potential reuse
            self._concord_model = concord
        
        # ================================================================
        # Step 4: Extract Embeddings
        # ================================================================
        if self.output_key not in adata.obsm:
            raise ValueError(
                f"CONCORD embeddings not found in adata.obsm['{self.output_key}']. "
                "Check CONCORD training logs for errors."
            )
        
        embeddings = adata.obsm[self.output_key].copy()
        logger.info(f"CONCORD embeddings extracted. Shape: {embeddings.shape}")
        
        # Save extraction metadata
        if self.save_dir:
            metadata = {
                'model_name': 'CONCORD',
                'n_epochs': self.n_epochs,
                'latent_dim': self.latent_dim,
                'clr_beta': self.clr_beta,
                'clr_temperature': self.clr_temperature,
                'n_features': len(selected_features) if selected_features else adata.n_vars,
                'n_cells': adata.n_obs,
                'embedding_shape': list(embeddings.shape),
            }
            with open(self.save_dir / 'extraction_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return embeddings


def main():
    parser = create_argument_parser()
    
    # CONCORD-specific arguments
    parser.add_argument(
        "--batch_key",
        type=str,
        required=True,
        help="Key in adata.obs for batch/domain information"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=100,
        help="Latent dimension (default: 100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size (default: 256)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate (default: 1e-2)"
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=1.0,
        help="Fraction of data to use for training (default: 1.0)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (optional)"
    )
    
    args = parser.parse_args()
    
    extractor = ConcordExtractor(
        batch_key=args.batch_key,
        n_epochs=args.n_epochs,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        train_frac=args.train_frac,
        save_dir=args.save_dir,
    )
    
    run_extraction(
        extractor,
        args.input,
        args.output,
        save_metadata=args.save_metadata,
    )


if __name__ == "__main__":
    main()
