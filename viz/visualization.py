"""
Embedding visualization (UMAP, PCA) for method embeddings.

This module produces UMAP plots colored by batch and label only. Batch-effect
metric values (ASW_batch, iLISI, cLISI, etc.) are not plotted here; they are
shown in original range and direction in the summarizer boxplots and tables
(see utils/results_summarizer.py and utils/boxplot_generator.py).
"""
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import anndata as ad
import scanpy as sc
import logging
logger = logging.getLogger('ml_logger')
from utils.logs_ import get_logger

from utils.sampling import sample_adata
logger = get_logger()

# Keys that are cached for PCA/UMAP
PCA_UMAP_KEYS = ['X_pca', 'X_umap', 'X_pca_umap']


class EmbeddingVisualizer:
    """Class for visualizing embeddings with optional batch and label coloring."""
    def __init__(self, embedding, obs, save_dir=".", auto_subsample=True):
        
        self.auto_subsample=auto_subsample
        self.embedding = embedding
            
        self.obs = obs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_pca_umap_from_cache(self, adata_original, cache_path):
        """
        Load cached PCA/UMAP and align with current adata by cell index.
        
        Args:
            adata_original: AnnData object to load PCA/UMAP into
            cache_path: Path to the cached PCA/UMAP h5ad file
            
        Returns:
            bool: True if cache was loaded successfully, False otherwise
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return False
        
        try:
            logger.info(f'Loading cached PCA/UMAP from: {cache_path}')
            cached_adata = ad.read_h5ad(cache_path)
            
            # Check if cell indices match (allowing for QC filtering)
            cached_cells = set(cached_adata.obs.index)
            current_cells = set(adata_original.obs.index)
            
            # Check if current cells are a subset of cached cells
            if not current_cells.issubset(cached_cells):
                logger.warning('Current cells are not a subset of cached cells. Recomputing PCA/UMAP.')
                return False
            
            # Align by cell index and load into adata
            for key in PCA_UMAP_KEYS:
                if key in cached_adata.obsm:
                    # Align by cell index
                    adata_original.obsm[key] = cached_adata[adata_original.obs.index].obsm[key]
                    logger.info(f'Loaded {key} from cache (shape: {adata_original.obsm[key].shape})')
            
            return True
            
        except Exception as e:
            logger.warning(f'Failed to load PCA/UMAP cache: {e}. Recomputing.')
            return False

    def save_pca_umap_to_cache(self, adata_original, cache_path):
        """
        Save PCA/UMAP to a lightweight cache file.
        
        Saves only cell indices and obsm entries (PCA, UMAP) for efficient reuse.
        
        Args:
            adata_original: AnnData object with computed PCA/UMAP
            cache_path: Path to save the cache file
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create lightweight adata with just cell indices and PCA/UMAP obsm entries
        cache_adata = ad.AnnData(obs=adata_original.obs[[]])  # Empty obs, keeps index
        
        for key in PCA_UMAP_KEYS:
            if key in adata_original.obsm:
                cache_adata.obsm[key] = adata_original.obsm[key]
        
        # Add metadata
        cache_adata.uns['cache_info'] = {
            'n_cells': adata_original.n_obs,
            'keys': list(cache_adata.obsm.keys()),
        }
        
        cache_adata.write_h5ad(cache_path)
        logger.info(f'Saved PCA/UMAP cache to: {cache_path}')
        logger.info(f'  Cached keys: {list(cache_adata.obsm.keys())}')

    def compute_pca_umap(self, adata_original):
        '''
        Compute PCA on original adata (with subsampling for large datasets).
        This is dataset-dependent, not method-dependent.
        
        Computes:
        1. X_pca: PCA on original expression data (subsampled if dataset is large)
        
        Note: UMAP on original data is skipped to save computation time.
        '''
        # 1. Compute PCA on original expression data (if not already computed)
        if 'X_pca' not in adata_original.obsm:
            logger.info('Computing PCA on original expression data (dataset-dependent)')
            
            # Subsample if dataset is large to speed up PCA computation
            n_cells = adata_original.n_obs
            max_cells_for_pca = 50000  # Subsample if more than 50k cells
            
            if n_cells > max_cells_for_pca:
                logger.info(f'Dataset has {n_cells} cells. Subsample to {max_cells_for_pca} for PCA computation.')
                # Import for sparse matrix handling
                from scipy import sparse
                import numpy as np
                
                # Create a subsampled copy for PCA computation
                adata_subsampled = sample_adata(adata_original, sample_size=max_cells_for_pca, stratify_by=None)
                
                # Compute mean from subsampled data (matching scanpy's zero_center=True behavior)
                # This mean will be used to center both subsample (via sc.tl.pca) and full dataset
                X_subsampled = adata_subsampled.X
                if sparse.issparse(X_subsampled):
                    # For sparse matrices, compute mean efficiently without full conversion
                    # Mean = sum / count, where count accounts for zeros
                    pca_mean = np.array(X_subsampled.mean(axis=0)).flatten()
                else:
                    pca_mean = X_subsampled.mean(axis=0)
                
                # Compute PCA on subsample (scanpy will center internally using zero_center=True by default)
                # Note: sc.tl.pca centers the data internally, so the mean we computed should match
                sc.tl.pca(adata_subsampled, n_comps=50, zero_center=True)
                
                # Get the PCA components from the subsample
                # These are stored in adata_subsampled.varm['PCs']
                pca_model = adata_subsampled.varm['PCs']  # Already a numpy array
                
                # Apply PCA transformation to full dataset
                # Use math trick: (X - mean) @ PCs = X @ PCs - mean @ PCs
                # This avoids converting sparse to dense (centering destroys sparsity)
                X_full = adata_original.X
                
                # Precompute the mean projection (small vector of size n_pcs)
                mean_projection = pca_mean @ pca_model  # shape: (n_pcs,)
                
                logger.info(f'Projecting {n_cells} cells onto PCA (keeping sparse)')
                
                if sparse.issparse(X_full):
                    # Sparse @ dense multiplication is efficient and memory-friendly
                    X_pca_full = X_full @ pca_model - mean_projection
                else:
                    X_pca_full = np.asarray(X_full) @ pca_model - mean_projection
                
                X_pca_full = X_pca_full.astype(np.float32)
                
                # Store in original adata
                adata_original.obsm['X_pca'] = X_pca_full
                # Also store the PCA model components for potential reuse
                adata_original.varm['PCs'] = pca_model
                adata_original.var['mean'] = pd.Series(pca_mean, index=adata_original.var.index)
                
                logger.info(f'Saved PCA to X_pca in original adata (computed on {max_cells_for_pca} subsampled cells, applied to all {n_cells} cells)')
            else:
                sc.tl.pca(adata_original, n_comps=50)
                logger.info('Saved PCA to X_pca in original adata')
        else:
            logger.info('PCA on original data already exists in dataset, skipping computation')
        
        # UMAP on original data is skipped to save computation time

    def save_pca_umap_to_adata(self, adata_original, cache_path=None):
        '''
        Load or compute PCA and UMAP on original adata, with optional caching.
        
        If cache_path is provided:
        1. Try to load from cache
        2. If cache doesn't exist or is incompatible, compute and save to cache
        
        Args:
            adata_original: AnnData object to compute/load PCA/UMAP for
            cache_path: Optional path to cache file for PCA/UMAP
        '''
        # Try to load from cache first
        if cache_path is not None:
            if self.load_pca_umap_from_cache(adata_original, cache_path):
                logger.info('Successfully loaded PCA/UMAP from cache')
                return
        
        # Compute PCA/UMAP
        self.compute_pca_umap(adata_original)
        
        # Save to cache if path provided
        if cache_path is not None:
            self.save_pca_umap_to_cache(adata_original, cache_path)

    def plot(self, adata_original=None, embedding_key=None, save_to_adata=True, pca_umap_cache_path=None):
        """
        Generate and save **PRE** embedding visualizations for all methods.
        
        Behavior:
        - Computes (or loads from cache) PCA on the original expression data and stores in
          ``adata_original.obsm['X_pca']`` (dataset-dependent, not method-dependent).
        - Builds a UMAP on the **pre-integration representation**:
          - Prefer ``X_pca`` if present, otherwise use the raw expression ``X``.
        - Saves UMAP plots in ``plots/embeddings``:
          - ``embedding_batch.png``  (colored by ``obs['batch']``)
          - ``embedding_label.png`` (colored by ``obs['label']``)
        
        The **POST** visualizations for each method are handled separately by
        ``BatchEffectsEvaluator.visualize_batch_effects`` (``umap_*_post.png``),
        so we do not duplicate POST plots here.
        
        Args:
            adata_original: Original AnnData object (required for PRE visualization).
            embedding_key: Unused here; kept for API compatibility.
            save_to_adata: If True and adata_original provided, save PCA to original adata.
            pca_umap_cache_path: Path to cache PCA (dataset-specific, reused across methods).
        """
        if adata_original is None:
            logger.warning(
                "EmbeddingVisualizer.plot called without adata_original; "
                "skipping PRE visualization."
            )
            return

        if save_to_adata:
            # Compute or load PCA (dataset-dependent, not method-dependent)
            self.save_pca_umap_to_adata(adata_original, cache_path=pca_umap_cache_path)

        # Build a PRE UMAP using original data (prefer X_pca if available)
        adata_pre = adata_original.copy()
        if self.auto_subsample and adata_pre.n_obs > 10000:
            adata_pre = sample_adata(adata_pre, sample_size=5000, stratify_by=None)

        if 'X_pca' in adata_pre.obsm:
            use_rep = 'X_pca'
            logger.info("Computing PRE UMAP using X_pca")
        else:
            use_rep = 'X'
            logger.info("Computing PRE UMAP using X")

        sc.pp.neighbors(adata_pre, use_rep=use_rep, n_neighbors=15)
        sc.tl.umap(adata_pre)

        # Save PRE UMAPs colored by batch and label
        if 'batch' in adata_pre.obs.columns:
            fig = sc.pl.umap(
                adata_pre,
                color='batch',
                show=False,
                wspace=0.4,
                frameon=False,
                return_fig=True,
            )
            fig.savefig(self.save_dir / 'embedding_batch.png', dpi=200, bbox_inches='tight')
            logger.info("Saved PRE UMAP plot colored by batch to 'embedding_batch.png'")

        if 'label' in adata_pre.obs.columns:
            fig = sc.pl.umap(
                adata_pre,
                color='label',
                show=False,
                wspace=0.4,
                frameon=False,
                return_fig=True,
            )
            fig.savefig(self.save_dir / 'embedding_label.png', dpi=200, bbox_inches='tight')
            logger.info("Saved PRE UMAP plot colored by label to 'embedding_label.png'")

