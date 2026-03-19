"""
Baseline embedding methods for batch effects evaluation.

These methods provide reference points for evaluating batch correction:
- Best case: Perfect cell type separation
- Worst case: No correction or batch-specific processing
- Random baselines: Shuffled embeddings
- Integration methods: scVI, scANVI, BBKNN, Harmony (scib benchmarks)
"""
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, Dict, Any
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.logs_ import get_logger

logger = get_logger()


# =============================================================================
# Integration baseline methods (scVI, scANVI, BBKNN, Harmony)
# These are the standard benchmarks from scib
# =============================================================================

def _run_scvi(
    adata: AnnData,
    batch_key: str,
    n_latent: int = 30,
    max_epochs: int = 400,
    early_stopping: bool = True,
    random_seed: int = 42,
    num_workers: int = 4
) -> np.ndarray:
    """
    Run scVI for batch integration.
    
    Args:
        adata: AnnData object with counts in .X or .layers['counts']
        batch_key: Key in adata.obs for batch labels
        n_latent: Number of latent dimensions
        max_epochs: Maximum training epochs
        early_stopping: Whether to use early stopping
        random_seed: Random seed
        num_workers: Number of DataLoader workers for faster data loading
    
    Returns:
        Latent representation (n_cells, n_latent)
    """
    try:
        import scvi
    except ImportError:
        raise ImportError("scvi-tools is required for scVI baseline. Install with: pip install scvi-tools")
    
    # Set random seed (try scvi.settings first, fallback to direct seed setting)
    try:
        scvi.settings.seed = random_seed
    except AttributeError:
        # Newer scvi-tools versions: use direct seed setting
        from utils.rand import set_random_seeds
        set_random_seeds(random_seed)
        logger.info(f"Set random seed {random_seed} via direct seed setting (scvi.settings not available)")
    
    # Configure DataLoader workers for faster data loading
    scvi.settings.dl_num_workers = num_workers
    logger.info(f"Set scVI DataLoader num_workers to {num_workers}")
    
    # Create a copy for scVI
    adata_scvi = adata.copy()
    
    # Determine which layer to use for counts
    # scVI expects raw counts
    if 'counts' in adata_scvi.layers:
        layer = 'counts'
        logger.info("Using 'counts' layer for scVI")
    elif adata_scvi.X is not None:
        # Check if X looks like counts (integers or near-integers)
        X_sample = adata_scvi.X[:100] if adata_scvi.n_obs > 100 else adata_scvi.X
        if hasattr(X_sample, 'toarray'):
            X_sample = X_sample.toarray()
        is_counts = np.allclose(X_sample, np.round(X_sample), rtol=0.01)
        if is_counts:
            layer = None  # Use X directly
            logger.info("Using X matrix for scVI (appears to be counts)")
        else:
            # Try to find counts layer with different names
            for counts_name in ['raw_counts', 'raw', 'spliced', 'unspliced']:
                if counts_name in adata_scvi.layers:
                    layer = counts_name
                    logger.info(f"Using '{counts_name}' layer for scVI")
                    break
            else:
                logger.warning("No counts layer found. scVI may produce suboptimal results on normalized data.")
                layer = None
    else:
        layer = None
    
    # Setup scVI
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        layer=layer,
        batch_key=batch_key
    )
    
    # Compute adaptive max_epochs based on dataset size (as in scib benchmark)
    adaptive_epochs = int(np.min([round((20000 / adata.n_obs) * 400), max_epochs]))
    logger.info(f"Training scVI for {adaptive_epochs} epochs (dataset size: {adata.n_obs})")
    
    # Create and train model
    model = scvi.model.SCVI(adata_scvi, n_latent=n_latent)
    model.train(
        max_epochs=adaptive_epochs,
        early_stopping=early_stopping,
        early_stopping_patience=25,
        check_val_every_n_epoch=1,
        train_size=0.9
    )
    
    # Get latent representation
    latent = model.get_latent_representation()
    logger.info(f"scVI latent representation shape: {latent.shape}")
    
    return latent.astype(np.float32)


def _run_scanvi(
    adata: AnnData,
    batch_key: str,
    label_key: str,
    n_latent: int = 30,
    max_epochs_scvi: int = 400,
    max_epochs_scanvi: int = 20,
    early_stopping: bool = True,
    random_seed: int = 42
) -> np.ndarray:
    """
    Run scANVI for batch integration (semi-supervised).
    
    Args:
        adata: AnnData object with counts
        batch_key: Key in adata.obs for batch labels
        label_key: Key in adata.obs for cell type labels
        n_latent: Number of latent dimensions
        max_epochs_scvi: Maximum training epochs for scVI pretraining
        max_epochs_scanvi: Maximum training epochs for scANVI
        early_stopping: Whether to use early stopping
        random_seed: Random seed
    
    Returns:
        Latent representation (n_cells, n_latent)
    """
    try:
        import scvi
    except ImportError:
        raise ImportError("scvi-tools is required for scANVI baseline. Install with: pip install scvi-tools")
    
    # Set random seed (try scvi.settings first, fallback to direct seed setting)
    try:
        scvi.settings.seed = random_seed
    except AttributeError:
        # Newer scvi-tools versions: use direct seed setting
        from utils.rand import set_random_seeds
        set_random_seeds(random_seed)
    
    # Create a copy for scANVI
    adata_scanvi = adata.copy()
    
    # Determine which layer to use for counts
    if 'counts' in adata_scanvi.layers:
        layer = 'counts'
    else:
        layer = None
    
    # Setup for scVI first
    scvi.model.SCVI.setup_anndata(
        adata_scanvi,
        layer=layer,
        batch_key=batch_key
    )
    
    # Compute adaptive epochs
    adaptive_epochs_scvi = int(np.min([round((20000 / adata.n_obs) * 400), max_epochs_scvi]))
    
    # Train scVI first
    logger.info(f"Training scVI (pretraining for scANVI) for {adaptive_epochs_scvi} epochs")
    model_scvi = scvi.model.SCVI(adata_scanvi, n_latent=n_latent)
    model_scvi.train(
        max_epochs=adaptive_epochs_scvi,
        early_stopping=early_stopping,
        early_stopping_patience=25,
        check_val_every_n_epoch=1,
        train_size=0.9
    )
    
    # Create scANVI from scVI
    adaptive_epochs_scanvi = int(np.min([10, np.max([2, round(adaptive_epochs_scvi / 3.0)])]))
    logger.info(f"Training scANVI for {adaptive_epochs_scanvi} epochs")
    
    model_scanvi = scvi.model.SCANVI.from_scvi_model(
        model_scvi,
        labels_key=label_key,
        unlabeled_category="unlabelled"
    )
    model_scanvi.train(max_epochs=adaptive_epochs_scanvi)
    
    # Get latent representation
    latent = model_scanvi.get_latent_representation()
    logger.info(f"scANVI latent representation shape: {latent.shape}")
    
    return latent.astype(np.float32)


def _run_bbknn(
    adata: AnnData,
    batch_key: str,
    n_pcs: int = 50,
    neighbors_within_batch: int = 3,
    random_seed: int = 42
) -> np.ndarray:
    """
    Run BBKNN for batch integration.
    
    BBKNN operates on the neighbor graph, not embeddings.
    We return the PCA embedding but the neighbor graph is corrected.
    
    Args:
        adata: AnnData object
        batch_key: Key in adata.obs for batch labels
        n_pcs: Number of PCs to use
        neighbors_within_batch: Number of neighbors within each batch
        random_seed: Random seed
    
    Returns:
        PCA embedding (the integration is in the neighbor graph)
    """
    try:
        import bbknn
    except ImportError:
        raise ImportError("bbknn is required for BBKNN baseline. Install with: pip install bbknn")
    
    # Create a copy
    adata_bbknn = adata.copy()
    
    # BBKNN requires every batch to have at least neighbors_within_batch cells.
    # After subsampling, some batches may have only 1–2 cells; cap to min batch size.
    batch_counts = adata_bbknn.obs[batch_key].value_counts()
    min_batch_size = int(batch_counts.min())
    n_within = min(neighbors_within_batch, max(1, min_batch_size))
    if min_batch_size < 1:
        raise ValueError(
            "BBKNN requires every batch to have at least 1 cell. "
            f"Found batch with 0 cells (batch_key={batch_key})."
        )
    if n_within < neighbors_within_batch:
        logger.warning(
            f"Some batches have only {min_batch_size} cell(s). "
            f"Using neighbors_within_batch={n_within} (requested {neighbors_within_batch})."
        )
    
    # Compute PCA if not present
    if 'X_pca' not in adata_bbknn.obsm:
        logger.info("Computing PCA for BBKNN")
        sc.tl.pca(adata_bbknn, n_comps=n_pcs, random_state=random_seed)
    
    # Run BBKNN
    logger.info(f"Running BBKNN with {n_within} neighbors within batch")
    bbknn.bbknn(
        adata_bbknn,
        batch_key=batch_key,
        n_pcs=n_pcs,
        neighbors_within_batch=n_within
    )
    
    # Return PCA embedding (BBKNN modifies the neighbor graph)
    # Note: Metrics using neighbor graph will use the corrected graph
    return adata_bbknn.obsm['X_pca'].astype(np.float32)


def _run_harmony(
    adata: AnnData,
    batch_key: str,
    n_pcs: int = 50,
    max_iter: int = 50,
    random_seed: int = 42
) -> np.ndarray:
    """
    Run Harmony for batch integration (same pattern as Concord run_harmony_and_evaluate.py).
    
    Uses harmonypy directly: run_harmony(pcs, meta, batch_key, random_state=...).
    Batch column is cast to categorical/string so harmonypy's describe().loc['unique']
    works with pandas 2.x (avoids KeyError when batch is numeric).
    
    Args:
        adata: AnnData object
        batch_key: Key in adata.obs for batch labels
        n_pcs: Number of PCs to use
        max_iter: Maximum iterations for Harmony
        random_seed: Random seed
    
    Returns:
        Harmony-corrected PCA embedding (n_cells, n_pcs)
    """
    try:
        import harmonypy as hm
    except ImportError:
        raise ImportError("Harmony is required. Install with: pip install harmonypy")

    adata_harmony = adata.copy()

    # Compute PCA if not present
    if "X_pca" not in adata_harmony.obsm:
        logger.info("Computing PCA for Harmony")
        sc.tl.pca(adata_harmony, n_comps=n_pcs, random_state=random_seed)

    pcs = adata_harmony.obsm["X_pca"]
    meta = adata_harmony.obs.copy()
    # Harmonypy uses meta[vars_use].describe().loc['unique']; pandas 2.x only
    # includes 'unique' for object/categorical. Cast batch to categorical so
    # describe() has 'unique' (same idea as Concord's categorical batch for plotting).
    if not isinstance(meta[batch_key].dtype, pd.CategoricalDtype):
        meta[batch_key] = pd.Categorical(meta[batch_key].astype(str))

    logger.info("Running Harmony via harmonypy (same as Concord run_harmony_and_evaluate.py)")
    harmony_out = hm.run_harmony(
        pcs,
        meta,
        batch_key,
        max_iter_harmony=max_iter,
        random_state=random_seed,
    )
    Z_corr = np.asarray(harmony_out.Z_corr)
    n_cells = adata_harmony.n_obs
    # obsm must be (n_obs, n_comps); harmonypy may return (n_comps, n_obs) or (n_obs, n_comps)
    if Z_corr.shape[0] == n_cells:
        result = Z_corr
    else:
        result = Z_corr.T
    return result.astype(np.float32)


def _run_scanorama(
    adata: AnnData,
    batch_key: str,
    random_seed: int = 42
) -> np.ndarray:
    """
    Run Scanorama for batch integration.
    
    Args:
        adata: AnnData object
        batch_key: Key in adata.obs for batch labels
        random_seed: Random seed
    
    Returns:
        Scanorama-corrected embedding
    """
    try:
        import scanorama
    except ImportError:
        raise ImportError("Scanorama is required. Install with: pip install scanorama")
    
    # Split by batch, run Scanorama, then reassemble using batch masks
    # (avoids concatenate which changes obs_names and breaks reindexing)
    batches = adata.obs[batch_key].unique().tolist()
    adatas = [adata[adata.obs[batch_key] == b].copy() for b in batches]

    logger.info(f"Running Scanorama on {len(batches)} batches")
    scanorama.integrate_scanpy(adatas, dimred=50)

    # Collect embeddings back in original row order via the same batch masks
    n_dims = adatas[0].obsm['X_scanorama'].shape[1]
    embedding = np.zeros((adata.n_obs, n_dims), dtype=np.float32)
    for batch, batch_adata in zip(batches, adatas):
        mask = (adata.obs[batch_key] == batch).values
        embedding[mask] = batch_adata.obsm['X_scanorama'].astype(np.float32)

    return embedding


def _run_pca_qc(
    adata: AnnData,
    batch_key: str,
    label_key: str = 'label',
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    random_seed: int = 42
) -> np.ndarray:
    """
    Run PCA baseline with full QC preprocessing (standard scanpy workflow BEFORE batch correction).
    
    This represents the "pre-batch-correction" baseline: what you get with standard
    scanpy preprocessing (normalize → log1p → HVG → scale → PCA) without any
    batch integration method.
    
    Pipeline:
    1. Check if data is normalized (normalize + log1p if needed)
    2. Select highly variable genes (batch-aware using seurat_v3 flavor)
    3. Scale (center + clip to max_value=10)
    4. PCA
    
    Args:
        adata: AnnData object (can be raw counts or already normalized)
        batch_key: Key in adata.obs for batch labels (used for batch-aware HVG)
        label_key: Key in adata.obs for cell type labels (not used in PCA, kept for consistency)
        n_hvgs: Number of highly variable genes to select
        n_pcs: Number of principal components
        random_seed: Random seed for reproducibility
    
    Returns:
        PCA embedding (n_cells, n_pcs)
    """
    import scipy.sparse as sp
    
    logger.info(f"Running PCA baseline with QC preprocessing (HVG={n_hvgs}, n_PCs={n_pcs})")
    
    # Work on a copy to avoid modifying original
    adata_pca = adata.copy()
    
    # Step 1: Check if normalization is needed
    # Heuristic: if max value > 50, data is likely raw counts
    X = adata_pca.X
    max_val = float(X.max()) if sp.issparse(X) else float(np.max(X))
    
    if max_val > 50:
        logger.info("Data appears to be raw counts (max > 50). Applying normalize_total + log1p")
        sc.pp.normalize_total(adata_pca, target_sum=1e4)
        sc.pp.log1p(adata_pca)
    else:
        logger.info(f"Data appears already normalized (max={max_val:.2f}). Skipping normalization.")
    
    # Step 2: Select highly variable genes (batch-aware)
    # Use seurat_v3 flavor which works well with batch effects
    try:
        logger.info(f"Selecting {n_hvgs} HVGs (batch-aware with batch_key='{batch_key}')")
        sc.pp.highly_variable_genes(
            adata_pca,
            n_top_genes=n_hvgs,
            batch_key=batch_key,
            flavor='seurat_v3',
            subset=False  # Don't subset yet, just mark
        )
        n_hvg_found = adata_pca.var['highly_variable'].sum()
        logger.info(f"Found {n_hvg_found} highly variable genes")
        
        # Subset to HVGs
        adata_pca = adata_pca[:, adata_pca.var['highly_variable']].copy()
    except Exception as e:
        logger.warning(f"Batch-aware HVG failed: {e}. Falling back to non-batch-aware HVG.")
        # Fallback: non-batch-aware HVG
        sc.pp.highly_variable_genes(
            adata_pca,
            n_top_genes=n_hvgs,
            flavor='seurat_v3',
            subset=True
        )
    
    # Step 3: Scale (center + clip)
    logger.info("Scaling data (zero_center=True, max_value=10)")
    sc.pp.scale(adata_pca, zero_center=True, max_value=10)
    
    # Step 4: PCA
    logger.info(f"Computing PCA with {n_pcs} components")
    sc.tl.pca(adata_pca, n_comps=n_pcs, random_state=random_seed)
    
    pca_embedding = adata_pca.obsm['X_pca'].astype(np.float32)
    logger.info(f"PCA baseline embedding shape: {pca_embedding.shape}")
    
    return pca_embedding


def _randomize_features(X: np.ndarray, partition: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Randomize features, optionally within partitions.
    
    Args:
        X: Feature matrix (n_cells, n_features)
        partition: Optional partition labels. If provided, randomization happens within each partition.
    
    Returns:
        Randomized feature matrix
    """
    X_out = X.copy()
    if partition is None:
        partition = np.full(X.shape[0], 0)
    else:
        partition = np.asarray(partition)
    
    for partition_name in np.unique(partition):
        partition_idx = np.argwhere(partition == partition_name).flatten()
        X_out[partition_idx] = X[np.random.permutation(partition_idx)]
    
    return X_out


def _perfect_embedding(partition: np.ndarray, jitter: float = 0.01) -> np.ndarray:
    """
    Create a perfect embedding based on partition labels (one-hot encoding).
    
    Args:
        partition: Partition labels (e.g., cell types)
        jitter: Amount of random jitter to add (None for no jitter)
    
    Returns:
        Embedding matrix (n_cells, n_partitions)
    """
    le = LabelEncoder()
    encoded = le.fit_transform(partition)[:, None]
    ohe = OneHotEncoder(sparse_output=False)
    embedding = ohe.fit_transform(encoded)
    
    if jitter is not None and jitter > 0:
        embedding = embedding + np.random.uniform(-jitter, jitter, embedding.shape)
    
    return embedding.astype(np.float32)


def create_baseline_embedding(
    adata: AnnData,
    method: str,
    batch_key: str = 'batch',
    label_key: str = 'label',
    use_rep: str = 'X',
    n_comps: int = 50,
    random_seed: int = 42,
    num_workers: int = 6
) -> np.ndarray:
    """
    Create a baseline embedding for batch effects evaluation.
    
    Args:
        adata: AnnData object with data
        method: Baseline method name. Options:
            - 'embed_cell_types': Perfect embedding based on cell types (best case for biology)
            - 'embed_cell_types_jittered': Perfect embedding with jitter
            - 'shuffle_integration': Randomize all features
            - 'shuffle_integration_by_batch': Randomize features within batches
            - 'shuffle_integration_by_cell_type': Randomize features within cell types
        batch_key: Key in adata.obs for batch labels
        label_key: Key in adata.obs for cell type/biological labels
        use_rep: Representation to use ('X', 'X_pca', or layer name)
        n_comps: Number of PCA components (for PCA-based methods)
        random_seed: Random seed for reproducibility
    
    Returns:
        Embedding matrix (n_cells, n_dims)
    """
    np.random.seed(random_seed)
    
    # Get data matrix
    if use_rep == 'X':
        X = adata.X
    elif use_rep in adata.obsm:
        X = adata.obsm[use_rep]
    elif use_rep in adata.layers:
        X = adata.layers[use_rep]
    else:
        raise ValueError(f"Representation '{use_rep}' not found in adata")
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X)
    
    if method == 'embed_cell_types':
        # Perfect embedding based on cell types (best case for biological signal)
        logger.info('Creating baseline: Perfect embedding based on cell types')
        return _perfect_embedding(adata.obs[label_key], jitter=None)
    
    elif method == 'embed_cell_types_jittered':
        # Perfect embedding with jitter
        logger.info('Creating baseline: Perfect embedding with jitter')
        return _perfect_embedding(adata.obs[label_key], jitter=0.01)
    
    elif method == 'shuffle_integration':
        # Randomize all features (random baseline)
        logger.info('Creating baseline: Shuffled features (random)')
        X_shuffled = _randomize_features(X)
        adata_temp = adata.copy()
        adata_temp.X = X_shuffled
        sc.tl.pca(adata_temp, n_comps=n_comps, random_state=random_seed)
        return adata_temp.obsm['X_pca']
    
    elif method == 'shuffle_integration_by_batch':
        # Randomize features within each batch (preserves batch structure)
        logger.info('Creating baseline: Shuffled features within batches')
        X_shuffled = _randomize_features(X, partition=adata.obs[batch_key].values)
        adata_temp = adata.copy()
        adata_temp.X = X_shuffled
        sc.tl.pca(adata_temp, n_comps=n_comps, random_state=random_seed)
        return adata_temp.obsm['X_pca']
    
    elif method == 'shuffle_integration_by_cell_type':
        # Randomize features within each cell type (preserves cell type structure)
        logger.info('Creating baseline: Shuffled features within cell types')
        X_shuffled = _randomize_features(X, partition=adata.obs[label_key].values)
        adata_temp = adata.copy()
        adata_temp.X = X_shuffled
        sc.tl.pca(adata_temp, n_comps=n_comps, random_state=random_seed)
        return adata_temp.obsm['X_pca']
    
    elif method == 'scvi':
        # scVI integration
        logger.info('Creating baseline: scVI integration')
        return _run_scvi(
            adata, 
            batch_key=batch_key,
            random_seed=random_seed,
        )
    
    elif method == 'scanvi':
        # scANVI integration (semi-supervised)
        logger.info('Creating baseline: scANVI integration')
        return _run_scanvi(
            adata, 
            batch_key=batch_key,
            label_key=label_key,
            random_seed=random_seed
            )
    
    elif method == 'bbknn':
        # BBKNN integration
        logger.info('Creating baseline: BBKNN integration')
        return _run_bbknn(
            adata, 
            batch_key=batch_key,
            n_pcs=n_comps,
            random_seed=random_seed
        )
    
    elif method == 'harmony':
        # Harmony integration (Seurat-like)
        logger.info('Creating baseline: Harmony integration')
        return _run_harmony(
            adata, 
            batch_key=batch_key,
            n_pcs=n_comps,
            random_seed=random_seed
        )
    
    elif method == 'scanorama':
        # Scanorama integration
        logger.info('Creating baseline: Scanorama integration')
        return _run_scanorama(
            adata, 
            batch_key=batch_key,
            random_seed=random_seed
        )
    
    elif method == 'pca_qc':
        # PCA with full QC preprocessing (standard scanpy workflow BEFORE batch correction)
        logger.info('Creating baseline: PCA with QC preprocessing (HVG + scale + PCA)')
        return _run_pca_qc(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            n_hvgs=2000,  # Standard HVG count
            n_pcs=n_comps,
            random_seed=random_seed
        )
    
    else:
        raise ValueError(f"Unknown baseline method: {method}. "
                         f"Options: {BASELINE_METHODS + INTEGRATION_METHODS}")


# List of available simple baseline methods (fast, always run)
# (pca_qc is in INTEGRATION_METHODS and is the canonical pre-integration baseline)
BASELINE_METHODS = [
    'embed_cell_types',
    'embed_cell_types_jittered',
    'shuffle_integration',
    'shuffle_integration_by_batch',
    'shuffle_integration_by_cell_type'
]

# List of integration baseline methods (slower, standard benchmarks)
INTEGRATION_METHODS = [
    'pca_qc',      # Standard scanpy workflow: HVG + scale + PCA (pre-batch-correction baseline)
    'scvi',
    'scanvi',
    'bbknn',
    'harmony',
    'scanorama'
]

# List of training-heavy integration methods that require model training
# These methods can be skipped via skip_training_heavy_baselines parameter
TRAINING_HEAVY_METHODS = [
    'scvi',      # Requires training a VAE model
    'scanvi',    # Requires training scVI first, then scANVI
]

# All available methods
ALL_BASELINE_METHODS = BASELINE_METHODS + INTEGRATION_METHODS
