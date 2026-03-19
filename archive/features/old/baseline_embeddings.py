"""
Baseline embedding generators for batch integration evaluation.
Adapted from task_batch_integration control methods.
"""
import numpy as np
import scanpy as sc
from typing import Optional
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.logs_ import get_logger

logger = get_logger()


def _randomize_features(X, partition=None, random_state=0):
    """
    Randomize features within partitions.
    
    Args:
        X: Feature matrix (cells x features)
        partition: Partition labels (e.g., batch labels). If None, randomize globally.
        random_state: Random seed
        
    Returns:
        Randomized feature matrix
    """
    rng = np.random.RandomState(random_state)
    X_out = X.copy()
    if partition is None:
        partition = np.full(X.shape[0], 0)
    else:
        partition = np.asarray(partition)
    for partition_name in np.unique(partition):
        partition_idx = np.argwhere(partition == partition_name).flatten()
        X_out[partition_idx] = X[rng.permutation(partition_idx)]
    return X_out


def _perfect_embedding(partition, jitter=0.01, random_state=0):
    """
    Create perfect embedding based on partition labels (one-hot encoded).
    
    Args:
        partition: Partition labels (e.g., cell_type)
        jitter: Amount of random jitter to add
        random_state: Random seed
        
    Returns:
        Embedding matrix
    """
    rng = np.random.RandomState(random_state)
    embedding = OneHotEncoder(sparse_output=False).fit_transform(
        LabelEncoder().fit_transform(partition)[:, None]
    )
    if jitter is not None and jitter > 0:
        embedding = embedding + rng.uniform(-jitter, jitter, embedding.shape)
    return embedding


def generate_baseline_embeddings(
    adata: AnnData,
    baseline_type: str = "no_integration",
    batch_key: str = "batch",
    label_key: str = "cell_type",
    embedding_key: Optional[str] = None,
    random_state: int = 0
) -> np.ndarray:
    """
    Generate baseline embeddings for batch integration evaluation.
    
    Args:
        adata: AnnData object with expression data
        baseline_type: Type of baseline to generate:
            - "no_integration": Use existing PCA (if available) or compute global PCA
            - "no_integration_batch": Compute PCA separately per batch
            - "perfect_cell_type": Perfect embedding based on cell type labels
            - "perfect_cell_type_jittered": Perfect embedding with jitter
            - "shuffle": Randomize all features globally
            - "shuffle_by_batch": Randomize features within each batch
            - "shuffle_by_cell_type": Randomize features within each cell type
        batch_key: Key in obs for batch labels
        label_key: Key in obs for cell type labels
        embedding_key: Key to use for existing embeddings (e.g., "X_pca")
        random_state: Random seed
        
    Returns:
        Embedding matrix (n_cells x n_dims)
    """
    rng = np.random.RandomState(random_state)
    
    if baseline_type == "no_integration":
        # Use existing PCA if available, otherwise compute global PCA
        if embedding_key and embedding_key in adata.obsm:
            logger.info(f"Using existing {embedding_key} for no_integration baseline")
            return adata.obsm[embedding_key]
        else:
            logger.info("Computing global PCA for no_integration baseline")
            adata_temp = adata.copy()
            if "X_pca" not in adata_temp.obsm:
                sc.pp.pca(adata_temp, n_comps=50, random_state=random_state)
            return adata_temp.obsm["X_pca"]
    
    elif baseline_type == "no_integration_batch":
        # Compute PCA separately per batch
        logger.info("Computing PCA per batch for no_integration_batch baseline")
        n_comps = 50
        embeddings = np.zeros((adata.shape[0], n_comps), dtype=float)
        
        for batch in adata.obs[batch_key].unique():
            batch_idx = adata.obs[batch_key] == batch
            n_batch_cells = np.sum(batch_idx)
            
            if n_batch_cells <= n_comps:
                n_comps_batch = max(1, n_batch_cells - 1)
                solver = "full"
            else:
                n_comps_batch = n_comps
                solver = "arpack"
            
            adata_batch = adata[batch_idx].copy()
            sc.pp.pca(adata_batch, n_comps=n_comps_batch, svd_solver=solver, random_state=random_state)
            embeddings[batch_idx, :n_comps_batch] = adata_batch.obsm["X_pca"]
        
        return embeddings
    
    elif baseline_type == "perfect_cell_type":
        logger.info("Generating perfect cell type embedding")
        return _perfect_embedding(adata.obs[label_key], jitter=0.0, random_state=random_state)
    
    elif baseline_type == "perfect_cell_type_jittered":
        logger.info("Generating perfect cell type embedding with jitter")
        return _perfect_embedding(adata.obs[label_key], jitter=0.01, random_state=random_state)
    
    elif baseline_type == "shuffle":
        logger.info("Generating shuffled embedding (randomize all features)")
        # Randomize normalized expression, then compute PCA
        adata_temp = adata.copy()
        if "normalized" in adata_temp.layers:
            adata_temp.X = _randomize_features(adata_temp.layers["normalized"], partition=None, random_state=random_state)
        else:
            adata_temp.X = _randomize_features(adata_temp.X, partition=None, random_state=random_state)
        sc.pp.pca(adata_temp, n_comps=50, random_state=random_state)
        return adata_temp.obsm["X_pca"]
    
    elif baseline_type == "shuffle_by_batch":
        logger.info("Generating shuffled-by-batch embedding")
        # Randomize features within each batch, then compute PCA
        adata_temp = adata.copy()
        if "normalized" in adata_temp.layers:
            adata_temp.X = _randomize_features(
                adata_temp.layers["normalized"], 
                partition=adata_temp.obs[batch_key], 
                random_state=random_state
            )
        else:
            adata_temp.X = _randomize_features(
                adata_temp.X, 
                partition=adata_temp.obs[batch_key], 
                random_state=random_state
            )
        sc.pp.pca(adata_temp, n_comps=50, random_state=random_state)
        return adata_temp.obsm["X_pca"]
    
    elif baseline_type == "shuffle_by_cell_type":
        logger.info("Generating shuffled-by-cell-type embedding")
        # Randomize features within each cell type, then compute PCA
        adata_temp = adata.copy()
        if "normalized" in adata_temp.layers:
            adata_temp.X = _randomize_features(
                adata_temp.layers["normalized"], 
                partition=adata_temp.obs[label_key], 
                random_state=random_state
            )
        else:
            adata_temp.X = _randomize_features(
                adata_temp.X, 
                partition=adata_temp.obs[label_key], 
                random_state=random_state
            )
        sc.pp.pca(adata_temp, n_comps=50, random_state=random_state)
        return adata_temp.obsm["X_pca"]
    
    else:
        raise ValueError(
            f"Unknown baseline_type: {baseline_type}. "
            f"Choose from: no_integration, no_integration_batch, perfect_cell_type, "
            f"perfect_cell_type_jittered, shuffle, shuffle_by_batch, shuffle_by_cell_type"
        )


def add_baseline_embeddings(
    adata: AnnData,
    baseline_types: list = None,
    batch_key: str = "batch",
    label_key: str = "cell_type",
    embedding_key: Optional[str] = "X_pca",
    random_state: int = 0
) -> AnnData:
    """
    Add multiple baseline embeddings to adata.obsm.
    
    Args:
        adata: AnnData object
        baseline_types: List of baseline types to generate. If None, generates all.
        batch_key: Key in obs for batch labels
        label_key: Key in obs for cell type labels
        embedding_key: Key to use for existing embeddings
        random_state: Random seed
        
    Returns:
        AnnData with baseline embeddings added to obsm
    """
    if baseline_types is None:
        baseline_types = [
            "no_integration",
            "no_integration_batch",
            "perfect_cell_type",
            "shuffle",
            "shuffle_by_batch"
        ]
    
    for baseline_type in baseline_types:
        try:
            embedding = generate_baseline_embeddings(
                adata,
                baseline_type=baseline_type,
                batch_key=batch_key,
                label_key=label_key,
                embedding_key=embedding_key,
                random_state=random_state
            )
            obsm_key = f"X_baseline_{baseline_type}"
            adata.obsm[obsm_key] = embedding
            logger.info(f"Added baseline embedding: {obsm_key} (shape: {embedding.shape})")
        except Exception as e:
            logger.warning(f"Failed to generate {baseline_type} baseline: {e}")
    
    return adata