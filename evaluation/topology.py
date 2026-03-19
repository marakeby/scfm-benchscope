"""
Topological Data Analysis (TDA) evaluation module.

Evaluates how well embeddings preserve topological structure in synthetic datasets
with known topologies (trajectories, loops, trees).

Implements metrics from CONCORD's benchmarking framework:
- Mantel correlation: Compares true vs. embedding distance matrices
- Geodesic distances: Shortest path distances on k-NN graphs
- Generation-based correlations: For lineage/tree structures
- Trustworthiness: Local neighborhood preservation (sklearn)
- Betti curve stability: Persistent homology stability (gtda)
- Betti number accuracy: Observed vs expected Betti numbers
- Cell distance correlation: Pairwise distance correlations
- State dispersion correlation: Variance within states
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Literal, Sequence
from anndata import AnnData
import numpy as np
import scanpy as sc
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist, pdist
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr, pearsonr

try:
    from utils.logs_ import get_logger
    logger = get_logger()
except Exception:
    import logging
    logger = logging.getLogger(__name__)

try:
    from skbio.stats.distance import DistanceMatrix, mantel
    HAS_SKBIO = True
except ImportError:
    HAS_SKBIO = False
    logger.warning("scikit-bio not available. Mantel correlation will be unavailable.")

# Optional: gtda for persistent homology
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import BettiCurve
    HAS_GTDA = True
except ImportError:
    HAS_GTDA = False
    logger.warning("giotto-tda not available. Betti curve metrics will be unavailable.")

# Trustworthiness from sklearn
try:
    from sklearn.manifold import trustworthiness as sklearn_trustworthiness
    HAS_SKLEARN_TRUST = True
except ImportError:
    HAS_SKLEARN_TRUST = False
    logger.warning("sklearn.manifold.trustworthiness not available.")


def _clean_distance_matrix(D: np.ndarray, nan_action: str = "drop", fill_value: Optional[float] = None) -> np.ndarray:
    """
    Clean distance matrix for Mantel test compatibility.
    
    Args:
        D: Distance matrix
        nan_action: How to handle NaNs ("drop" or "fill")
        fill_value: Value to fill NaNs with (if nan_action="fill")
        
    Returns:
        Cleaned distance matrix
    """
    D = np.asarray(D, dtype=float)
    
    # Enforce symmetry
    D = 0.5 * (D + D.T)
    
    # Set diagonal to zero
    np.fill_diagonal(D, 0.0)
    
    # Handle NaNs
    if np.isnan(D).any():
        if nan_action == "fill":
            if fill_value is None:
                fill_value = np.nanmax(D) * 1.1
            D = np.nan_to_num(D, nan=fill_value)
        elif nan_action == "drop":
            keep = ~np.isnan(D).any(axis=1)
            D = D[keep][:, keep]
        else:
            raise ValueError("nan_action must be 'drop' or 'fill'")
    
    return D


def compute_mantel_correlation(
    D1: np.ndarray,
    D2: np.ndarray,
    method: str = "spearman",
    permutations: int = 999,
) -> tuple[float, float, float]:
    """
    Compute Mantel correlation between two distance matrices.
    
    Args:
        D1: First distance matrix (e.g., true topology distances)
        D2: Second distance matrix (e.g., embedding distances)
        method: Correlation method ("spearman" or "pearson")
        permutations: Number of permutations for significance testing
        
    Returns:
        Tuple of (correlation_coefficient, p_value, z_score)
    """
    if not HAS_SKBIO:
        raise ImportError("scikit-bio is required for Mantel correlation. Install with: pip install scikit-bio")
    
    D1 = _clean_distance_matrix(D1, nan_action="drop")
    D2 = _clean_distance_matrix(D2, nan_action="drop")
    
    if D1.shape != D2.shape:
        raise ValueError(f"Distance matrices must have the same shape. Got {D1.shape} and {D2.shape}")
    
    dm1 = DistanceMatrix(D1)
    dm2 = DistanceMatrix(D2)
    
    r, p, z = mantel(
        dm1, dm2,
        method=method,
        permutations=permutations,
        alternative="two-sided",
        strict=False,
    )
    
    return float(r), float(p), float(z)


def compute_geodesic_distances(
    adata: AnnData,
    embedding_key: str,
    n_neighbors: int = 30,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute geodesic (shortest path) distances on k-NN graph.
    
    Args:
        adata: AnnData object with embeddings
        embedding_key: Key in adata.obsm containing embeddings
        n_neighbors: Number of neighbors for k-NN graph
        metric: Distance metric for neighbor computation
        
    Returns:
        Geodesic distance matrix (n_cells × n_cells)
    """
    # Compute neighbors if not already computed
    if "neighbors" not in adata.uns or adata.uns["neighbors"]["params"]["use_rep"] != embedding_key:
        logger.info(f"Computing neighbors for geodesic distances using {embedding_key}")
        sc.pp.neighbors(
            adata,
            use_rep=embedding_key,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=0,
        )
    
    # Get connectivity matrix
    conn = adata.obsp["connectivities"]
    if sparse.issparse(conn):
        conn = conn.toarray()
    
    # Convert to distance-like weights (1 - connectivity, or use actual distances)
    # For geodesic, we want edge weights to be distances
    # Use distances if available, otherwise use 1 - connectivity
    if "distances" in adata.obsp:
        dist_graph = adata.obsp["distances"]
        if sparse.issparse(dist_graph):
            dist_graph = dist_graph.toarray()
        # Set infinite distances to a large value
        dist_graph[np.isinf(dist_graph)] = np.nanmax(dist_graph[~np.isinf(dist_graph)]) * 2
    else:
        # Fallback: use 1 - connectivity as distance proxy
        dist_graph = 1.0 - conn
        dist_graph[conn == 0] = np.inf
    
    # Compute shortest paths using Dijkstra
    geodesic_dist = dijkstra(
        dist_graph,
        directed=False,
        unweighted=False,
    )
    
    # Replace infinite distances with NaN
    geodesic_dist[np.isinf(geodesic_dist)] = np.nan
    
    return geodesic_dist


def compute_embedding_distances(
    adata: AnnData,
    embedding_key: str,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute pairwise distances in embedding space.
    
    Args:
        adata: AnnData object with embeddings
        embedding_key: Key in adata.obsm containing embeddings
        metric: Distance metric ("euclidean" or "cosine")
        
    Returns:
        Distance matrix (n_cells × n_cells)
    """
    embeddings = adata.obsm[embedding_key]
    return cdist(embeddings, embeddings, metric=metric)


# ============================================================================
# CONCORD-style Metrics: Trustworthiness
# ============================================================================

def compute_trustworthiness(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: Sequence[int] = (10, 20, 30, 50, 100),
    metric: str = "euclidean",
) -> Dict[str, float]:
    """
    Compute trustworthiness scores at multiple neighborhood sizes.
    
    Trustworthiness measures how well local neighborhoods are preserved
    in the low-dimensional embedding. From CONCORD benchmark.
    
    Args:
        X_high: High-dimensional ground truth data (n_samples, n_features)
        X_low: Low-dimensional embedding (n_samples, n_components)
        n_neighbors: Neighborhood sizes to evaluate
        metric: Distance metric for neighbor computation
        
    Returns:
        Dictionary with 'mean', 'scores', and individual k values
    """
    if not HAS_SKLEARN_TRUST:
        logger.warning("sklearn.manifold.trustworthiness not available")
        return {"trustworthiness_mean": np.nan}
    
    scores = {}
    for k in n_neighbors:
        if k >= X_high.shape[0]:
            logger.warning(f"n_neighbors={k} >= n_samples={X_high.shape[0]}, skipping")
            continue
        try:
            score = sklearn_trustworthiness(X_high, X_low, n_neighbors=k, metric=metric)
            scores[f"trustworthiness_k{k}"] = score
        except Exception as e:
            logger.warning(f"Failed to compute trustworthiness for k={k}: {e}")
            scores[f"trustworthiness_k{k}"] = np.nan
    
    valid_scores = [v for v in scores.values() if not np.isnan(v)]
    mean_score = np.mean(valid_scores) if valid_scores else np.nan
    
    return {
        "trustworthiness_mean": mean_score,
        **scores,
    }


# ============================================================================
# CONCORD-style Metrics: Cell Distance Correlation
# ============================================================================

def compute_cell_distance_correlation(
    D_true: np.ndarray,
    D_embed: np.ndarray,
    corr_types: Sequence[str] = ("spearmanr", "pearsonr"),
) -> Dict[str, float]:
    """
    Compute correlation between true and embedding pairwise distances.
    
    This is the 'cell distance correlation' metric from CONCORD.
    Uses the upper triangle of distance matrices (condensed form).
    
    Args:
        D_true: True distance matrix (n × n)
        D_embed: Embedding distance matrix (n × n)
        corr_types: Correlation methods to compute
        
    Returns:
        Dictionary of correlation coefficients
    """
    # Get upper triangle (condensed form)
    n = D_true.shape[0]
    idx_upper = np.triu_indices(n, k=1)
    d_true_flat = D_true[idx_upper]
    d_embed_flat = D_embed[idx_upper]
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(d_true_flat) | np.isnan(d_embed_flat))
    d_true_valid = d_true_flat[valid_mask]
    d_embed_valid = d_embed_flat[valid_mask]
    
    if len(d_true_valid) < 10:
        logger.warning("Too few valid distance pairs for correlation")
        return {f"cell_distance_{ct}": np.nan for ct in corr_types}
    
    results = {}
    if "spearmanr" in corr_types:
        r, _ = spearmanr(d_true_valid, d_embed_valid)
        results["cell_distance_spearmanr"] = r
    if "pearsonr" in corr_types:
        r, _ = pearsonr(d_true_valid, d_embed_valid)
        results["cell_distance_pearsonr"] = r
    
    return results


# ============================================================================
# CONCORD-style Metrics: State Dispersion Correlation
# ============================================================================

def compute_state_dispersion(
    adata: AnnData,
    embedding_key: str,
    state_key: str,
    dispersion_metric: str = "var",
) -> Dict[str, float]:
    """
    Compute dispersion (variance) of embedding within each state/cluster.
    
    From CONCORD benchmark: measures how tightly cells of the same
    state are clustered in embedding space.
    
    Args:
        adata: AnnData object with embeddings
        embedding_key: Key in adata.obsm for embedding
        state_key: Key in adata.obs for state/cluster labels
        dispersion_metric: 'var' or 'std'
        
    Returns:
        Dictionary mapping state -> dispersion value
    """
    data = adata.obsm[embedding_key]
    state_labels = adata.obs[state_key].values
    unique_states = np.unique(state_labels)
    
    dispersions = {}
    for state in unique_states:
        mask = state_labels == state
        state_data = data[mask]
        
        if dispersion_metric == "var":
            disp = np.mean(np.var(state_data, axis=0))
        elif dispersion_metric == "std":
            disp = np.mean(np.std(state_data, axis=0))
        else:
            raise ValueError(f"Invalid dispersion_metric: {dispersion_metric}")
        
        dispersions[str(state)] = disp
    
    return dispersions


def compute_state_dispersion_correlation(
    dispersion_embed: Dict[str, float],
    dispersion_truth: Dict[str, float],
    corr_types: Sequence[str] = ("spearmanr", "pearsonr"),
) -> Dict[str, float]:
    """
    Correlate state dispersion between embedding and ground truth.
    
    Args:
        dispersion_embed: Dispersion per state in embedding
        dispersion_truth: Dispersion per state in ground truth
        corr_types: Correlation methods to compute
        
    Returns:
        Dictionary of correlation coefficients
    """
    # Align states
    common_states = sorted(set(dispersion_embed.keys()) & set(dispersion_truth.keys()))
    if len(common_states) < 3:
        logger.warning("Too few common states for dispersion correlation")
        return {f"state_dispersion_{ct}": np.nan for ct in corr_types}
    
    embed_vals = np.array([dispersion_embed[s] for s in common_states])
    truth_vals = np.array([dispersion_truth[s] for s in common_states])
    
    results = {}
    if "spearmanr" in corr_types:
        r, _ = spearmanr(embed_vals, truth_vals)
        results["state_dispersion_spearmanr"] = r
    if "pearsonr" in corr_types:
        r, _ = pearsonr(embed_vals, truth_vals)
        results["state_dispersion_pearsonr"] = r
    
    return results


# ============================================================================
# CONCORD-style Metrics: Betti Curves (Persistent Homology)
# ============================================================================

def compute_persistent_homology(
    X: np.ndarray,
    homology_dimensions: Sequence[int] = (0, 1, 2),
    max_points: Optional[int] = 2000,
    random_state: Optional[int] = 42,
) -> Optional[np.ndarray]:
    """
    Compute persistent homology using Vietoris-Rips complex.
    
    From CONCORD benchmark: uses giotto-tda for persistent homology.
    
    Args:
        X: Point cloud data (n_samples, n_features)
        homology_dimensions: Dimensions to compute (default: 0, 1, 2)
        max_points: Maximum points to use (subsamples if exceeded)
        random_state: Random seed for subsampling
        
    Returns:
        Persistence diagrams or None if gtda not available
    """
    if not HAS_GTDA:
        logger.warning("giotto-tda not available for persistent homology")
        return None
    
    # Subsample if too large
    if max_points is not None and X.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
    
    logger.info(f"Computing persistent homology for {X.shape[0]} points...")
    
    VR = VietorisRipsPersistence(homology_dimensions=list(homology_dimensions))
    diagrams = VR.fit_transform(X[None, :, :])  # Shape: (1, n_features, 3)
    
    return diagrams


def compute_betti_stability(betti_values: np.ndarray) -> float:
    """
    Compute Betti curve stability score.
    
    From CONCORD: stability = 1 / (1 + variance(betti_values))
    Score is in [0, 1], higher is more stable.
    
    Args:
        betti_values: Array of Betti numbers across filtration
        
    Returns:
        Stability score in [0, 1]
    """
    var = np.var(betti_values)
    return 1.0 / (1.0 + var)


def compute_betti_statistics(
    diagrams: np.ndarray,
    expected_betti_numbers: Sequence[int] = (0, 0, 0),
    n_bins: int = 100,
) -> Dict[str, float]:
    """
    Compute Betti curve statistics from persistence diagrams.
    
    From CONCORD benchmark: computes stability, L1 distance, and accuracy.
    
    Args:
        diagrams: Persistence diagrams from compute_persistent_homology
        expected_betti_numbers: Expected Betti numbers for each dimension
        n_bins: Number of bins for Betti curve
        
    Returns:
        Dictionary with Betti statistics
    """
    if not HAS_GTDA:
        return {
            "betti_curve_stability": np.nan,
            "betti_number_L1": np.nan,
            "betti_number_accuracy": np.nan,
        }
    
    # Compute Betti curves
    betti_transformer = BettiCurve(n_bins=n_bins)
    betti_curves = betti_transformer.fit_transform(diagrams)
    samplings = betti_transformer.samplings_
    
    # Process each homology dimension
    observed_betti_numbers = []
    stabilities = []
    
    for dim in sorted(samplings.keys()):
        betti_values = betti_curves[0][dim, :]
        
        # Stability for this dimension
        stability = compute_betti_stability(betti_values)
        stabilities.append(stability)
        
        # Mode as observed Betti number (most frequent value)
        from scipy.stats import mode
        observed_betti = mode(betti_values, keepdims=False)[0]
        observed_betti_numbers.append(int(observed_betti))
    
    # Average stability across dimensions
    avg_stability = np.mean(stabilities)
    
    # L1 distance between observed and expected Betti numbers
    observed = np.array(observed_betti_numbers[:len(expected_betti_numbers)])
    expected = np.array(expected_betti_numbers[:len(observed)])
    l1_distance = np.sum(np.abs(observed - expected))
    
    # Betti number accuracy: 1 / (1 + L1)
    accuracy = 1.0 / (1.0 + l1_distance)
    
    return {
        "betti_curve_stability": avg_stability,
        "betti_number_L1": l1_distance,
        "betti_number_accuracy": accuracy,
        "observed_betti_numbers": str(observed_betti_numbers),
        "expected_betti_numbers": str(list(expected_betti_numbers)),
    }


class TopologyEvaluator:
    """
    Evaluates topological structure preservation in embeddings.
    
    For synthetic datasets with known topologies (trajectory, loop, tree),
    computes metrics that measure how well the embedding preserves the
    true topological structure.
    
    Implements CONCORD benchmark metrics:
    - Mantel correlation (embedding and geodesic vs true topology)
    - Trustworthiness (local neighborhood preservation)
    - Betti curve stability (persistent homology)
    - Betti number accuracy (expected vs observed)
    - Cell distance correlation
    - State dispersion correlation
    """
    
    # Expected Betti numbers for each topology type
    # Betti numbers: (H0=connected components, H1=loops, H2=voids)
    EXPECTED_BETTI = {
        "trajectory": [1, 0, 0],  # 1 connected component, no loops
        "loop": [1, 1, 0],        # 1 connected component, 1 loop
        "tree": [1, 0, 0],        # 1 connected component (tree has no loops)
        "cluster": [0, 0, 0],     # Multiple disconnected components (varies)
    }
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        topology_type: Literal["trajectory", "loop", "tree", "cluster"],
        label_key: str = "time",  # For trajectory/loop: "time", for tree: "branch"
        save_dir: Optional[str] = None,
        auto_subsample: bool = True,
        metric: str = "euclidean",
        n_neighbors: int = 30,
        groundtruth_key: Optional[str] = None,  # Ground truth embedding for trustworthiness
        expected_betti_numbers: Optional[Sequence[int]] = None,  # Override expected Betti
        compute_betti: bool = True,  # Whether to compute persistent homology
        trustworthiness_neighbors: Sequence[int] = (10, 30, 50, 100),
        **kwargs,
    ):
        """
        Initialize topology evaluator.
        
        Args:
            adata: AnnData object with embeddings and topology metadata
            embedding_key: Key in adata.obsm containing embeddings
            topology_type: Type of topology ("trajectory", "loop", "tree", "cluster")
            label_key: Key in adata.obs for topology labels (e.g., "time", "branch")
            save_dir: Directory to save results
            auto_subsample: Whether to subsample large datasets (>10k cells)
            metric: Distance metric for neighbor computation
            n_neighbors: Number of neighbors for geodesic computation
            groundtruth_key: Key in adata.obsm for ground truth embedding (for trustworthiness)
            expected_betti_numbers: Expected Betti numbers (auto-detected from topology_type if None)
            compute_betti: Whether to compute persistent homology metrics
            trustworthiness_neighbors: Neighborhood sizes for trustworthiness
            **kwargs: Additional parameters
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.topology_type = topology_type
        self.label_key = label_key
        self.save_dir = Path(save_dir) if save_dir else None
        self.auto_subsample = auto_subsample
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.groundtruth_key = groundtruth_key
        self.compute_betti = compute_betti
        self.trustworthiness_neighbors = trustworthiness_neighbors
        
        # Set expected Betti numbers
        if expected_betti_numbers is not None:
            self.expected_betti_numbers = list(expected_betti_numbers)
        else:
            self.expected_betti_numbers = self.EXPECTED_BETTI.get(topology_type, [0, 0, 0])
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_true_topology_distances(self) -> np.ndarray:
        """
        Compute true topology distance matrix based on topology type.
        
        Returns:
            True topology distance matrix
        """
        if self.topology_type in ["trajectory", "loop"]:
            # For trajectories/loops, use pseudotime distance
            time_values = self.adata.obs[self.label_key].values
            # Handle circular topology for loops
            if self.topology_type == "loop":
                # For loops, compute circular distance
                time_max = time_values.max()
                time_min = time_values.min()
                time_range = time_max - time_min
                # Circular distance: min(|t1 - t2|, range - |t1 - t2|)
                time_diff = np.abs(time_values[:, None] - time_values[None, :])
                D = np.minimum(time_diff, time_range - time_diff)
            else:
                # Linear trajectory: absolute time difference
                D = np.abs(time_values[:, None] - time_values[None, :])
        
        elif self.topology_type == "tree":
            # For trees, use branch labels to compute hierarchical distance
            # Simple approach: distance = 0 if same branch, 1 if different
            branch_labels = self.adata.obs[self.label_key].values
            # More sophisticated: could use depth information if available
            D = (branch_labels[:, None] != branch_labels[None, :]).astype(float)
            # Could be enhanced with actual tree structure if available in adata.uns
        
        elif self.topology_type == "cluster":
            # For clusters, distance = 0 if same cluster, 1 if different
            cluster_labels = self.adata.obs[self.label_key].values
            D = (cluster_labels[:, None] != cluster_labels[None, :]).astype(float)
        
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        return D
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate topological structure preservation using CONCORD benchmark metrics.
        
        Computes:
        1. Mantel correlation (embedding and geodesic vs true topology)
        2. Cell distance correlation (Spearman/Pearson)
        3. Trustworthiness (local neighborhood preservation)
        4. Betti curve stability (persistent homology)
        5. Betti number accuracy (expected vs observed)
        6. State dispersion correlation (if labels available)
        
        Returns:
            Dictionary of topology metrics
        """
        logger.info(f"Evaluating topology preservation for embedding: {self.embedding_key}, topology: {self.topology_type}")
        logger.info(f"Expected Betti numbers: {self.expected_betti_numbers}")
        
        # Subsample if needed
        if self.auto_subsample and self.adata.shape[0] > 10000:
            logger.info("Subsampling dataset for topology evaluation (>10k cells)")
            try:
                from utils.sampling import sample_adata
                adata_eval = sample_adata(self.adata, sample_size=5000, stratify_by=self.label_key)
            except ImportError:
                logger.warning("utils.sampling not available, using random subsampling")
                idx = np.random.choice(self.adata.shape[0], size=5000, replace=False)
                adata_eval = self.adata[idx].copy()
        else:
            adata_eval = self.adata.copy()
        
        results_dict = {}
        
        # ================================================================
        # 1. Compute distance matrices
        # ================================================================
        
        # 1a. True topology distances
        logger.info("Computing true topology distances...")
        try:
            D_true = self._compute_true_topology_distances()
        except Exception as e:
            logger.warning(f"Failed to compute true topology distances: {e}")
            D_true = None
        
        # 1b. Embedding distances
        logger.info("Computing embedding distances...")
        try:
            D_embed = compute_embedding_distances(
                adata_eval,
                embedding_key=self.embedding_key,
                metric=self.metric,
            )
        except Exception as e:
            logger.warning(f"Failed to compute embedding distances: {e}")
            D_embed = None
        
        # 1c. Geodesic distances
        logger.info("Computing geodesic distances...")
        try:
            D_geodesic = compute_geodesic_distances(
                adata_eval,
                embedding_key=self.embedding_key,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
            )
        except Exception as e:
            logger.warning(f"Failed to compute geodesic distances: {e}")
            D_geodesic = None
        
        # ================================================================
        # 2. Mantel correlation (CONCORD: cell distance correlation)
        # ================================================================
        
        # 2a. Mantel: true topology vs embedding distances
        if D_true is not None and D_embed is not None and HAS_SKBIO:
            logger.info("Computing Mantel correlation (true topology vs embedding)...")
            try:
                r, p, z = compute_mantel_correlation(D_true, D_embed, method="spearman")
                results_dict["mantel_correlation_embedding"] = r
                results_dict["mantel_pvalue_embedding"] = p
            except Exception as e:
                logger.warning(f"Mantel correlation (embedding) failed: {e}")
        
        # 2b. Mantel: true topology vs geodesic distances
        if D_true is not None and D_geodesic is not None and HAS_SKBIO:
            logger.info("Computing Mantel correlation (true topology vs geodesic)...")
            try:
                r, p, z = compute_mantel_correlation(D_true, D_geodesic, method="spearman")
                results_dict["mantel_correlation_geodesic"] = r
                results_dict["mantel_pvalue_geodesic"] = p
            except Exception as e:
                logger.warning(f"Mantel correlation (geodesic) failed: {e}")
        
        # ================================================================
        # 3. Cell distance correlation (CONCORD-style Pearson/Spearman)
        # ================================================================
        if D_true is not None and D_embed is not None:
            logger.info("Computing cell distance correlation...")
            try:
                corr_results = compute_cell_distance_correlation(D_true, D_embed)
                results_dict.update(corr_results)
            except Exception as e:
                logger.warning(f"Cell distance correlation failed: {e}")
        
        # ================================================================
        # 4. Trustworthiness (CONCORD geometry benchmark)
        # ================================================================
        if self.groundtruth_key is not None and self.groundtruth_key in adata_eval.obsm:
            logger.info(f"Computing trustworthiness vs ground truth: {self.groundtruth_key}")
            try:
                X_high = adata_eval.obsm[self.groundtruth_key]
                X_low = adata_eval.obsm[self.embedding_key]
                trust_results = compute_trustworthiness(
                    X_high, X_low,
                    n_neighbors=self.trustworthiness_neighbors,
                    metric=self.metric,
                )
                results_dict.update(trust_results)
            except Exception as e:
                logger.warning(f"Trustworthiness computation failed: {e}")
        else:
            # If no ground truth, compute trustworthiness against the original data (X)
            if hasattr(adata_eval, 'X') and adata_eval.X is not None:
                logger.info("Computing trustworthiness vs original expression data")
                try:
                    X_high = adata_eval.X
                    if sparse.issparse(X_high):
                        X_high = X_high.toarray()
                    X_low = adata_eval.obsm[self.embedding_key]
                    trust_results = compute_trustworthiness(
                        X_high, X_low,
                        n_neighbors=self.trustworthiness_neighbors,
                        metric=self.metric,
                    )
                    results_dict.update(trust_results)
                except Exception as e:
                    logger.warning(f"Trustworthiness computation failed: {e}")
        
        # ================================================================
        # 5. Betti curve metrics (CONCORD topology benchmark)
        # ================================================================
        if self.compute_betti and HAS_GTDA:
            logger.info("Computing persistent homology metrics...")
            try:
                embeddings = adata_eval.obsm[self.embedding_key]
                diagrams = compute_persistent_homology(
                    embeddings,
                    homology_dimensions=(0, 1, 2),
                    max_points=2000,
                    random_state=42,
                )
                if diagrams is not None:
                    betti_results = compute_betti_statistics(
                        diagrams,
                        expected_betti_numbers=self.expected_betti_numbers,
                    )
                    results_dict.update(betti_results)
            except Exception as e:
                logger.warning(f"Betti curve computation failed: {e}")
        elif self.compute_betti and not HAS_GTDA:
            logger.warning("giotto-tda not available, skipping Betti curve metrics")
        
        # ================================================================
        # 6. State dispersion correlation (CONCORD geometry benchmark)
        # ================================================================
        if self.label_key in adata_eval.obs.columns:
            logger.info("Computing state dispersion...")
            try:
                disp_embed = compute_state_dispersion(
                    adata_eval,
                    embedding_key=self.embedding_key,
                    state_key=self.label_key,
                    dispersion_metric="var",
                )
                # Store mean dispersion
                results_dict["state_dispersion_mean"] = np.mean(list(disp_embed.values()))
                
                # If ground truth embedding available, compute correlation
                if self.groundtruth_key is not None and self.groundtruth_key in adata_eval.obsm:
                    disp_truth = compute_state_dispersion(
                        adata_eval,
                        embedding_key=self.groundtruth_key,
                        state_key=self.label_key,
                        dispersion_metric="var",
                    )
                    disp_corr = compute_state_dispersion_correlation(disp_embed, disp_truth)
                    results_dict.update(disp_corr)
            except Exception as e:
                logger.warning(f"State dispersion computation failed: {e}")
        
        # ================================================================
        # Save results
        # ================================================================
        
        # Remove NaN values (but keep string values like observed_betti_numbers)
        results_dict = {
            k: v for k, v in results_dict.items() 
            if not (isinstance(v, float) and np.isnan(v))
        }
        
        # Save to adata.uns
        if "topology" not in self.adata.uns:
            self.adata.uns["topology"] = {}
        self.adata.uns["topology"][self.embedding_key] = results_dict
        logger.info(f"Saved topology results to adata.uns['topology']['{self.embedding_key}']")
        
        # Save to CSV file
        if self.save_dir:
            metrics_file = self.save_dir / "topology_metrics.csv"
            
            # Separate numeric and non-numeric columns for CSV
            numeric_results = {
                k: v for k, v in results_dict.items() 
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            new_row = pd.DataFrame.from_dict({self.embedding_key: numeric_results}, orient="index")
            
            if metrics_file.exists():
                try:
                    existing_df = pd.read_csv(metrics_file, index_col=0)
                    if self.embedding_key in existing_df.index:
                        existing_df = existing_df.drop(self.embedding_key)
                    combined_df = pd.concat([existing_df, new_row])
                    combined_df.to_csv(metrics_file)
                except Exception as e:
                    logger.warning(f"Error appending to existing file, overwriting: {e}")
                    new_row.to_csv(metrics_file)
            else:
                new_row.to_csv(metrics_file)
            logger.info(f"Saved topology results to '{metrics_file}'")
        
        logger.info(f"Topology evaluation complete. Metrics: {list(results_dict.keys())}")
        return results_dict
