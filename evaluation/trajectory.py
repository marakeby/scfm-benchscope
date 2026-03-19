"""
Trajectory evaluation module.

Implements CONCORD's trajectory/pseudotime benchmark:
- Compute pseudotime from embeddings via shortest path on k-NN graph
- Correlate with ground truth time
- Additional metrics: path curvature, geodesic distance ratio

This is distinct from topology evaluation (Betti numbers) - trajectory evaluation
focuses on how well embeddings preserve developmental/temporal ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr, spearmanr, kendalltau

# Try to import logger, fall back to standard logging
try:
    from utils.logs_ import get_logger
    logger = get_logger()
except Exception:
    import logging
    logger = logging.getLogger(__name__)


def shortest_path_on_knn_graph(
    embedding: np.ndarray,
    point_a: int,
    point_b: Optional[int] = None,
    k: int = 30,
    metric: str = 'euclidean'
) -> Tuple[List[int], np.ndarray]:
    """
    Compute shortest path between two points on a k-NN graph.
    
    Uses Dijkstra's algorithm on the k-NN graph built from the embedding.
    This replicates CONCORD's path_analysis.shortest_path_on_knn_graph.
    
    Args:
        embedding: Cell embeddings (n_cells, n_dims)
        point_a: Starting cell index
        point_b: Ending cell index. If None, uses furthest point from point_a
        k: Number of nearest neighbors
        metric: Distance metric ('euclidean' or 'cosine')
    
    Returns:
        path: List of cell indices forming the shortest path
        dist_matrix: Distance matrix from point_a to all other points
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_cells = embedding.shape[0]
    
    # Build k-NN graph
    if metric == 'cosine':
        # Normalize for cosine distance
        embedding_norm = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10)
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(embedding_norm)
        distances, indices = nn.kneighbors(embedding_norm)
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
        nn.fit(embedding)
        distances, indices = nn.kneighbors(embedding)
    
    # Remove self-connections
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Build sparse adjacency matrix with distances as weights
    rows = np.repeat(np.arange(n_cells), k)
    cols = indices.flatten()
    data = distances.flatten()
    graph = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    
    # Make symmetric (undirected graph)
    graph = graph + graph.T
    graph.data = np.minimum.reduceat(
        graph.data, 
        np.arange(0, len(graph.data), 2) if len(graph.data) > 1 else [0]
    ) if len(graph.data) > 0 else graph.data
    
    # Compute shortest path using Dijkstra
    dist_matrix, predecessors = dijkstra(
        csgraph=graph, 
        directed=False, 
        indices=[point_a], 
        return_predecessors=True
    )
    
    # Determine point_b if not provided (furthest reachable point)
    if point_b is None:
        # Exclude unreachable points (infinite distance)
        reachable = dist_matrix[0] < np.inf
        if not reachable.any():
            logger.warning("No reachable points from start point")
            return [], dist_matrix
        point_b = np.argmax(np.where(reachable, dist_matrix[0], -np.inf))
        logger.info(f"Auto-selected furthest point: {point_b}")
    
    # Trace path from point_b back to point_a
    if dist_matrix[0, point_b] == np.inf:
        logger.warning(f"Point {point_b} is not reachable from {point_a}")
        return [], dist_matrix
    
    path = []
    i = point_b
    while i != point_a and i != -9999:  # -9999 is scipy's "no predecessor" value
        path.append(i)
        i = predecessors[0, i]
    
    if i == point_a:
        path.append(point_a)
        path = path[::-1]
    else:
        logger.warning("Could not trace path back to start point")
        return [], dist_matrix
    
    return path, dist_matrix


def compute_pseudotime_from_path(
    embedding: np.ndarray,
    path: List[int]
) -> np.ndarray:
    """
    Compute pseudotime for all cells by projecting onto the shortest path.
    
    Each cell's pseudotime is determined by finding the closest point on 
    the path (piecewise linear interpolation).
    
    Args:
        embedding: Cell embeddings (n_cells, n_dims)
        path: List of cell indices forming the path
    
    Returns:
        pseudotimes: Normalized pseudotime values [0, 1] for all cells
    """
    if len(path) < 2:
        logger.warning("Path too short for pseudotime computation")
        return np.full(embedding.shape[0], np.nan)
    
    # Extract path coordinates
    path_coords = embedding[path]
    
    # Compute cumulative distance along path
    path_distances = [0.0]
    for i in range(len(path_coords) - 1):
        dist = np.linalg.norm(path_coords[i + 1] - path_coords[i])
        path_distances.append(path_distances[-1] + dist)
    path_distances = np.array(path_distances)
    
    # Project each cell onto the path
    pseudotimes = np.zeros(embedding.shape[0])
    
    for cell_idx, point in enumerate(embedding):
        min_dist = np.inf
        best_t = 0.0
        
        for i in range(len(path_coords) - 1):
            seg_start = path_coords[i]
            seg_end = path_coords[i + 1]
            
            # Project point onto segment
            seg_vector = seg_end - seg_start
            seg_length_sq = np.dot(seg_vector, seg_vector)
            
            if seg_length_sq < 1e-10:
                # Degenerate segment
                proj = seg_start
                t = 0.0
            else:
                t = np.dot(point - seg_start, seg_vector) / seg_length_sq
                t = np.clip(t, 0.0, 1.0)
                proj = seg_start + t * seg_vector
            
            dist = np.linalg.norm(point - proj)
            
            if dist < min_dist:
                min_dist = dist
                # Interpolate pseudotime based on segment position
                best_t = path_distances[i] + t * (path_distances[i + 1] - path_distances[i])
        
        pseudotimes[cell_idx] = best_t
    
    # Normalize to [0, 1]
    if pseudotimes.max() > pseudotimes.min():
        pseudotimes = (pseudotimes - pseudotimes.min()) / (pseudotimes.max() - pseudotimes.min())
    
    return pseudotimes


def compute_path_curvature(
    embedding: np.ndarray,
    path: List[int]
) -> float:
    """
    Compute curvature of a path (geodesic distance / euclidean distance).
    
    Curvature > 1 means the path is longer than the straight line.
    Higher curvature indicates more winding/complex trajectory.
    
    Args:
        embedding: Cell embeddings
        path: List of cell indices
    
    Returns:
        curvature: Ratio of geodesic to euclidean distance
    """
    if len(path) < 2:
        return np.nan
    
    path_coords = embedding[path]
    
    # Geodesic distance (sum of segment lengths)
    geodesic = sum(
        np.linalg.norm(path_coords[i + 1] - path_coords[i])
        for i in range(len(path_coords) - 1)
    )
    
    # Euclidean distance (start to end)
    euclidean = np.linalg.norm(path_coords[-1] - path_coords[0])
    
    if euclidean < 1e-10:
        return np.inf
    
    return geodesic / euclidean


class TrajectoryEvaluator:
    """
    Evaluates trajectory/pseudotime preservation in embeddings.
    
    Implements CONCORD's trajectory benchmark methodology:
    1. Compute shortest path on k-NN graph from start to end point
    2. Project all cells onto path to get pseudotime
    3. Correlate with ground truth time
    
    Additional metrics:
    - Path curvature (straightness of trajectory)
    - Multiple correlation types (Pearson, Spearman, Kendall)
    """
    
    def __init__(
        self,
        adata: ad.AnnData,
        embedding_key: str,
        time_key: str,
        save_dir: Union[str, Path],
        start_point: Optional[int] = None,
        end_point: Optional[int] = None,
        k: int = 30,
        metric: str = 'euclidean',
        auto_detect_endpoints: bool = True,
        correlation_types: List[str] = ['pearsonr', 'spearmanr', 'kendalltau'],
        **kwargs
    ):
        """
        Initialize trajectory evaluator.
        
        Args:
            adata: AnnData object with embeddings
            embedding_key: Key in adata.obsm for embeddings
            time_key: Key in adata.obs for ground truth time
            save_dir: Directory to save results
            start_point: Cell index for trajectory start (or auto-detect)
            end_point: Cell index for trajectory end (or auto-detect)
            k: Number of neighbors for k-NN graph
            metric: Distance metric ('euclidean' or 'cosine')
            auto_detect_endpoints: If True, auto-detect start/end from time_key
            correlation_types: Which correlations to compute
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.time_key = time_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.metric = metric
        self.correlation_types = correlation_types
        
        # Validate inputs
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm")
        if time_key not in adata.obs:
            raise ValueError(f"Time key '{time_key}' not found in adata.obs")
        
        self.embedding = np.array(adata.obsm[embedding_key])
        self.true_time = np.array(adata.obs[time_key])
        
        # Handle start/end points
        if auto_detect_endpoints and start_point is None:
            # Start at cell with minimum time
            start_point = int(np.argmin(self.true_time))
            logger.info(f"Auto-detected start point: {start_point} (time={self.true_time[start_point]:.3f})")
        
        if auto_detect_endpoints and end_point is None:
            # End at cell with maximum time
            end_point = int(np.argmax(self.true_time))
            logger.info(f"Auto-detected end point: {end_point} (time={self.true_time[end_point]:.3f})")
        
        self.start_point = start_point
        self.end_point = end_point
    
    def evaluate(self) -> Dict:
        """
        Run trajectory evaluation.
        
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        logger.info(f"Computing trajectory metrics for embedding: {self.embedding_key}")
        logger.info(f"Start point: {self.start_point}, End point: {self.end_point}")
        
        # 1. Compute shortest path
        logger.info("Computing shortest path on k-NN graph...")
        path, dist_matrix = shortest_path_on_knn_graph(
            embedding=self.embedding,
            point_a=self.start_point,
            point_b=self.end_point,
            k=self.k,
            metric=self.metric
        )
        
        if len(path) < 2:
            logger.error("Failed to compute valid path")
            results['error'] = "Failed to compute path"
            return results
        
        results['path_length'] = len(path)
        logger.info(f"Path length: {len(path)} cells")
        
        # 2. Compute pseudotime
        logger.info("Computing pseudotime from shortest path...")
        pseudotime = compute_pseudotime_from_path(self.embedding, path)
        
        # Store pseudotime in adata
        pseudotime_key = f"{self.embedding_key}_pseudotime"
        self.adata.obs[pseudotime_key] = pseudotime
        results['pseudotime_key'] = pseudotime_key
        
        # 3. Compute correlations with ground truth
        logger.info("Computing correlations with ground truth time...")
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(pseudotime) | np.isnan(self.true_time))
        pseudo_valid = pseudotime[valid_mask]
        true_valid = self.true_time[valid_mask]
        
        if 'pearsonr' in self.correlation_types:
            corr, pval = pearsonr(pseudo_valid, true_valid)
            results['pearsonr'] = corr
            results['pearsonr_pval'] = pval
            logger.info(f"Pearson correlation: {corr:.4f} (p={pval:.2e})")
        
        if 'spearmanr' in self.correlation_types:
            corr, pval = spearmanr(pseudo_valid, true_valid)
            results['spearmanr'] = corr
            results['spearmanr_pval'] = pval
            logger.info(f"Spearman correlation: {corr:.4f} (p={pval:.2e})")
        
        if 'kendalltau' in self.correlation_types:
            corr, pval = kendalltau(pseudo_valid, true_valid)
            results['kendalltau'] = corr
            results['kendalltau_pval'] = pval
            logger.info(f"Kendall tau: {corr:.4f} (p={pval:.2e})")
        
        # 4. Compute path curvature
        curvature = compute_path_curvature(self.embedding, path)
        results['path_curvature'] = curvature
        logger.info(f"Path curvature: {curvature:.4f}")
        
        # 5. Compute additional metrics
        # Time coverage (how much of the time range is covered)
        time_range = self.true_time.max() - self.true_time.min()
        path_time_range = self.true_time[path].max() - self.true_time[path].min()
        results['time_coverage'] = path_time_range / time_range if time_range > 0 else 0
        
        # Monotonicity (how monotonic is the path w.r.t. time)
        path_times = self.true_time[path]
        monotonic_increasing = np.sum(np.diff(path_times) > 0) / max(len(path) - 1, 1)
        results['path_monotonicity'] = monotonic_increasing
        
        # 6. Store in adata.uns
        uns_key = f"trajectory_metrics_{self.embedding_key}"
        self.adata.uns[uns_key] = results
        
        # 7. Save results
        self._save_results(results, path)
        
        return results
    
    def _save_results(self, results: Dict, path: List[int]) -> None:
        """Save evaluation results to files."""
        # Save metrics as JSON
        metrics_file = self.save_dir / f"trajectory_metrics_{self.embedding_key}.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in results.items()
            }
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save path
        path_file = self.save_dir / f"trajectory_path_{self.embedding_key}.npy"
        np.save(path_file, np.array(path))
        
        # Save as CSV for easy viewing
        csv_file = self.save_dir / f"trajectory_metrics_{self.embedding_key}.csv"
        df = pd.DataFrame([results])
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV to {csv_file}")


def evaluate_trajectory(
    adata: ad.AnnData,
    embedding_key: str,
    time_key: str,
    save_dir: Union[str, Path],
    **kwargs
) -> Dict:
    """
    Convenience function to evaluate trajectory preservation.
    
    Args:
        adata: AnnData object
        embedding_key: Key for embeddings in adata.obsm
        time_key: Key for ground truth time in adata.obs
        save_dir: Directory to save results
        **kwargs: Additional arguments for TrajectoryEvaluator
    
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = TrajectoryEvaluator(
        adata=adata,
        embedding_key=embedding_key,
        time_key=time_key,
        save_dir=save_dir,
        **kwargs
    )
    return evaluator.evaluate()
