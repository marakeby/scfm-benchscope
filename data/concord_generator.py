"""
Generate synthetic single-cell datasets using CONCORD's simulation framework.

This module provides functions to generate datasets with different topologies
(cluster, trajectory, loop, tree) and batch effects for evaluation in scFM_eval.

IMPORTANT: This module generates datasets EXACTLY as CONCORD's benchmark does,
including storing ground truth embeddings (PCA_no_noise, PCA_wt_noise) that
are used for computing trustworthiness and other geometry metrics.

The parameters are matched to the official CONCORD benchmark notebooks:
- simulation_cluster_full.ipynb
- simulation_trajectory_full.ipynb
- simulation_oneloop_full.ipynb
- simulation_tree_full.ipynb

NOTE ON SYNTHETIC DATA BENCHMARKING:
=====================================
Synthetic data uses generic gene names (Gene_1, Gene_2, etc.) without real
biological identities. This means:

APPROPRIATE for synthetic benchmarks:
- CONCORD variants (concord_hcl, concord_knn, contrastive)
- Integration methods (scVI, Harmony, LIGER, Scanorama)
- PCA, UMAP (baseline methods)

NOT APPROPRIATE for synthetic benchmarks:
- scConcept: Requires real Ensembl IDs for its pretrained tokenizer
- scimilarity: Requires gene ontology and real gene annotations
- Other foundation models with pretrained gene vocabularies

Use real biological datasets (BRCA, Bassez, etc.) for evaluating foundation models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from concord.simulation import (
    BatchConfig,
    ClusterConfig,
    SimConfig,
    Simulation,
    TrajectoryConfig,
    TreeConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Ground Truth & Metadata Helpers
# ============================================================================


def _compute_ground_truth_embeddings(adata: ad.AnnData, n_comps: int = 30) -> None:
    """
    Compute ground truth PCA embeddings for benchmark evaluation.
    
    CONCORD benchmarks use these for trustworthiness and geometry metrics:
    - PCA_no_noise: PCA on the no-noise layer (true signal)
    - PCA_wt_noise: PCA on the with-noise layer
    
    The naming follows CONCORD's official benchmark notebooks exactly.
    
    Args:
        adata: AnnData object with 'no_noise' and 'wt_noise' layers
        n_comps: Number of PCA components (default 30, matching CONCORD benchmarks)
    """
    import concord as ccd
    
    # Compute PCA on no-noise data (ground truth)
    # Naming matches CONCORD benchmark: PCA_no_noise (not just 'no_noise')
    if 'no_noise' in adata.layers:
        ccd.ul.run_pca(
            adata, 
            source_key='no_noise', 
            result_key='PCA_no_noise',  # stores in obsm['PCA_no_noise']
            n_pc=min(n_comps, adata.n_vars - 1), 
            random_state=42
        )
        # Also store as 'no_noise' for compatibility with some CONCORD functions
        adata.obsm['no_noise'] = adata.obsm['PCA_no_noise'].copy()
    
    # Compute PCA on with-noise data
    # Naming matches CONCORD benchmark: PCA_wt_noise (not just 'wt_noise')
    if 'wt_noise' in adata.layers:
        ccd.ul.run_pca(
            adata, 
            source_key='wt_noise', 
            result_key='PCA_wt_noise',  # stores in obsm['PCA_wt_noise']
            n_pc=min(n_comps, adata.n_vars - 1), 
            random_state=42
        )
        # Also store as 'wt_noise' for compatibility with some CONCORD functions
        adata.obsm['wt_noise'] = adata.obsm['PCA_wt_noise'].copy()
    
    # Store ground truth keys in uns for easy access
    # Use the PCA_ prefixed names as primary (matching CONCORD benchmarks)
    adata.uns['groundtruth_key'] = 'PCA_no_noise'
    adata.uns['groundtruth_dispersion_key'] = 'PCA_wt_noise'


# NOTE: Gene identifier mapping removed for synthetic data.
# Synthetic data uses generic gene names (Gene_1, Gene_2, etc.) which is appropriate
# for benchmarking integration methods but NOT for foundation models that require
# real gene annotations. See module docstring for details.


# ============================================================================
# Dataset Generation Functions (Matching Official CONCORD Benchmark Parameters)
# ============================================================================


def generate_cluster_dataset(
    n_cells: Union[int, List[int]] = [200, 100, 100, 50, 30],
    n_genes: Union[int, List[int]] = [100, 100, 50, 30, 20],
    n_states: int = 5,
    n_batches: int = 2,
    seed: int = 42,
    output_path: Optional[Union[Path, str]] = None,
    # State parameters (matching simulation_cluster_full.ipynb)
    state_level: float = 5.0,
    state_min_level: float = 0.0,
    state_dispersion: Union[float, List[float]] = [5.0, 4.0, 4.0, 3.0, 2.0],
    global_non_specific_gene_fraction: float = 0.10,
    pairwise_non_specific_gene_fraction: Optional[Dict] = None,
    # Batch parameters
    batch_level: Union[float, List[float]] = [5.0, 5.0],
    batch_dispersion: Union[float, List[float]] = [3.0, 3.0],
    batch_feature_frac: float = 0.15,
    latent_dim: int = 30,
) -> ad.AnnData:
    """
    Generate a cluster-based synthetic dataset matching CONCORD benchmark.
    
    Default parameters match simulation_cluster_full.ipynb exactly.
    
    Args:
        n_cells: Number of cells per cluster (list) or total (int)
        n_genes: Number of genes per cluster (list) or total (int)
        n_states: Number of clusters
        n_batches: Number of batches
        seed: Random seed
        output_path: Path to save the dataset (optional)
        state_level: Expression level for cluster-specific genes
        state_min_level: Minimum expression level
        state_dispersion: Noise dispersion per cluster
        global_non_specific_gene_fraction: Fraction of genes shared across all clusters
        pairwise_non_specific_gene_fraction: Dict of (cluster_i, cluster_j) -> fraction
        batch_level: Batch effect level
        batch_dispersion: Batch effect noise
        batch_feature_frac: Fraction of genes affected by batch
        latent_dim: Latent dimension for PCA embeddings
        
    Returns:
        AnnData object with synthetic cluster data
    """
    # Default pairwise non-specific gene fractions (matching benchmark)
    if pairwise_non_specific_gene_fraction is None:
        pairwise_non_specific_gene_fraction = {
            (0, 1): 0.7,
            (2, 3): 0.4,
        }
    
    sim_cfg = SimConfig(
        n_cells=n_cells,
        n_genes=n_genes,
        seed=seed,
        non_neg=True,
        to_int=True,
    )
    
    cluster_cfg = ClusterConfig(
        n_states=n_states,
        distribution="normal",
        level=state_level,
        min_level=state_min_level,
        dispersion=state_dispersion,
        program_structure="uniform",
        program_on_time_fraction=0.3,
        global_non_specific_gene_fraction=global_non_specific_gene_fraction,
        pairwise_non_specific_gene_fraction=pairwise_non_specific_gene_fraction,
    )
    
    batch_cfg = BatchConfig(
        n_batches=n_batches,
        effect_type="batch_specific_features",
        distribution="normal",
        level=batch_level if isinstance(batch_level, list) else [batch_level] * n_batches,
        dispersion=batch_dispersion if isinstance(batch_dispersion, list) else [batch_dispersion] * n_batches,
        feature_frac=batch_feature_frac,
        cell_proportion=[1 / n_batches] * n_batches,
    )
    
    sim = Simulation(sim_cfg, cluster_cfg, batch_cfg)
    adata, adata_pre = sim.simulate_data()
    
    # Add metadata
    adata.obs["topology"] = "cluster"
    adata.uns["topology"] = "cluster"
    adata.uns["n_states"] = n_states
    adata.uns["state_key"] = "cluster"
    adata.uns["batch_key"] = "batch"
    
    # Expected Betti numbers for clusters
    adata.uns["expected_betti_numbers"] = [n_states, 0, 0]
    
    # Compute ground truth embeddings for benchmark metrics
    _compute_ground_truth_embeddings(adata, n_comps=latent_dim)
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)
        logger.info(f"Saved cluster dataset to {output_path}")
    
    return adata


def generate_trajectory_dataset(
    n_cells: int = 2000,
    n_genes: int = 1000,
    program_num: int = 5,
    n_batches: int = 2,
    seed: int = 42,
    output_path: Optional[Union[Path, str]] = None,
    # State parameters (matching simulation_trajectory_full.ipynb)
    state_level: float = 10.0,
    state_min_level: float = 0.0,
    state_dispersion: float = 6.0,
    program_on_time_fraction: float = 0.2,
    cell_block_size_ratio: float = 0.6,
    # Batch parameters
    batch_level: Union[float, List[float]] = [10.0, 10.0],
    batch_dispersion: Union[float, List[float]] = [6.0, 6.0],
    batch_feature_frac: float = 0.1,
    latent_dim: int = 30,
) -> ad.AnnData:
    """
    Generate a linear trajectory synthetic dataset matching CONCORD benchmark.
    
    Default parameters match simulation_trajectory_full.ipynb exactly.
    
    Args:
        n_cells: Total number of cells
        n_genes: Total number of genes
        program_num: Number of expression programs along trajectory
        n_batches: Number of batches
        seed: Random seed
        output_path: Path to save the dataset (optional)
        state_level: Expression level
        state_min_level: Minimum expression level
        state_dispersion: Noise dispersion
        program_on_time_fraction: Fraction of time each program is fully on
        cell_block_size_ratio: Cell block size ratio
        batch_level: Batch effect level
        batch_dispersion: Batch effect noise
        batch_feature_frac: Fraction of genes affected by batch
        latent_dim: Latent dimension for PCA embeddings
        
    Returns:
        AnnData object with synthetic trajectory data
    """
    sim_cfg = SimConfig(
        n_cells=n_cells,
        n_genes=n_genes,
        seed=seed,
        non_neg=True,
        to_int=True,
    )
    
    traj_cfg = TrajectoryConfig(
        program_num=program_num,
        program_structure="linear_bidirectional",
        cell_block_size_ratio=cell_block_size_ratio,
        program_on_time_fraction=program_on_time_fraction,
        distribution="normal",
        level=state_level,
        dispersion=state_dispersion,
        min_level=state_min_level,
        loop_to=None,
    )
    
    batch_cfg = BatchConfig(
        n_batches=n_batches,
        effect_type="batch_specific_features",
        distribution="normal",
        level=batch_level if isinstance(batch_level, list) else [batch_level] * n_batches,
        dispersion=batch_dispersion if isinstance(batch_dispersion, list) else [batch_dispersion] * n_batches,
        feature_frac=batch_feature_frac,
    )
    
    sim = Simulation(sim_cfg, traj_cfg, batch_cfg)
    adata, adata_pre = sim.simulate_data()
    
    # Add metadata
    adata.obs["topology"] = "trajectory"
    adata.uns["topology"] = "trajectory"
    adata.uns["program_num"] = program_num
    adata.uns["state_key"] = "time"
    adata.uns["batch_key"] = "batch"
    
    # Expected Betti numbers for trajectory: [1, 0, 0] (one connected component, no loops)
    adata.uns["expected_betti_numbers"] = [1, 0, 0]
    
    # Compute ground truth embeddings for benchmark metrics
    _compute_ground_truth_embeddings(adata, n_comps=latent_dim)
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)
        logger.info(f"Saved trajectory dataset to {output_path}")
    
    return adata


def generate_loop_dataset(
    n_cells: int = 2000,
    n_genes: int = 1000,
    program_num: int = 5,
    n_batches: int = 2,
    seed: int = 42,
    output_path: Optional[Union[Path, str]] = None,
    # State parameters (matching simulation_oneloop_full.ipynb)
    state_level: float = 10.0,
    state_min_level: float = 0.0,
    state_dispersion: float = 6.0,
    program_on_time_fraction: float = 0.2,
    cell_block_size_ratio: float = 0.6,
    loop_to: Union[int, List[int]] = [0],
    # Batch parameters
    batch_level: Union[float, List[float]] = [10.0, 10.0],
    batch_dispersion: Union[float, List[float]] = [6.0, 6.0],
    batch_feature_frac: float = 0.1,
    latent_dim: int = 30,
) -> ad.AnnData:
    """
    Generate a loop (circular trajectory) synthetic dataset matching CONCORD benchmark.
    
    Default parameters match simulation_oneloop_full.ipynb exactly.
    
    Args:
        n_cells: Total number of cells
        n_genes: Total number of genes
        program_num: Number of expression programs
        n_batches: Number of batches
        seed: Random seed
        output_path: Path to save the dataset (optional)
        state_level: Expression level
        state_min_level: Minimum expression level
        state_dispersion: Noise dispersion
        program_on_time_fraction: Fraction of time each program is fully on
        cell_block_size_ratio: Cell block size ratio
        loop_to: Which program(s) to loop back to
        batch_level: Batch effect level
        batch_dispersion: Batch effect noise
        batch_feature_frac: Fraction of genes affected by batch
        latent_dim: Latent dimension for PCA embeddings
        
    Returns:
        AnnData object with synthetic loop data
    """
    sim_cfg = SimConfig(
        n_cells=n_cells,
        n_genes=n_genes,
        seed=seed,
        non_neg=True,
        to_int=True,
    )
    
    loop_cfg = TrajectoryConfig(
        program_num=program_num,
        loop_to=loop_to,
        cell_block_size_ratio=cell_block_size_ratio,
        program_structure="linear_bidirectional",
        program_on_time_fraction=program_on_time_fraction,
        distribution="normal",
        level=state_level,
        dispersion=state_dispersion,
        min_level=state_min_level,
    )
    
    batch_cfg = BatchConfig(
        n_batches=n_batches,
        effect_type="batch_specific_features",
        distribution="normal",
        level=batch_level if isinstance(batch_level, list) else [batch_level] * n_batches,
        dispersion=batch_dispersion if isinstance(batch_dispersion, list) else [batch_dispersion] * n_batches,
        feature_frac=batch_feature_frac,
    )
    
    sim = Simulation(sim_cfg, loop_cfg, batch_cfg)
    adata, adata_pre = sim.simulate_data()
    
    # Add metadata
    adata.obs["topology"] = "loop"
    adata.uns["topology"] = "loop"
    adata.uns["program_num"] = program_num
    adata.uns["state_key"] = "time"
    adata.uns["batch_key"] = "batch"
    
    # Expected Betti numbers for loop: [1, 1, 0] (one connected component, one loop)
    adata.uns["expected_betti_numbers"] = [1, 1, 0]
    
    # Compute ground truth embeddings for benchmark metrics
    _compute_ground_truth_embeddings(adata, n_comps=latent_dim)
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)
        logger.info(f"Saved loop dataset to {output_path}")
    
    return adata


def generate_tree_dataset(
    n_cells: int = 1000,
    n_genes: int = 1000,
    branching_factor: Union[int, List[int]] = 2,
    depth: int = 2,
    n_batches: int = 2,
    seed: int = 42,
    output_path: Optional[Union[Path, str]] = None,
    # State parameters (matching simulation_tree_full.ipynb)
    state_level: float = 10.0,
    state_min_level: float = 1.0,
    state_dispersion: float = 3.0,
    program_on_time_fraction: float = 0.10,
    program_gap_size: int = 1,
    program_decay: float = 0.7,
    cellcount_decay: float = 1.0,
    noise_in_block: bool = False,
    # Batch parameters
    batch_level: Union[float, List[float]] = [10.0, 10.0],
    batch_dispersion: Union[float, List[float]] = [3.0, 3.0],
    batch_feature_frac: float = 0.1,
    latent_dim: int = 30,
) -> ad.AnnData:
    """
    Generate a tree (hierarchical branching) synthetic dataset matching CONCORD benchmark.
    
    Default parameters match simulation_tree_full.ipynb exactly.
    
    Args:
        n_cells: Total number of cells
        n_genes: Total number of genes
        branching_factor: Number of branches per level (int or list)
        depth: Tree depth
        n_batches: Number of batches
        seed: Random seed
        output_path: Path to save the dataset (optional)
        state_level: Expression level
        state_min_level: Minimum expression level
        state_dispersion: Noise dispersion
        program_on_time_fraction: Fraction of time each program is fully on
        program_gap_size: Gap between program activations
        program_decay: Decay factor for program intensity
        cellcount_decay: Decay factor for cell count
        noise_in_block: Whether to add noise within each block
        batch_level: Batch effect level
        batch_dispersion: Batch effect noise
        batch_feature_frac: Fraction of genes affected by batch
        latent_dim: Latent dimension for PCA embeddings
        
    Returns:
        AnnData object with synthetic tree data
    """
    sim_cfg = SimConfig(
        n_cells=n_cells,
        n_genes=n_genes,
        seed=seed,
        non_neg=True,
        to_int=True,
    )
    
    tree_cfg = TreeConfig(
        branching_factor=branching_factor,
        depth=depth,
        distribution="normal",
        level=state_level,
        min_level=state_min_level,
        dispersion=state_dispersion,
        program_structure="dimension_increase",
        program_on_time_fraction=program_on_time_fraction,
        program_gap_size=program_gap_size,
        program_decay=program_decay,
        cellcount_decay=cellcount_decay,
        noise_in_block=noise_in_block,
    )
    
    batch_cfg = BatchConfig(
        n_batches=n_batches,
        effect_type="batch_specific_features",
        distribution="normal",
        level=batch_level if isinstance(batch_level, list) else [batch_level] * n_batches,
        dispersion=batch_dispersion if isinstance(batch_dispersion, list) else [batch_dispersion] * n_batches,
        feature_frac=batch_feature_frac,
    )
    
    sim = Simulation(sim_cfg, tree_cfg, batch_cfg)
    adata, adata_pre = sim.simulate_data()
    
    # Add metadata
    adata.obs["topology"] = "tree"
    adata.uns["topology"] = "tree"
    adata.uns["branching_factor"] = branching_factor
    adata.uns["depth"] = depth
    adata.uns["state_key"] = "time"
    adata.uns["batch_key"] = "batch"
    
    # Expected Betti numbers for tree: [1, 0, 0] (one connected component, no loops)
    adata.uns["expected_betti_numbers"] = [1, 0, 0]
    
    # Compute ground truth embeddings for benchmark metrics
    _compute_ground_truth_embeddings(adata, n_comps=latent_dim)
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)
        logger.info(f"Saved tree dataset to {output_path}")
    
    return adata


def generate_concord_dataset_by_name(
    dataset_name: str,
    output_path: Optional[Union[Path, str]] = None,
    n_cells: Optional[int] = None,
    n_genes: Optional[int] = None,
    n_batches: int = 2,
    seed: int = 42,
    **kwargs,
) -> ad.AnnData:
    """
    Generate a specific CONCORD dataset by name with benchmark-matched parameters.
    
    Args:
        dataset_name: Name of the dataset (cluster, trajectory, loop, or tree)
        output_path: Path to save the dataset (optional)
        n_cells: Override default cell count (optional)
        n_genes: Override default gene count (optional)
        n_batches: Number of batches
        seed: Random seed
        **kwargs: Additional parameters passed to the specific generator
        
    Returns:
        AnnData object with synthetic data
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_name = dataset_name.lower().replace("concord_", "").replace(".h5ad", "")
    
    if dataset_name == "cluster":
        # Use benchmark defaults unless overridden
        return generate_cluster_dataset(
            n_cells=n_cells if n_cells is not None else [200, 100, 100, 50, 30],
            n_genes=n_genes if n_genes is not None else [100, 100, 50, 30, 20],
            n_batches=n_batches,
            seed=seed,
            output_path=output_path,
            **kwargs,
        )
    elif dataset_name == "trajectory":
        return generate_trajectory_dataset(
            n_cells=n_cells if n_cells is not None else 2000,
            n_genes=n_genes if n_genes is not None else 1000,
            n_batches=n_batches,
            seed=seed,
            output_path=output_path,
            **kwargs,
        )
    elif dataset_name == "loop":
        return generate_loop_dataset(
            n_cells=n_cells if n_cells is not None else 2000,
            n_genes=n_genes if n_genes is not None else 1000,
            n_batches=n_batches,
            seed=seed,
            output_path=output_path,
            **kwargs,
        )
    elif dataset_name == "tree":
        return generate_tree_dataset(
            n_cells=n_cells if n_cells is not None else 1000,
            n_genes=n_genes if n_genes is not None else 1000,
            n_batches=n_batches,
            seed=seed,
            output_path=output_path,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. "
            f"Must be one of: cluster, trajectory, loop, tree"
        )


# ============================================================================
# Benchmark Pipeline Integration
# ============================================================================


def run_concord_benchmark_pipeline(
    adata: ad.AnnData,
    embedding_methods: Optional[List[str]] = None,
    external_embedders: Optional[Dict[str, Callable[[ad.AnnData], np.ndarray]]] = None,
    state_key: Optional[str] = None,
    batch_key: str = "batch",
    groundtruth_key: str = "PCA_no_noise",
    latent_dim: int = 30,
    seed: int = 42,
    device: str = "cpu",
    save_dir: Optional[Union[Path, str]] = None,
    run_benchmarks: tuple = ("scib", "probe", "geometry", "topology"),
    run_integration_methods: bool = True,
    concord_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full CONCORD benchmark pipeline on a synthetic dataset.
    
    This function:
    1. Runs integration methods (CONCORD variants, scVI, Harmony, etc.)
    2. Optionally runs external embedding methods (scConcept, scimilarity, etc.)
    3. Computes benchmark metrics (scib, probe, geometry, topology)
    4. Generates visualizations
    
    Args:
        adata: AnnData object (should be generated by one of the generate_* functions)
        embedding_methods: List of CONCORD integration methods to run
            Options: "concord_hcl", "concord_knn", "contrastive", "unintegrated",
                     "scanorama", "liger", "harmony", "scvi", "scanvi"
        external_embedders: Dict mapping method names to embedding functions.
            Each function should take an AnnData and return embeddings (n_cells, latent_dim).
            Example: {"scConcept": my_scconcept_embed_fn, "scimilarity": my_scimilarity_fn}
        state_key: Column in adata.obs for cell state/cluster labels.
            If None, uses adata.uns["state_key"] or "cluster"
        batch_key: Column in adata.obs for batch info
        groundtruth_key: Key in adata.obsm for ground truth embeddings
        latent_dim: Latent dimensionality
        seed: Random seed
        device: Device for CONCORD ("cuda:0", "cpu", etc.)
        save_dir: Directory to save results and plots
        run_benchmarks: Tuple of benchmark types to run
        run_integration_methods: Whether to run CONCORD integration methods
        concord_kwargs: Additional kwargs for CONCORD training
        verbose: Print progress
        
    Returns:
        Dict containing:
            - "adata": Updated AnnData with embeddings
            - "benchmark_results": Combined benchmark DataFrame
            - "profile_logs": Runtime/memory profiling (if integration methods run)
            - "diagrams": Persistence diagrams (if topology run)
            - "geometry_results": Geometry metrics (if geometry run)
    """
    import concord as ccd
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine state key
    if state_key is None:
        state_key = adata.uns.get("state_key", "cluster")
    
    # Get expected Betti numbers
    expected_betti = adata.uns.get("expected_betti_numbers", [1, 0, 0])
    
    results = {"adata": adata}
    all_embedding_keys = [groundtruth_key, "PCA_wt_noise"]
    
    # Default integration methods (appropriate for synthetic data - no foundation models)
    # Foundation models (scConcept, scimilarity) require real gene annotations
    if embedding_methods is None:
        embedding_methods = [
            "concord_hcl", "concord_knn", "contrastive",
            "unintegrated", "scanorama", "liger", "harmony", "scvi",
        ]
    
    # Default CONCORD kwargs
    if concord_kwargs is None:
        concord_kwargs = {
            'batch_size': 32,
            'n_epochs': 20,
            'preload_dense': True,
            'verbose': False,
        }
    
    # =========== Run CONCORD integration methods ===========
    if run_integration_methods:
        if verbose:
            logger.info("Running integration methods pipeline...")
        
        profile_logs = ccd.bm.run_integration_methods_pipeline(
            adata=adata,
            methods=embedding_methods,
            batch_key=batch_key,
            count_layer="counts",
            class_key=state_key,
            latent_dim=latent_dim,
            device=device,
            return_corrected=False,
            seed=seed,
            compute_umap=False,
            verbose=verbose,
            concord_kwargs=concord_kwargs,
            save_dir=save_dir,
        )
        results["profile_logs"] = profile_logs
        all_embedding_keys.extend(embedding_methods)
    
    # =========== Run external embedding methods ===========
    if external_embedders is not None:
        for method_name, embed_fn in external_embedders.items():
            if verbose:
                logger.info(f"Running external method: {method_name}")
            try:
                embeddings = embed_fn(adata)
                adata.obsm[method_name] = embeddings
                all_embedding_keys.append(method_name)
                if verbose:
                    logger.info(f"  {method_name}: shape {embeddings.shape}")
            except Exception as e:
                logger.warning(f"Failed to run {method_name}: {e}")
    
    # =========== Compute UMAP for all embeddings ===========
    if verbose:
        logger.info("Computing UMAP for all embeddings...")
    
    for key in all_embedding_keys:
        if key in adata.obsm:
            try:
                ccd.ul.run_umap(
                    adata, 
                    source_key=key, 
                    result_key=f'{key}_UMAP', 
                    n_components=2, 
                    n_neighbors=30, 
                    min_dist=0.5, 
                    metric='cosine', 
                    random_state=seed
                )
            except Exception as e:
                logger.warning(f"UMAP failed for {key}: {e}")
    
    # =========== Run benchmark pipeline ===========
    if verbose:
        logger.info(f"Running benchmarks: {run_benchmarks}")
    
    benchmark_out = ccd.bm.run_benchmark_pipeline(
        adata,
        embedding_keys=all_embedding_keys,
        state_key=state_key,
        batch_key=batch_key,
        groundtruth_key=groundtruth_key,
        save_dir=save_dir / "benchmarks" if save_dir else None,
        run=run_benchmarks,
        plot_individual=False,
        expected_betti_numbers=expected_betti,
        max_points=min(1500, adata.n_obs),
        seed=seed,
    )
    
    results["benchmark_results"] = benchmark_out.get("combined")
    results["diagrams"] = benchmark_out.get("topology_diagrams")
    results["geometry_results"] = benchmark_out.get("geometry_results")
    results["full_benchmark_output"] = benchmark_out
    
    return results


def plot_benchmark_results(
    adata: ad.AnnData,
    benchmark_results: pd.DataFrame,
    embedding_keys: List[str],
    diagrams: Optional[Dict] = None,
    geometry_results: Optional[Dict] = None,
    state_key: str = "cluster",
    batch_key: str = "batch",
    save_dir: Optional[Union[Path, str]] = None,
    seed: int = 42,
) -> None:
    """
    Generate all benchmark visualizations using CONCORD's plotting functions.
    
    Args:
        adata: AnnData with embeddings
        benchmark_results: Combined benchmark DataFrame
        embedding_keys: List of embedding keys to plot
        diagrams: Persistence diagrams dict (optional)
        geometry_results: Geometry metrics dict (optional)
        state_key: Column for cell state labels
        batch_key: Column for batch labels
        save_dir: Directory to save plots
        seed: Random seed
    """
    import concord as ccd
    import matplotlib.pyplot as plt
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom RC for publication-quality plots
    custom_rc = {'font.family': 'Arial'}
    
    # =========== Embedding plots (UMAP and KNN) ===========
    color_bys = [state_key, batch_key]
    basis_types = ['UMAP', 'KNN']
    
    with plt.rc_context(rc=custom_rc):
        ccd.pl.plot_all_embeddings(
            adata,
            embedding_keys,
            color_bys=color_bys,
            basis_types=basis_types,
            pal={'time': 'viridis', 'batch': 'Set1'},
            k=30,
            edges_width=0,
            font_size=8,
            point_size=2,
            alpha=0.8,
            rasterized=True,
            figsize=(0.9 * len(embedding_keys), 1),
            ncols=len(embedding_keys),
            seed=seed,
            save_dir=save_dir,
            save_format='svg'
        )
    
    # =========== Benchmark table ===========
    if benchmark_results is not None:
        table_plot_kw = dict(pal="viridis", pal_agg="viridis", cmap_method="minmax", dpi=300)
        with plt.rc_context(rc=custom_rc):
            ccd.bm.plot_benchmark_table(
                benchmark_results.dropna(axis=0, how='any'),
                save_path=save_dir / "benchmark_table.svg" if save_dir else None,
                agg_name="Aggregate score",
                figsize=(25, 6),
                **table_plot_kw
            )
    
    # =========== Persistence diagrams ===========
    if diagrams is not None:
        diagrams_ordered = {key: diagrams[key] for key in embedding_keys if key in diagrams}
        if diagrams_ordered:
            ccd.pl.plot_persistence_diagrams(
                diagrams_ordered, 
                base_size=(1.3, 1.5), 
                dpi=300, 
                marker_size=4, 
                n_cols=len(diagrams_ordered), 
                fontsize=10, 
                save_path=save_dir / "persistence_diagrams.pdf" if save_dir else None,
                legend=False, 
                label_axes=False, 
                axis_ticks=False
            )
            
            ccd.pl.plot_betti_curves(
                diagrams_ordered, 
                nbins=100, 
                base_size=(1.3, 1.5), 
                n_cols=len(diagrams_ordered), 
                fontsize=10, 
                save_path=save_dir / "betti_curves.pdf" if save_dir else None,
                dpi=300, 
                legend=False, 
                label_axes=False, 
                axis_ticks=False
            )
    
    # =========== Geometry plots ===========
    if geometry_results is not None:
        # Trustworthiness
        if 'trustworthiness' in geometry_results:
            trustworthiness_scores = geometry_results['trustworthiness']['scores']
            with plt.rc_context(rc=custom_rc):
                ccd.pl.plot_trustworthiness(
                    trustworthiness_scores, 
                    text_shift=0.2, 
                    min_gap=0.002, 
                    legend=False, 
                    save_path=save_dir / "trustworthiness.pdf" if save_dir else None,
                    figsize=(2.8, 1.9)
                )
        
        # State dispersion correlation
        if 'state_dispersion_corr' in geometry_results:
            dispersion_dict = geometry_results['state_dispersion_corr']['dispersion']
            correlation_df = geometry_results['state_dispersion_corr']['correlation']
            with plt.rc_context(rc=custom_rc):
                ccd.pl.plot_geometry_scatter(
                    data_dict=dispersion_dict,
                    correlation=correlation_df,
                    s=30, 
                    c='darkblue',
                    ground_key='PCA_wt_noise',
                    linear_fit=True,
                    n_cols=len(dispersion_dict), 
                    figsize=(1.5, 1.75), 
                    dpi=300, 
                    save_path=save_dir / "state_dispersion_scatter.pdf" if save_dir else None
                )
    
    # =========== Heatmaps ===========
    _, _, state_pal = ccd.pl.get_color_mapping(adata, state_key, pal='Paired', seed=seed)
    _, _, batch_pal = ccd.pl.get_color_mapping(adata, batch_key, pal='Set1', seed=seed)
    pal = {state_key: state_pal, batch_key: batch_pal}
    
    with plt.rc_context(rc=custom_rc):
        figsize = (2.3, 1.8)
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 3, figsize[1]), dpi=600)
        
        ccd.pl.heatmap_with_annotations(
            adata, val='no_noise', obs_keys=[state_key], ax=axes[0],
            use_clustermap=False, pal=pal, yticklabels=False, 
            cluster_cols=False, cluster_rows=False, value_annot=False,
            cmap='viridis', title='State (no noise)'
        )
        ccd.pl.heatmap_with_annotations(
            adata, val='wt_noise', obs_keys=[state_key], ax=axes[1],
            use_clustermap=False, pal=pal, yticklabels=False,
            cluster_cols=False, cluster_rows=False, value_annot=False,
            cmap='viridis', title='State + noise'
        )
        ccd.pl.heatmap_with_annotations(
            adata, val='X', obs_keys=[state_key, batch_key], ax=axes[2],
            use_clustermap=False, pal=pal, yticklabels=False,
            cluster_cols=False, cluster_rows=False, value_annot=False,
            cmap='viridis', title='State + noise + batch'
        )
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "data_heatmaps.svg", dpi=600, bbox_inches='tight')
        plt.close()


# ============================================================================
# Convenience function for full pipeline
# ============================================================================


def run_full_topology_benchmark(
    topology: str,
    external_embedders: Optional[Dict[str, Callable]] = None,
    save_dir: Optional[Union[Path, str]] = None,
    seed: int = 42,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to generate data, run benchmarks, and plot results.
    
    Args:
        topology: One of "cluster", "trajectory", "loop", "tree"
        external_embedders: Dict of external embedding functions
        save_dir: Directory to save all outputs
        seed: Random seed
        device: Device for CONCORD
        **kwargs: Additional args passed to the dataset generator
        
    Returns:
        Dict with all results
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    logger.info(f"Generating {topology} dataset...")
    adata = generate_concord_dataset_by_name(
        dataset_name=topology,
        output_path=save_dir / f"concord_{topology}.h5ad" if save_dir else None,
        seed=seed,
        **kwargs,
    )
    
    # Run benchmark pipeline
    logger.info("Running benchmark pipeline...")
    results = run_concord_benchmark_pipeline(
        adata=adata,
        external_embedders=external_embedders,
        save_dir=save_dir,
        seed=seed,
        device=device,
    )
    
    # Get all embedding keys (using PCA_ prefixed names for CONCORD benchmark compatibility)
    all_keys = ["PCA_no_noise", "PCA_wt_noise"]
    for key in adata.obsm.keys():
        if not key.endswith("_UMAP") and key not in all_keys:
            all_keys.append(key)
    
    # Plot results
    logger.info("Generating plots...")
    plot_benchmark_results(
        adata=results["adata"],
        benchmark_results=results["benchmark_results"],
        embedding_keys=all_keys,
        diagrams=results.get("diagrams"),
        geometry_results=results.get("geometry_results"),
        state_key=adata.uns.get("state_key", "cluster"),
        batch_key="batch",
        save_dir=save_dir,
        seed=seed,
    )
    
    logger.info("Done!")
    return results
