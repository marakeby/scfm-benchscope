"""
Deterministic linear random projection — stands in for a foundation model during tests.

Use `MockEmbeddingExtractor` (features.mock_extractor) to plug this into run_exp.
"""

from __future__ import annotations

import numpy as np


class MockEmbeddingModel:
    """
    Maps a cell–gene matrix to a low-dimensional embedding via a fixed random matrix.

    Same (seed, n_genes, out_dim) always yields the same projection weights.
    """

    def __init__(
        self,
        n_genes: int,
        out_dim: int = 16,
        seed: int = 0,
    ) -> None:
        if n_genes < 1:
            raise ValueError("n_genes must be >= 1")
        self.n_genes = n_genes
        self.out_dim = out_dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Small weights keep embeddings in a stable range for UMAP / metrics.
        self._proj = rng.normal(0.0, 0.05, size=(n_genes, out_dim)).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Row-wise embedding: (n_cells, n_genes) -> (n_cells, out_dim)."""
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")
        if x.shape[1] != self.n_genes:
            raise ValueError(
                f"Expected {self.n_genes} genes (columns), got {x.shape[1]}. "
                "Rebuild the model with the current n_genes after QC/HVG, or set n_genes in YAML."
            )
        dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x, dtype=np.float32)
        return dense @ self._proj
