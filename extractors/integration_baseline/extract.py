"""
Integration baseline extractor.

Wraps the existing ``create_baseline_embedding`` function from
``evaluation.baseline_embeddings`` as a first-class extractor so that
integration baselines (harmony, bbknn, scanorama, pca_qc) can be run
through the normal ``run_exp.py`` pipeline with their own YAML configs.

Each baseline method produces an embedding that is stored just like any
FM embedding. Downstream evaluations (batch_effects, biological_signal,
annotation) run exactly the same way as for foundation models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scanpy as sc

from extractors.base_extract import BaseExtractor

logger = logging.getLogger(__name__)


class IntegrationBaselineExtractor(BaseExtractor):
    """Extractor that runs a classical integration method and returns the corrected embedding.

    Supported methods: ``harmony``, ``bbknn``, ``scanorama``, ``pca_qc``,
    ``scvi``, ``scanvi``.

    The ``integration_method`` parameter (required) selects which method to
    use. All other parameters are forwarded to
    ``evaluation.baseline_embeddings.create_baseline_embedding``.

    Example YAML ``embedding`` section::

        embedding:
          method: harmony            # registry key
          viz: true
          eval: true
          params:
            integration_method: harmony
            batch_key: assay
            label_key: cell_type
            n_comps: 50
    """

    def __init__(self, params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(params=params, **kwargs)
        self._integration_method: str = self.params.get("integration_method", "harmony")
        self._batch_key: str = self.params.get("batch_key", "batch")
        self._label_key: str = self.params.get("label_key", "label")
        self._n_comps: int = int(self.params.get("n_comps", 50))
        self._random_seed: int = int(self.params.get("random_seed", 42))
        self._num_workers: int = int(self.params.get("num_workers", 6))

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._integration_method

    @property
    def embedding_dim(self) -> int:
        return self._n_comps

    def load_model(self) -> None:
        """No model to load for classical integration methods."""
        logger.info(
            "%s integration baseline extractor ready (no model to load)",
            self._integration_method,
        )
        self._model_loaded = True

    def extract_embeddings(self, adata: sc.AnnData) -> np.ndarray:
        """Run the integration method on *adata* and return the embedding.

        Args:
            adata: AnnData object (expression matrix + obs metadata).

        Returns:
            Embedding array of shape ``(n_cells, embedding_dim)``.
        """
        from evaluation.baseline_embeddings import create_baseline_embedding

        logger.info(
            "Running integration baseline: %s  (n_cells=%d, batch_key=%s, label_key=%s)",
            self._integration_method,
            adata.n_obs,
            self._batch_key,
            self._label_key,
        )

        embedding = create_baseline_embedding(
            adata=adata,
            method=self._integration_method,
            batch_key=self._batch_key,
            label_key=self._label_key,
            use_rep="X",
            n_comps=self._n_comps,
            random_seed=self._random_seed,
            num_workers=self._num_workers,
        )

        logger.info(
            "%s embedding shape: %s",
            self._integration_method,
            embedding.shape,
        )
        return embedding
