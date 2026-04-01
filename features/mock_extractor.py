"""
Mock embedding extractor for pipeline tests (no pretrained weights).

Intended use: swap the `embedding` block in a task YAML (e.g. cancer subtyping
`yaml/examples/mock_subtype_eval.yaml`) while keeping the same `dataset` and
`classification` fragments as the real evaluation.

How to add a *real* model (same pattern):
1. Implement a small wrapper in `models/` that loads weights and maps X -> embeddings.
2. Subclass `EmbeddingExtractor`, implement `fit_transform(loader)`:
   - read `loader.adata.X` (dense or sparse),
   - write `loader.adata.obsm[self.output_key] = embeddings`,
   - return the embedding array (numpy).
3. Register `method` -> `X_*` in `EmbeddingExtractor.DEFAULT_OUTPUT_KEYS` and
   `run/run_exp.py` `embedding_method_map` if you want legacy fallbacks.
4. Point YAML `embedding.module` / `embedding.class` at your extractor, set `method`,
   and pass hyperparameters under `embedding.params`.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from features.extractor import EmbeddingExtractor
from models.mock_embedding_model import MockEmbeddingModel
from utils.logs_ import get_logger


class MockEmbeddingExtractor(EmbeddingExtractor):
    """Linear mock embeddings; fast, no GPU, good for pipeline smoke tests."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.log = get_logger()
        self.out_dim = int(self.params.get("out_dim", 16))
        self.seed = int(self.params.get("seed", 0))
        self._n_genes_override: int | None = self.params.get("n_genes")
        self.log.info(
            "MockEmbeddingExtractor out_dim=%s seed=%s", self.out_dim, self.seed
        )

    def fit_transform(self, loader: Any):
        adata = loader.adata
        x = adata.X
        n_genes = (
            int(self._n_genes_override)
            if self._n_genes_override is not None
            else (x.shape[1] if hasattr(x, "shape") else len(adata.var))
        )
        model = MockEmbeddingModel(
            n_genes=n_genes, out_dim=self.out_dim, seed=self.seed
        )
        emb = model.transform(x)
        adata.obsm[self.output_key] = emb
        self.log.info("Mock embeddings shape %s -> obsm[%s]", emb.shape, self.output_key)
        return np.asarray(emb, dtype=np.float32)
