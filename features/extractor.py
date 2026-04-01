from __future__ import annotations

from typing import Any, Dict, Optional

from utils.logs_ import get_logger

logger = get_logger()


class EmbeddingExtractor:
    """
    Unified embedding extractor API.

    Concrete extractors should:
    - inherit from this class
    - implement `fit_transform(loader)` and store embeddings in `loader.adata.obsm[output_key]`
    - return the embedding matrix (numpy array / torch tensor / etc. -> runner will persist to h5ad)
    """

    # Central default mapping (kept here, not in the runner).
    # Existing code expects these exact keys in many places.
    DEFAULT_OUTPUT_KEYS: Dict[str, str] = {
        "PCA": "X_pca",
        "HVG": "X_hvg",
        "scVI": "X_scVI",
        "geneformer": "X_geneformer",
        "scgpt": "X_scGPT",
        "scfoundation": "X_scfoundation",
        "scimilarity": "X_scimilarity",
        "cellplm": "X_CellPLM",
        "mock": "X_mock",
    }

    def __init__(self, params: Dict[str, Any]):
        self.method: str = params.get("method", "")
        self.params: Dict[str, Any] = params.get("params", {}) or {}
        # Optional explicit key override in YAML:
        # embedding.output_key: X_something
        self.output_key: str = (
            params.get("output_key")
            or self.params.get("output_key")
            or self.DEFAULT_OUTPUT_KEYS.get(self.method, f"X_{self.method}")
        )

    @staticmethod
    def validate_config(params: Dict[str, Any]) -> None:
        assert "method" in params, "Missing required parameter: 'method'"

    def fit_transform(self, loader: Any):
        raise NotImplementedError("Concrete extractors must implement fit_transform(loader)")

    def get_output_key(self) -> str:
        return self.output_key