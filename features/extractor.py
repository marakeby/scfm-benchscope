from __future__ import annotations

from typing import Any, ClassVar, Dict, FrozenSet, Optional

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
        "nicheformer": "X_nicheformer",
        "scconcept": "X_scconcept",
        "state": "X_state",
        "mock": "X_mock",
    }

    # Subclasses set which ``params`` keys hold filesystem paths relative to ``MODELS_PATH``
    # (see ``setup_path.resolve_under_models_path``). Empty = no automatic resolution.
    MODELS_PATH_KEYS: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self, params: Dict[str, Any]):
        self.method: str = params.get("method", "")
        self.params: Dict[str, Any] = params.get("params", {}) or {}
        if self.MODELS_PATH_KEYS:
            from setup_path import resolve_models_path_params

            resolve_models_path_params(self.params, self.MODELS_PATH_KEYS, recurse=True)
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