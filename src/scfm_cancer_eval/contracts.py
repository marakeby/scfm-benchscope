"""Small public contracts shared by library users and the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    """Runtime boundary for a model that produces cell embeddings."""

    output_key: str

    def fit_transform(self, loader: Any) -> Any:
        """Create embeddings for ``loader`` and return the embedding matrix."""


@dataclass(frozen=True)
class EvaluationModelConfig:
    """Serializable reference to an importable model adapter."""

    model_id: str
    adapter: str
    params: Mapping[str, Any] = field(default_factory=dict)
    output_key: str | None = None

    def to_embedding_config(
        self,
        *,
        evaluate_embedding: bool = True,
        visualize: bool = False,
    ) -> dict[str, Any]:
        """Translate this public contract to the existing embedding config shape."""
        module, separator, class_name = self.adapter.rpartition(".")
        if not separator or not module or not class_name:
            raise ValueError(
                "adapter must be a fully qualified class path, "
                "for example 'my_package.models.MyAdapter'"
            )

        config: dict[str, Any] = {
            "method": self.model_id,
            "module": module,
            "class": class_name,
            "viz": bool(visualize),
            "eval": bool(evaluate_embedding),
            "params": dict(self.params),
        }
        if self.output_key:
            config["output_key"] = self.output_key
        return config


@dataclass(frozen=True)
class RunResult:
    """Paths and identity returned by the public evaluation API.

    Stage 3 adds schema validation and parsed result data while retaining these
    fields.
    """

    run_id: str
    output_dir: Path
    results_path: Path
    metrics_path: Path

