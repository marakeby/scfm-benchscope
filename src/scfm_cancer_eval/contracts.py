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
    """Validated paths, identity, and payload returned by the public API."""

    run_id: str
    output_dir: Path
    results_path: Path
    metrics_path: Path
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        results_path: str | Path,
        *,
        expected_run_id: str | None = None,
    ) -> "RunResult":
        """Load and validate a completed run from ``results.json``."""
        from scfm_cancer_eval.utils.results_json import read_results_json

        resolved_path = Path(results_path)
        payload = read_results_json(str(resolved_path))
        run_id = payload["run"]["run_id"]
        if expected_run_id is not None and run_id != expected_run_id:
            raise ValueError(
                f"results run_id {run_id!r} does not match expected "
                f"{expected_run_id!r}"
            )
        return cls(
            run_id=run_id,
            output_dir=resolved_path.parent,
            results_path=resolved_path,
            metrics_path=resolved_path.parent / "metrics.json",
            payload=payload,
        )

    @property
    def status(self) -> str:
        return str(self.payload["run"]["status"])

    @property
    def evaluations(self) -> list[Mapping[str, Any]]:
        return list(self.payload["evaluations"])

