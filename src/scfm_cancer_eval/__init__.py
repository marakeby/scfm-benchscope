"""scfm-cancer-eval: single-cell foundation model evaluation harness."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "EvaluationModelConfig",
    "EvaluationOptions",
    "ModelAdapter",
    "RunResult",
    "ResultsValidationError",
    "evaluate",
]

_PUBLIC_IMPORTS = {
    "EvaluationModelConfig": (
        "scfm_cancer_eval.contracts",
        "EvaluationModelConfig",
    ),
    "EvaluationOptions": ("scfm_cancer_eval.api", "EvaluationOptions"),
    "ModelAdapter": ("scfm_cancer_eval.contracts", "ModelAdapter"),
    "RunResult": ("scfm_cancer_eval.contracts", "RunResult"),
    "ResultsValidationError": (
        "scfm_cancer_eval.utils.results_json",
        "ResultsValidationError",
    ),
    "evaluate": ("scfm_cancer_eval.api", "evaluate"),
}


def __getattr__(name: str) -> Any:
    """Load scientific dependencies only when their public API is requested."""
    try:
        module_name, attribute_name = _PUBLIC_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
