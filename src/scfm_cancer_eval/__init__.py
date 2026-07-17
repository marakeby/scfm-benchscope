"""scfm-cancer-eval: single-cell foundation model evaluation harness."""

from scfm_cancer_eval.api import EvaluationOptions, evaluate
from scfm_cancer_eval.contracts import (
    EvaluationModelConfig,
    ModelAdapter,
    RunResult,
)
from scfm_cancer_eval.utils.results_json import ResultsValidationError

__version__ = "0.1.0"

__all__ = [
    "EvaluationModelConfig",
    "EvaluationOptions",
    "ModelAdapter",
    "RunResult",
    "ResultsValidationError",
    "evaluate",
]
