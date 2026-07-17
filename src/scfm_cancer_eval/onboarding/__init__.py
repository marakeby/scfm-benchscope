"""Contracts used by model discovery and integration planning."""

from scfm_cancer_eval.onboarding.candidate import (
    MODEL_CANDIDATE_SCHEMA_NAME,
    MODEL_CANDIDATE_SCHEMA_VERSION,
    CandidateValidationError,
    ModelCandidate,
    load_model_candidate,
    model_candidate_schema,
    validate_model_candidate,
)

__all__ = [
    "MODEL_CANDIDATE_SCHEMA_NAME",
    "MODEL_CANDIDATE_SCHEMA_VERSION",
    "CandidateValidationError",
    "ModelCandidate",
    "load_model_candidate",
    "model_candidate_schema",
    "validate_model_candidate",
]
