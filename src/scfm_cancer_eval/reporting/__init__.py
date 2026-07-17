"""Result discovery, comparison exports, and human-readable reports."""

from scfm_cancer_eval.reporting.comparison import (
    COMPARISON_SCHEMA_NAME,
    COMPARISON_SCHEMA_VERSION,
    ComparisonArtifacts,
    ComparisonRecord,
    build_comparison_payload,
    build_comparison_records,
    write_comparison_exports,
)
from scfm_cancer_eval.reporting.discovery import (
    DiscoveryIssue,
    DiscoveryResult,
    ResultDiscoveryError,
    RunSummary,
    discover_results,
)

__all__ = [
    "COMPARISON_SCHEMA_NAME",
    "COMPARISON_SCHEMA_VERSION",
    "ComparisonArtifacts",
    "ComparisonRecord",
    "DiscoveryIssue",
    "DiscoveryResult",
    "ResultDiscoveryError",
    "RunSummary",
    "build_comparison_payload",
    "build_comparison_records",
    "discover_results",
    "write_comparison_exports",
]
