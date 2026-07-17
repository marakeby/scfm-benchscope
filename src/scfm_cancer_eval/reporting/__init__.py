"""Result discovery, comparison exports, and human-readable reports."""

from scfm_cancer_eval.reporting.discovery import (
    DiscoveryIssue,
    DiscoveryResult,
    ResultDiscoveryError,
    RunSummary,
    discover_results,
)

__all__ = [
    "DiscoveryIssue",
    "DiscoveryResult",
    "ResultDiscoveryError",
    "RunSummary",
    "discover_results",
]
