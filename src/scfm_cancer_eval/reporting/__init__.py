"""Result discovery, comparison exports, and human-readable reports."""

from scfm_cancer_eval.reporting.bootstrap_metrics import (
    BootstrapMetricsBundle,
    create_bootstrap_metrics_bundle,
)
from scfm_cancer_eval.reporting.collect_metrics import (
    CollectMetricsBundle,
    create_collect_metrics_bundle,
)
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
from scfm_cancer_eval.reporting.html_report import (
    render_html_report,
    write_html_report,
)
from scfm_cancer_eval.reporting.workflow import (
    ReportBundle,
    create_report_bundle,
)

__all__ = [
    "BootstrapMetricsBundle",
    "COMPARISON_SCHEMA_NAME",
    "COMPARISON_SCHEMA_VERSION",
    "CollectMetricsBundle",
    "ComparisonArtifacts",
    "ComparisonRecord",
    "DiscoveryIssue",
    "DiscoveryResult",
    "ResultDiscoveryError",
    "ReportBundle",
    "RunSummary",
    "build_comparison_payload",
    "build_comparison_records",
    "create_bootstrap_metrics_bundle",
    "create_collect_metrics_bundle",
    "create_report_bundle",
    "discover_results",
    "render_html_report",
    "write_comparison_exports",
    "write_html_report",
]
