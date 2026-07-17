"""High-level reporting workflow shared by CLI and library callers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scfm_cancer_eval.reporting.comparison import (
    ComparisonArtifacts,
    write_comparison_exports,
)
from scfm_cancer_eval.reporting.discovery import DiscoveryResult, discover_results
from scfm_cancer_eval.reporting.html_report import write_html_report


@dataclass(frozen=True)
class ReportBundle:
    discovery: DiscoveryResult
    comparison: ComparisonArtifacts
    html_path: Path


def create_report_bundle(
    roots: Iterable[str | Path],
    output_dir: str | Path,
    *,
    strict: bool = False,
    title: str = "scFM evaluation report",
) -> ReportBundle:
    discovery = discover_results(roots, strict=strict)
    if not discovery.runs:
        raise ValueError("No valid results.json files were found")
    comparison = write_comparison_exports(discovery, output_dir)
    html_path = write_html_report(discovery, output_dir, title=title)
    return ReportBundle(
        discovery=discovery,
        comparison=comparison,
        html_path=html_path,
    )
