"""Command-line entry point for evaluations and result reporting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

_COMMANDS = {"run", "report", "compare", "candidate", "contract"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scfm-eval",
        description=(
            "Run scFM_eval experiment from YAML, or build reports from validated "
            "results."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "run",
        help="Run an experiment; existing calls may omit this command.",
        add_help=False,
    )

    report = subparsers.add_parser(
        "report",
        help="Discover runs below one output root and create a report bundle.",
    )
    report.add_argument(
        "root",
        nargs="?",
        default=os.environ.get("SCFM_OUTPUT_PATH", "output"),
        help="Output root to search recursively (default: SCFM_OUTPUT_PATH or output).",
    )
    _add_report_options(report)

    compare = subparsers.add_parser(
        "compare",
        help="Compare selected result files or run directories.",
    )
    compare.add_argument(
        "paths",
        nargs="+",
        help="One or more results.json files or directories to search.",
    )
    _add_report_options(compare)

    candidate = subparsers.add_parser(
        "candidate",
        help="Validate model evidence produced by a discovery agent.",
    )
    candidate_commands = candidate.add_subparsers(
        dest="candidate_command",
        required=True,
    )
    validate = candidate_commands.add_parser(
        "validate",
        help="Validate one versioned model-candidate JSON file.",
    )
    validate.add_argument("path", help="Path to model candidate JSON.")

    contract = subparsers.add_parser(
        "contract",
        help="Validate planner, execution, or review JSON.",
    )
    contract_commands = contract.add_subparsers(
        dest="contract_command",
        required=True,
    )
    contract_validate = contract_commands.add_parser(
        "validate",
        help="Validate one versioned onboarding contract.",
    )
    contract_validate.add_argument(
        "kind",
        choices=[
            "model-spec",
            "integration-plan",
            "execution-manifest",
            "review-decision",
        ],
    )
    contract_validate.add_argument("path", help="Path to the JSON document.")
    return parser


def _add_report_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        help="Directory for report.html, comparison.json, and comparison.csv.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested result is missing or invalid.",
    )
    parser.add_argument(
        "--title",
        default="scFM evaluation report",
        help="Title displayed in the HTML report.",
    )


def _run_report(
    roots: list[str],
    output_dir: Path,
    *,
    strict: bool,
    title: str,
) -> int:
    from scfm_cancer_eval.reporting import create_report_bundle

    bundle = create_report_bundle(
        roots,
        output_dir,
        strict=strict,
        title=title,
    )
    print(
        f"Discovered {bundle.discovery.valid_count} valid run(s); "
        f"skipped {len(bundle.discovery.issues)} issue(s)."
    )
    print(f"HTML report: {bundle.html_path}")
    print(f"JSON export: {bundle.comparison.json_path}")
    print(f"CSV export: {bundle.comparison.csv_path}")
    return 0


def _validate_candidate(path: str) -> int:
    from scfm_cancer_eval.onboarding import load_model_candidate

    candidate = load_model_candidate(path)
    print(f"Valid candidate: {candidate.candidate_id}")
    print(f"Model: {candidate.model_name}")
    print(f"Fingerprint: sha256:{candidate.fingerprint}")
    return 0


def _validate_contract(kind: str, path: str) -> int:
    from scfm_cancer_eval.onboarding import (
        load_execution_manifest,
        load_integration_plan,
        load_model_spec,
        load_review_decision,
    )

    loaders = {
        "model-spec": load_model_spec,
        "integration-plan": load_integration_plan,
        "execution-manifest": load_execution_manifest,
        "review-decision": load_review_decision,
    }
    document = loaders[kind](path)
    print(f"Valid {kind}: {document.document_id}")
    print(f"Fingerprint: sha256:{document.fingerprint}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    # Preserve the original CLI: a YAML path as the first argument still runs
    # directly, while ``run`` is an optional explicit spelling.
    if args and args[0] not in _COMMANDS and args[0] not in {"-h", "--help"}:
        from scfm_cancer_eval.run.run_exp import main as run_main

        run_main(args)
        return 0
    if args and args[0] == "run":
        from scfm_cancer_eval.run.run_exp import main as run_main

        run_main(args[1:])
        return 0

    parser = _parser()
    parsed = parser.parse_args(args)
    if parsed.command is None:
        parser.print_help()
        return 0

    try:
        if parsed.command == "report":
            output_dir = (
                Path(parsed.output)
                if parsed.output
                else Path(parsed.root) / "report"
            )
            return _run_report(
                [parsed.root],
                output_dir,
                strict=parsed.strict,
                title=parsed.title,
            )
        if parsed.command == "compare":
            output_dir = (
                Path(parsed.output)
                if parsed.output
                else Path.cwd() / "comparison-report"
            )
            return _run_report(
                parsed.paths,
                output_dir,
                strict=parsed.strict,
                title=parsed.title,
            )
        if (
            parsed.command == "candidate"
            and parsed.candidate_command == "validate"
        ):
            return _validate_candidate(parsed.path)
        if (
            parsed.command == "contract"
            and parsed.contract_command == "validate"
        ):
            return _validate_contract(parsed.kind, parsed.path)
    except ValueError as exc:
        parser.error(str(exc))

    parser.error(f"Unknown command: {parsed.command}")
    return 2
