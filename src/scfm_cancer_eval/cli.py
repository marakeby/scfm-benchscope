"""Command-line entry point for evaluations and result reporting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

_COMMANDS = {
    "run",
    "report",
    "compare",
    "candidate",
    "contract",
    "plan",
    "approval",
    "execute",
}


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
            "execution-approval",
            "review-decision",
        ],
    )
    contract_validate.add_argument("path", help="Path to the JSON document.")

    plan = subparsers.add_parser(
        "plan",
        help="Ask a selected AI provider for a proposal-only integration plan.",
    )
    plan.add_argument("candidate", help="Validated model candidate JSON.")
    plan.add_argument(
        "--provider",
        default=os.environ.get("SCFM_PLANNER_PROVIDER", "openai"),
        help="openai, anthropic, or module:attribute (default: openai).",
    )
    plan.add_argument(
        "--model",
        help="Provider model override.",
    )
    plan.add_argument(
        "-o",
        "--output",
        help="New workspace directory (default: planning/<candidate>-<time>).",
    )

    approval = subparsers.add_parser(
        "approval",
        help="Prepare or verify a human-reviewable execution bundle.",
    )
    approval_commands = approval.add_subparsers(
        dest="approval_command",
        required=True,
    )
    prepare = approval_commands.add_parser(
        "prepare",
        help="Resolve pixi.lock and build a bounded execution manifest.",
    )
    prepare.add_argument("candidate", help="Validated model candidate JSON.")
    prepare.add_argument("workspace", help="Ready planner workspace.")
    prepare.add_argument("-o", "--output", required=True)
    prepare.add_argument("--manifest-id", required=True)
    prepare.add_argument("--gpu-type", required=True)
    prepare.add_argument("--gpu-count", type=int, required=True)
    prepare.add_argument("--disk-gb", type=float, required=True)
    prepare.add_argument("--max-runtime-minutes", type=float, required=True)
    prepare.add_argument("--hourly-rate-usd", type=float, required=True)
    prepare.add_argument("--max-budget-usd", type=float, required=True)
    prepare.add_argument("--max-attempts", type=int, default=1)
    prepare.add_argument(
        "--retryable-step",
        action="append",
        choices=[
            "checkout",
            "create_environment",
            "install",
            "download_weights",
            "smoke_test",
            "evaluate",
        ],
    )
    prepare.add_argument("--secret", action="append", default=[])
    prepare.add_argument("--allow-host", action="append", default=[])
    prepare.add_argument("--experiment-path")

    verify = approval_commands.add_parser(
        "verify",
        help="Recheck an approval bundle before review or execution.",
    )
    verify.add_argument("bundle", help="Approval bundle directory.")

    grant = approval_commands.add_parser(
        "grant",
        help="Record that one verified bundle was approved by a merged PR.",
    )
    grant.add_argument("bundle", help="Verified approval bundle.")
    grant.add_argument("-o", "--output", required=True)
    grant.add_argument("--approval-id", required=True)
    grant.add_argument("--identity", required=True)
    grant.add_argument(
        "--method",
        choices=["github_pr", "manual"],
        default="github_pr",
    )
    grant.add_argument("--pr-url", required=True)
    grant.add_argument("--merge-commit", required=True)
    grant.add_argument(
        "--bundle-path",
        required=True,
        help="Relative repository path of the approved bundle.",
    )

    execute = subparsers.add_parser(
        "execute",
        help="Run one human-approved execution manifest with bounded retries.",
    )
    execute.add_argument("bundle", help="Verified approval bundle.")
    execute.add_argument("--approval", required=True)
    execute.add_argument("-o", "--output", required=True)
    execute.add_argument(
        "--transport",
        choices=["fake", "local", "ssh"],
        default="fake",
    )
    execute.add_argument("--local-root")
    execute.add_argument("--ssh-host")
    execute.add_argument("--ssh-user")
    execute.add_argument("--ssh-remote-root")
    execute.add_argument("--ssh-identity-file")
    execute.add_argument("--ssh-port", type=int, default=22)
    execute.add_argument(
        "--keep-job",
        action="store_true",
        help="Leave the remote/local job directory in place after the run.",
    )
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
        load_execution_approval,
        load_execution_manifest,
        load_integration_plan,
        load_model_spec,
        load_review_decision,
    )

    loaders = {
        "model-spec": load_model_spec,
        "integration-plan": load_integration_plan,
        "execution-manifest": load_execution_manifest,
        "execution-approval": load_execution_approval,
        "review-decision": load_review_decision,
    }
    document = loaders[kind](path)
    print(f"Valid {kind}: {document.document_id}")
    print(f"Fingerprint: sha256:{document.fingerprint}")
    return 0


def _plan_candidate(
    candidate_path: str,
    *,
    provider_name: str,
    model: str | None,
    output: str | None,
) -> int:
    from datetime import datetime, timezone

    from scfm_cancer_eval.onboarding import (
        load_model_candidate,
        load_planner_provider,
        plan_candidate,
    )

    candidate = load_model_candidate(candidate_path)
    provider = load_planner_provider(provider_name, model=model)
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("planning") / (
            f"{candidate.candidate_id}-{timestamp}"
        )
    else:
        output_path = Path(output)
    outcome = plan_candidate(candidate, provider, output_path)
    print(f"Planning status: {outcome.status}")
    print(f"Provider: {outcome.provider} ({outcome.model})")
    print(f"Workspace: {outcome.workspace}")
    for issue in outcome.issues:
        print(f"Issue: {issue}")
    return 0


def _prepare_approval(parsed: argparse.Namespace) -> int:
    from scfm_cancer_eval.onboarding import (
        ApprovalOptions,
        prepare_approval_bundle,
    )

    options = ApprovalOptions(
        manifest_id=parsed.manifest_id,
        gpu_type=parsed.gpu_type,
        gpu_count=parsed.gpu_count,
        disk_gb=parsed.disk_gb,
        max_runtime_minutes=parsed.max_runtime_minutes,
        hourly_rate_usd=parsed.hourly_rate_usd,
        max_budget_usd=parsed.max_budget_usd,
        max_attempts=parsed.max_attempts,
        retryable_steps=tuple(
            parsed.retryable_step
            or ("download_weights", "evaluate")
        ),
        secret_names=tuple(parsed.secret),
        additional_network_hosts=tuple(parsed.allow_host),
        experiment_path=parsed.experiment_path,
    )
    bundle = prepare_approval_bundle(
        parsed.candidate,
        parsed.workspace,
        parsed.output,
        options,
    )
    print(f"Approval bundle: {bundle.root}")
    print(f"Manifest: {bundle.root / 'execution-manifest.json'}")
    print(f"Fingerprint: sha256:{bundle.manifest.fingerprint}")
    print("Status: pending human review in one model-specific pull request")
    return 0


def _verify_approval(path: str) -> int:
    from scfm_cancer_eval.onboarding import verify_approval_bundle

    bundle = verify_approval_bundle(path)
    print(f"Valid approval bundle: {bundle.root}")
    print(f"Manifest fingerprint: sha256:{bundle.manifest.fingerprint}")
    return 0


def _grant_approval(parsed: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from scfm_cancer_eval.onboarding import (
        build_execution_approval,
        verify_approval_bundle,
        write_execution_approval,
    )

    bundle = verify_approval_bundle(parsed.bundle)
    approval = build_execution_approval(
        approval_id=parsed.approval_id,
        approved_at=datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        manifest_fingerprint=bundle.manifest.fingerprint,
        bundle_path=parsed.bundle_path,
        identity=parsed.identity,
        method=parsed.method,
        pull_request_url=parsed.pr_url,
        merge_commit=parsed.merge_commit,
    )
    path = write_execution_approval(parsed.output, approval)
    print(f"Execution approval: {path}")
    print(f"Manifest fingerprint: sha256:{approval.manifest_fingerprint}")
    return 0


def _execute_approved(parsed: argparse.Namespace) -> int:
    from scfm_cancer_eval.onboarding import (
        execute_approved_bundle,
        load_job_host,
    )

    host = load_job_host(
        parsed.transport,
        local_root=parsed.local_root,
        ssh_host=parsed.ssh_host,
        ssh_user=parsed.ssh_user,
        ssh_remote_root=parsed.ssh_remote_root,
        ssh_identity_file=parsed.ssh_identity_file,
        ssh_port=parsed.ssh_port,
    )
    outcome = execute_approved_bundle(
        parsed.bundle,
        parsed.approval,
        parsed.output,
        host,
        cleanup=not parsed.keep_job,
    )
    print(f"Execution status: {outcome.status}")
    print(f"Attempts: {outcome.attempts}")
    print(f"Estimated cost USD: {outcome.estimated_cost_usd}")
    print(f"Record: {outcome.record_path}")
    print("Scientific review is still required before publication")
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
        if parsed.command == "plan":
            return _plan_candidate(
                parsed.candidate,
                provider_name=parsed.provider,
                model=parsed.model,
                output=parsed.output,
            )
        if parsed.command == "approval":
            if parsed.approval_command == "prepare":
                return _prepare_approval(parsed)
            if parsed.approval_command == "verify":
                return _verify_approval(parsed.bundle)
            if parsed.approval_command == "grant":
                return _grant_approval(parsed)
        if parsed.command == "execute":
            return _execute_approved(parsed)
    except ValueError as exc:
        parser.error(str(exc))

    parser.error(f"Unknown command: {parsed.command}")
    return 2
