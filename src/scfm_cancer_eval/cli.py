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
    "review",
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
        help=(
            "Build a results.json comparison report, or collect/bootstrap "
            "embedding and classification metric tables for the dashboards."
        ),
    )
    report.add_argument(
        "root",
        nargs="?",
        default=os.environ.get("SCFM_OUTPUT_PATH", "output"),
        help="Output root to search recursively (default: SCFM_OUTPUT_PATH or output).",
    )
    report_mode = report.add_mutually_exclusive_group()
    report_mode.add_argument(
        "--collect",
        action="store_true",
        help=(
            "Aggregate embedding_metrics.csv and classification CV metric CSVs "
            "into dashboard CSV/JSON tables (no results.json report)."
        ),
    )
    report_mode.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "Run repeated subsampled embedding evaluation and write bootstrap "
            "aggregate CSV/JSON tables (no results.json report)."
        ),
    )
    report.add_argument(
        "--kind",
        choices=("all", "embedding", "classification"),
        default="all",
        help="With --collect: which metric family to aggregate (default: all).",
    )
    report.add_argument(
        "--folder-substring",
        default="cell_type",
        help=(
            "With --collect/--bootstrap: require this substring in a path "
            "component when discovering embedding runs (default: cell_type)."
        ),
    )
    report.add_argument(
        "--score-col",
        default="randomforest",
        help="With --collect: preferred classifier column in *cv_metrics.csv.",
    )
    report.add_argument(
        "--keep-luad-cancer-stage",
        action="store_true",
        help="With --collect: keep rows where exp == luad_cancer_stage.",
    )
    report.add_argument(
        "--include-arxiv",
        action="store_true",
        help="With --collect: include runs whose path contains 'arxiv'.",
    )
    report.add_argument(
        "--experiment-marker",
        default="brca_cell_type",
        help=(
            "With --bootstrap: marker appearing twice in the run path used to "
            "extract the model id (default: brca_cell_type)."
        ),
    )
    report.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="With --bootstrap: number of subsample repeats (default: 10).",
    )
    report.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="With --bootstrap: cells per subsample (default: 10000).",
    )
    report.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="With --bootstrap: base RNG seed (default: 42).",
    )
    report.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="With --bootstrap: EmbeddingEvaluator n_jobs (default: -1).",
    )
    report.add_argument(
        "--no-merge-results",
        action="store_true",
        help="With --bootstrap: do not update per-run results.json files.",
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

    review = subparsers.add_parser(
        "review",
        help="Record a post-run scientific decision for one execution.",
    )
    review_commands = review.add_subparsers(
        dest="review_command",
        required=True,
    )
    decide = review_commands.add_parser(
        "decide",
        help="Accept, reject, or request tuning for one completed run.",
    )
    decide.add_argument("run_dir", help="Execution output directory.")
    decide.add_argument("--decision-id", required=True)
    decide.add_argument(
        "--decision",
        required=True,
        choices=["accepted", "needs_tuning", "rejected"],
    )
    decide.add_argument("--identity", required=True)
    decide.add_argument(
        "--method",
        choices=["github_pr", "manual"],
        default="manual",
    )
    decide.add_argument("--rationale", required=True)
    decide.add_argument(
        "--include-in-reports",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Publication flag (defaults to true only for accepted).",
    )
    decide.add_argument(
        "--promote-baseline",
        action="store_true",
        help="Allowed only for accepted decisions.",
    )
    decide.add_argument(
        "--change",
        action="append",
        default=[],
        help="Tuning change (repeatable; required for needs_tuning).",
    )
    decide.add_argument("--expected-improvement")
    decide.add_argument(
        "--max-additional-budget-usd",
        type=float,
    )
    decide.add_argument("--previous-run-id")
    decide.add_argument("--attempt", type=int)
    return parser


def _add_report_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output directory. For the default results.json report: "
            "report.html, comparison.json, comparison.csv. "
            "For --collect/--bootstrap: dashboard metric CSV/JSON tables."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested result is missing or invalid.",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Publish only runs with an accepted scientific review.",
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
    accepted_only: bool,
    title: str,
) -> int:
    from scfm_cancer_eval.reporting import create_report_bundle

    bundle = create_report_bundle(
        roots,
        output_dir,
        strict=strict,
        accepted_only=accepted_only,
        title=title,
    )
    mode = "accepted-only" if accepted_only else "draft"
    print(
        f"Discovered {bundle.discovery.valid_count} valid run(s) ({mode}); "
        f"skipped {len(bundle.discovery.issues)} issue(s)."
    )
    print(f"HTML report: {bundle.html_path}")
    print(f"JSON export: {bundle.comparison.json_path}")
    print(f"CSV export: {bundle.comparison.csv_path}")
    return 0


def _run_collect_metrics(parsed: argparse.Namespace) -> int:
    from scfm_cancer_eval.reporting import create_collect_metrics_bundle

    output_dir = (
        Path(parsed.output) if parsed.output else Path(parsed.root) / "report"
    )
    print(
        "Starting metric collection "
        f"(root={parsed.root!s}, kind={parsed.kind}, output={output_dir})",
        flush=True,
    )
    print(
        "Note: the first pause is usually a recursive scan of the output tree; "
        "on large or mounted paths this can take a while.",
        flush=True,
    )
    bundle = create_collect_metrics_bundle(
        parsed.root,
        output_dir,
        kind=parsed.kind,
        folder_substring=parsed.folder_substring,
        score_col=parsed.score_col,
        keep_luad_cancer_stage=parsed.keep_luad_cancer_stage,
        include_arxiv=parsed.include_arxiv,
    )
    if bundle.embedding is not None:
        print(
            f"Embedding metrics: {bundle.embedding.row_count} row(s) -> "
            f"{bundle.embedding.csv_path} / {bundle.embedding.json_path}",
            flush=True,
        )
    if bundle.classification is not None:
        print(
            f"Classification metrics: {bundle.classification.row_count} row(s) -> "
            f"{bundle.classification.csv_path} / "
            f"{bundle.classification.json_path}",
            flush=True,
        )
    print(
        "Load CSV files in docs/classification.html or docs/embeddings.html "
        "(file picker), or place embedding.metrics.bootstrap.csv next to "
        "docs/embeddings.html after --bootstrap.",
        flush=True,
    )
    return 0


def _run_bootstrap_metrics(parsed: argparse.Namespace) -> int:
    from scfm_cancer_eval.reporting import create_bootstrap_metrics_bundle

    output_dir = (
        Path(parsed.output)
        if parsed.output
        else Path(parsed.root) / "report" / "embedding_bootstrap"
    )
    print(
        "Starting embedding bootstrap "
        f"(root={parsed.root!s}, output={output_dir}, "
        f"n_runs={parsed.n_runs}, sample_size={parsed.sample_size})",
        flush=True,
    )
    print(
        "Note: each model loads data.h5ad then runs repeated subsampled "
        "embedding evaluation; this is usually the slow step.",
        flush=True,
    )
    bundle = create_bootstrap_metrics_bundle(
        parsed.root,
        output_dir,
        folder_substring=parsed.folder_substring,
        experiment_marker=parsed.experiment_marker,
        sample_size=parsed.sample_size,
        n_runs=parsed.n_runs,
        base_seed=parsed.base_seed,
        n_jobs=parsed.n_jobs,
        merge_into_results_json=not parsed.no_merge_results,
    )
    print(
        f"Bootstrap completed for {bundle.model_count} model(s); "
        f"failures={bundle.failure_count}.",
        flush=True,
    )
    print(f"Mean CSV: {bundle.mean_csv}", flush=True)
    print(f"Std CSV: {bundle.std_csv}", flush=True)
    print(f"Median CSV: {bundle.median_csv}", flush=True)
    print(f"Bootstrap CSV: {bundle.bootstrap_csv}", flush=True)
    print(f"Bootstrap JSON: {bundle.bootstrap_json}", flush=True)
    if bundle.failed_csv is not None:
        print(f"Failed models: {bundle.failed_csv}", flush=True)
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


def _review_decide(parsed: argparse.Namespace) -> int:
    from scfm_cancer_eval.onboarding import ReviewOptions, record_review

    outcome = record_review(
        parsed.run_dir,
        ReviewOptions(
            decision_id=parsed.decision_id,
            decision=parsed.decision,
            identity=parsed.identity,
            rationale=parsed.rationale,
            method=parsed.method,
            include_in_reports=parsed.include_in_reports,
            promote_baseline=parsed.promote_baseline,
            tuning_changes=tuple(parsed.change),
            expected_improvement=parsed.expected_improvement,
            max_additional_budget_usd=parsed.max_additional_budget_usd,
            previous_run_id=parsed.previous_run_id,
            attempt=parsed.attempt,
        ),
    )
    print(f"Review decision: {outcome.decision.to_dict()['decision']}")
    print(f"Decision file: {outcome.decision_path}")
    print(f"Fingerprint: sha256:{outcome.decision.fingerprint}")
    if outcome.lineage_path is not None:
        print(f"Tuning lineage: {outcome.lineage_path}")
        print("Material changes still require a new pre-run approval PR")
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
            if parsed.collect:
                return _run_collect_metrics(parsed)
            if parsed.bootstrap:
                return _run_bootstrap_metrics(parsed)
            output_dir = (
                Path(parsed.output)
                if parsed.output
                else Path(parsed.root) / "report"
            )
            return _run_report(
                [parsed.root],
                output_dir,
                strict=parsed.strict,
                accepted_only=parsed.accepted_only,
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
                accepted_only=parsed.accepted_only,
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
        if parsed.command == "review" and parsed.review_command == "decide":
            return _review_decide(parsed)
    except ValueError as exc:
        parser.error(str(exc))

    parser.error(f"Unknown command: {parsed.command}")
    return 2
