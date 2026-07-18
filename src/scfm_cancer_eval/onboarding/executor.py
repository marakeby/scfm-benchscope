"""Execute only human-approved manifests with bounded retries."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from scfm_cancer_eval.onboarding.approval import (
    ApprovalBundle,
    verify_approval_bundle,
)
from scfm_cancer_eval.onboarding.execution_approval import (
    ExecutionApproval,
    load_execution_approval,
)
from scfm_cancer_eval.onboarding.execution_manifest import EXECUTION_STEPS
from scfm_cancer_eval.onboarding.hosts import (
    FakeHost,
    JobHost,
    LocalHost,
    SshHost,
    StepResult,
)

EXECUTION_RECORD_SCHEMA_NAME = "scfm_eval.execution_record"
EXECUTION_RECORD_SCHEMA_VERSION = "1.0.0"


class ExecutionError(ValueError):
    """Raised when an approved job cannot start or finish cleanly."""


@dataclass(frozen=True)
class ExecutionOutcome:
    root: Path
    status: str
    record_path: Path
    attempts: int
    estimated_cost_usd: float


def load_job_host(
    name: str,
    *,
    local_root: str | Path | None = None,
    ssh_host: str | None = None,
    ssh_user: str | None = None,
    ssh_remote_root: str | None = None,
    ssh_identity_file: str | None = None,
    ssh_port: int = 22,
) -> JobHost:
    if name == "fake":
        return FakeHost()
    if name == "local":
        if local_root is None:
            raise ExecutionError("--local-root is required for local transport")
        return LocalHost(Path(local_root))
    if name == "ssh":
        if not ssh_host or not ssh_user or not ssh_remote_root:
            raise ExecutionError(
                "ssh transport requires --ssh-host, --ssh-user, "
                "and --ssh-remote-root"
            )
        return SshHost(
            host=ssh_host,
            user=ssh_user,
            remote_root=ssh_remote_root,
            identity_file=ssh_identity_file,
            port=ssh_port,
        )
    raise ExecutionError(
        "transport must be one of: fake, local, ssh"
    )


def execute_approved_bundle(
    bundle_path: str | Path,
    approval_path: str | Path,
    output_dir: str | Path,
    host: JobHost,
    *,
    cleanup: bool = True,
    now: str | None = None,
) -> ExecutionOutcome:
    """Run one approved bundle. Refuses unapproved or altered inputs."""
    output = Path(output_dir)
    if output.exists():
        raise ExecutionError(f"Execution output already exists: {output}")

    bundle = verify_approval_bundle(bundle_path)
    approval = load_execution_approval(approval_path)
    _require_matching_approval(bundle, approval)

    manifest = bundle.manifest.to_dict()
    resources = manifest["resources"]
    retry_policy = manifest["retry_policy"]
    max_attempts = int(retry_policy["max_attempts"])
    retryable = set(retry_policy["retryable_steps"])
    timeout_seconds = int(float(resources["max_runtime_minutes"]) * 60)
    hourly_rate = float(resources["hourly_rate_usd"])
    gpu_count = int(resources["gpu_count"])
    max_budget = float(resources["max_budget_usd"])

    job_id = manifest["manifest_id"]
    started = time.monotonic()
    timestamp = now or datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )
    output.mkdir(parents=True)
    log_path = output / "executor.log"
    attempts: list[dict] = []
    status = "failed"
    remote_job = ""

    try:
        if isinstance(host, FakeHost) and host._root is None:
            host.bind_root(output / ".fake-host")
        remote_job = host.prepare(job_id)
        host.upload_tree(bundle.root, remote_job)
        _append_log(log_path, f"prepared job {remote_job}")

        for attempt in range(1, max_attempts + 1):
            elapsed_hours = (time.monotonic() - started) / 3600.0
            estimated = elapsed_hours * hourly_rate * gpu_count
            if estimated > max_budget:
                raise ExecutionError(
                    "execution stopped: estimated cost exceeds approved budget"
                )
            remaining = max(
                1,
                timeout_seconds - int(time.monotonic() - started),
            )
            attempt_record = _run_attempt(
                host,
                remote_job,
                bundle,
                attempt=attempt,
                timeout_seconds=remaining,
                retryable=retryable,
                log_path=log_path,
            )
            attempts.append(attempt_record)
            if attempt_record["status"] == "succeeded":
                status = "completed_unreviewed"
                break
            failed_step = attempt_record.get("failed_step")
            if failed_step not in retryable or attempt >= max_attempts:
                break
            _append_log(
                log_path,
                f"retrying after failure in {failed_step} "
                f"(attempt {attempt}/{max_attempts})",
            )

        host.download_tree(remote_job, output)
    finally:
        if cleanup and remote_job:
            try:
                host.cleanup(remote_job)
            except Exception as exc:  # pragma: no cover - best effort
                _append_log(log_path, f"cleanup failed: {exc}")

    elapsed_hours = (time.monotonic() - started) / 3600.0
    estimated_cost = round(elapsed_hours * hourly_rate * gpu_count, 6)
    record = {
        "schema": {
            "name": EXECUTION_RECORD_SCHEMA_NAME,
            "version": EXECUTION_RECORD_SCHEMA_VERSION,
        },
        "run_id": job_id,
        "created_at": timestamp,
        "manifest_fingerprint": bundle.manifest.fingerprint,
        "approval_fingerprint": approval.fingerprint,
        "status": status,
        "review_status": "completed_unreviewed"
        if status == "completed_unreviewed"
        else "not_applicable",
        "attempts": attempts,
        "resources": {
            "gpu_type": resources["gpu_type"],
            "gpu_count": gpu_count,
            "max_runtime_minutes": resources["max_runtime_minutes"],
            "hourly_rate_usd": hourly_rate,
            "max_budget_usd": max_budget,
            "estimated_cost_usd": estimated_cost,
        },
        "outputs": [
            path.name
            for path in sorted(output.iterdir())
            if path.is_file() or path.is_dir()
        ],
    }
    record_path = output / "execution-record.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if status != "completed_unreviewed":
        raise ExecutionError(
            f"approved execution failed after {len(attempts)} attempt(s); "
            f"see {record_path}"
        )
    return ExecutionOutcome(
        root=output,
        status=status,
        record_path=record_path,
        attempts=len(attempts),
        estimated_cost_usd=estimated_cost,
    )


def _require_matching_approval(
    bundle: ApprovalBundle,
    approval: ExecutionApproval,
) -> None:
    payload = approval.to_dict()
    if payload["status"] != "approved":
        raise ExecutionError("execution approval status is not approved")
    if approval.manifest_fingerprint != bundle.manifest.fingerprint:
        raise ExecutionError(
            "execution approval fingerprint does not match the bundle manifest"
        )
    request = json.loads(
        (bundle.root / "approval-request.json").read_text(encoding="utf-8")
    )
    if request["status"] != "pending_human_review":
        raise ExecutionError(
            "approval-request.json must remain the pending review artifact; "
            "use a separate execution-approval record for the merge decision"
        )


def _run_attempt(
    host: JobHost,
    remote_job: str,
    bundle: ApprovalBundle,
    *,
    attempt: int,
    timeout_seconds: int,
    retryable: set[str],
    log_path: Path,
) -> dict:
    del retryable  # enforced by the caller when deciding whether to retry
    manifest = bundle.manifest.to_dict()
    plan = bundle.integration_plan.to_dict()
    work_dir = remote_job
    bundle_dir = f"{remote_job}/bundle"
    step_results: list[dict] = []
    failed_step = None

    for step in EXECUTION_STEPS:
        command = _command_for_step(
            step,
            manifest=manifest,
            plan=plan,
            bundle_dir=bundle_dir,
            work_dir=work_dir,
        )
        _append_log(log_path, f"attempt {attempt} step {step}: {command}")
        if step == "download_weights":
            result = _download_weights_step(
                host,
                work_dir,
                manifest,
                timeout_seconds=timeout_seconds,
            )
        elif step == "collect_results":
            result = _collect_results_step(host, work_dir, manifest)
        elif step == "evaluate":
            result = host.run(
                command,
                cwd=work_dir,
                timeout_seconds=timeout_seconds,
                env={"SCFM_OUTPUT_PATH": f"{work_dir}/output"},
            )
        else:
            result = host.run(
                command,
                cwd=work_dir,
                timeout_seconds=timeout_seconds,
            )
        step_results.append(
            {
                "step": step,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "stdout_sha256": _sha256_text(result.stdout),
                "stderr_sha256": _sha256_text(result.stderr),
            }
        )
        _append_log(
            log_path,
            f"step {step} exit={result.returncode} timed_out={result.timed_out}",
        )
        if result.returncode != 0:
            failed_step = step
            return {
                "attempt": attempt,
                "status": "failed",
                "failed_step": failed_step,
                "steps": step_results,
            }
    return {
        "attempt": attempt,
        "status": "succeeded",
        "failed_step": None,
        "steps": step_results,
    }


def _command_for_step(
    step: str,
    *,
    manifest: Mapping,
    plan: Mapping,
    bundle_dir: str,
    work_dir: str,
) -> list[str]:
    repository = manifest["repository"]
    evaluation = manifest["evaluation"]
    installation = plan["installation"]
    if step == "checkout":
        return [
            "git",
            "clone",
            "--filter=blob:none",
            repository["url"],
            f"{work_dir}/src",
        ]
    if step == "create_environment":
        return [
            "pixi",
            "install",
            "--manifest-path",
            f"{bundle_dir}/pixi.toml",
            "--locked",
        ]
    if step == "install":
        package = f"{bundle_dir}/{installation['package_path']}"
        command = [
            "pixi",
            "run",
            "--manifest-path",
            f"{bundle_dir}/pixi.toml",
            "python",
            "-m",
            "pip",
            "install",
        ]
        if installation["editable"]:
            command.append("-e")
        if installation["no_deps"]:
            command.append("--no-deps")
        command.append(package)
        return command
    if step == "download_weights":
        return ["download_weights"]
    if step == "smoke_test":
        module = evaluation["adapter_module"]
        return [
            "pixi",
            "run",
            "--manifest-path",
            f"{bundle_dir}/pixi.toml",
            "python",
            "-c",
            f"import importlib; importlib.import_module({module!r})",
        ]
    if step == "evaluate":
        return [
            "scfm-eval",
            "run",
            f"{bundle_dir}/{evaluation['experiment_path']}",
        ]
    if step == "collect_results":
        return ["collect_results"]
    raise ExecutionError(f"unsupported execution step: {step}")


def _download_weights_step(
    host: JobHost,
    work_dir: str,
    manifest: Mapping,
    *,
    timeout_seconds: int,
) -> StepResult:
    allowed = set(manifest["permissions"]["network_hosts"])
    if isinstance(host, FakeHost):
        _write_fake_weights(Path(work_dir), manifest)
        return host.run(
            ["download_weights"],
            cwd=work_dir,
            timeout_seconds=timeout_seconds,
        )
    if isinstance(host, LocalHost):
        for weight in manifest["weights"]:
            destination = (
                Path(work_dir) / "weights" / Path(weight["filename"]).name
            )
            download_weight(
                weight["url"],
                destination,
                sha256=weight["sha256"],
                allowed_hosts=allowed,
            )
        return StepResult(0, "downloaded\n", "")
    if isinstance(host, SshHost):
        staging = Path(work_dir)  # remote path; download on controller first
        del staging
        import tempfile

        with tempfile.TemporaryDirectory(prefix="scfm-weights-") as tmp:
            local_root = Path(tmp)
            for weight in manifest["weights"]:
                destination = (
                    local_root / "weights" / Path(weight["filename"]).name
                )
                download_weight(
                    weight["url"],
                    destination,
                    sha256=weight["sha256"],
                    allowed_hosts=allowed,
                )
            host.upload_tree(local_root, f"{work_dir}/weights-upload")
            move = host.run(
                [
                    "cp",
                    "-a",
                    f"{work_dir}/weights-upload/bundle/weights",
                    f"{work_dir}/weights",
                ],
                cwd=work_dir,
                timeout_seconds=timeout_seconds,
            )
            return move
    raise ExecutionError("download_weights is not supported on this host")


def _collect_results_step(
    host: JobHost,
    work_dir: str,
    manifest: Mapping,
) -> StepResult:
    expected = [
        name
        for name in manifest["expected_outputs"]
        if name != "execution.log"
    ]
    if isinstance(host, FakeHost):
        output = Path(work_dir) / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "execution.log").write_text(
            "fake execution completed\n",
            encoding="utf-8",
        )
        for name in expected:
            path = output / name
            if not path.is_file():
                if name.endswith(".json"):
                    path.write_text("{}\n", encoding="utf-8")
                else:
                    path.write_text("stub\n", encoding="utf-8")
        return StepResult(0, "collected\n", "")
    if isinstance(host, LocalHost):
        output = Path(work_dir) / "output"
        missing = [name for name in expected if not (output / name).exists()]
        if missing:
            return StepResult(
                1,
                "",
                "missing expected outputs: " + ", ".join(missing),
            )
        (output / "execution.log").write_text(
            "local execution completed\n",
            encoding="utf-8",
        )
        return StepResult(0, "collected\n", "")
    # SSH: verify remote files exist without rewriting the approved plan.
    for name in expected:
        check = host.run(
            ["test", "-e", f"{work_dir}/output/{name}"],
            cwd=work_dir,
            timeout_seconds=60,
        )
        if check.returncode != 0:
            return StepResult(
                1,
                "",
                f"missing expected output: {name}",
            )
    return host.run(
        [
            "sh",
            "-c",
            f"printf 'remote execution completed\\n' > {work_dir}/output/execution.log",
        ],
        cwd=work_dir,
        timeout_seconds=60,
    )


def _write_fake_weights(work_dir: Path, manifest: Mapping) -> None:
    weights_dir = work_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    for weight in manifest["weights"]:
        path = weights_dir / Path(weight["filename"]).name
        path.write_text("fake-weight\n", encoding="utf-8")


def download_weight(
    url: str,
    destination: Path,
    *,
    sha256: str,
    allowed_hosts: set[str],
) -> None:
    """Download one weight to an allowlisted host and verify its digest."""
    host = urlsplit(url).hostname
    if host is None or host not in allowed_hosts:
        raise ExecutionError(f"weight host is not allowlisted: {host}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, method="GET")
    digest = hashlib.sha256()
    with urlopen(request, timeout=60) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        destination.unlink(missing_ok=True)
        raise ExecutionError(f"weight checksum mismatch for {destination.name}")


def _append_log(path: Path, message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
