"""Job hosts that run approved execution steps without changing the plan."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol


@dataclass(frozen=True)
class StepResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class JobHost(Protocol):
    """Transport used by the executor; never invents steps or secrets."""

    def prepare(self, job_id: str) -> str:
        """Create an isolated job directory and return its path."""

    def upload_tree(self, local_dir: Path, remote_dir: str) -> None:
        """Copy a local tree into the job directory."""

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        timeout_seconds: int,
        env: Mapping[str, str] | None = None,
    ) -> StepResult:
        """Run one argv list. Free-form shell strings are not accepted."""

    def download_tree(self, remote_dir: str, local_dir: Path) -> None:
        """Copy selected remote outputs back to the local result directory."""

    def cleanup(self, remote_dir: str) -> None:
        """Remove the job directory when the caller requests cleanup."""


@dataclass
class FakeHost:
    """In-memory host for dry runs and unit tests."""

    commands: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    jobs: dict[str, Path] = field(default_factory=dict)
    fail_step_prefix: str | None = None
    _root: Path | None = None

    def prepare(self, job_id: str) -> str:
        if self._root is None:
            raise RuntimeError("FakeHost requires bind_root() in tests")
        job = self._root / "jobs" / job_id
        job.mkdir(parents=True, exist_ok=False)
        self.jobs[job_id] = job
        return str(job)

    def bind_root(self, root: Path) -> None:
        self._root = root

    def upload_tree(self, local_dir: Path, remote_dir: str) -> None:
        destination = Path(remote_dir) / "bundle"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(local_dir, destination)

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        timeout_seconds: int,
        env: Mapping[str, str] | None = None,
    ) -> StepResult:
        del timeout_seconds, env
        self.commands.append((cwd, tuple(command)))
        label = command[0] if command else ""
        if self.fail_step_prefix and label.startswith(self.fail_step_prefix):
            return StepResult(1, "", f"forced failure for {label}")
        # Create expected result stubs when the evaluate step marker runs.
        if command[:2] == ["scfm-eval", "run"]:
            output = Path(cwd) / "output"
            output.mkdir(parents=True, exist_ok=True)
            (output / "results.json").write_text(
                '{"schema":{"name":"stub","version":"0"},"status":"ok"}\n',
                encoding="utf-8",
            )
            (output / "resolved_config.yaml").write_text(
                "run_id: stub\n",
                encoding="utf-8",
            )
        return StepResult(0, "ok\n", "")

    def download_tree(self, remote_dir: str, local_dir: Path) -> None:
        _copy_job_outputs(Path(remote_dir), local_dir)

    def cleanup(self, remote_dir: str) -> None:
        path = Path(remote_dir)
        if path.exists():
            shutil.rmtree(path)


@dataclass(frozen=True)
class LocalHost:
    """Run approved steps in a local job directory."""

    root: Path

    def prepare(self, job_id: str) -> str:
        job = self.root / "jobs" / job_id
        job.mkdir(parents=True, exist_ok=False)
        return str(job)

    def upload_tree(self, local_dir: Path, remote_dir: str) -> None:
        destination = Path(remote_dir) / "bundle"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(local_dir, destination)

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        timeout_seconds: int,
        env: Mapping[str, str] | None = None,
    ) -> StepResult:
        full_env = os.environ.copy()
        if env is not None:
            full_env.update(env)
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=full_env,
            )
        except subprocess.TimeoutExpired as exc:
            return StepResult(
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "timed out",
                timed_out=True,
            )
        return StepResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )

    def download_tree(self, remote_dir: str, local_dir: Path) -> None:
        _copy_job_outputs(Path(remote_dir), local_dir)

    def cleanup(self, remote_dir: str) -> None:
        path = Path(remote_dir)
        if path.exists():
            shutil.rmtree(path)


@dataclass(frozen=True)
class SshHost:
    """Copy a job to a GPU VM and run argv lists over SSH."""

    host: str
    user: str
    remote_root: str
    identity_file: str | None = None
    port: int = 22

    def prepare(self, job_id: str) -> str:
        remote = f"{self.remote_root.rstrip('/')}/{job_id}"
        self._ssh(["mkdir", "-p", remote])
        return remote

    def upload_tree(self, local_dir: Path, remote_dir: str) -> None:
        destination = f"{remote_dir}/bundle"
        self._ssh(["mkdir", "-p", destination])
        command = self._scp_base() + [
            "-r",
            f"{local_dir}/.",
            f"{self.user}@{self.host}:{destination}/",
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"scp upload failed: {completed.stderr.strip()}"
            )

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        timeout_seconds: int,
        env: Mapping[str, str] | None = None,
    ) -> StepResult:
        exports = ""
        if env:
            exports = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in env.items()
            ) + " "
        remote = (
            f"cd {shlex.quote(cwd)} && {exports}"
            + " ".join(shlex.quote(part) for part in command)
        )
        try:
            completed = subprocess.run(
                self._ssh_base() + [remote],
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return StepResult(
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "timed out",
                timed_out=True,
            )
        return StepResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )

    def download_tree(self, remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for name in ("output", "execution.log"):
            command = self._scp_base() + [
                "-r",
                f"{self.user}@{self.host}:{remote_dir}/{name}",
                str(local_dir),
            ]
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0 and name == "execution.log":
                continue
            if completed.returncode != 0:
                raise RuntimeError(
                    f"scp download failed for {name}: "
                    f"{completed.stderr.strip()}"
                )

    def cleanup(self, remote_dir: str) -> None:
        self._ssh(["rm", "-rf", remote_dir])

    def _ssh(self, command: list[str]) -> None:
        completed = subprocess.run(
            self._ssh_base() + [" ".join(shlex.quote(part) for part in command)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"ssh failed: {completed.stderr.strip()}")

    def _ssh_base(self) -> list[str]:
        command = [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        if self.identity_file:
            command.extend(["-i", self.identity_file])
        command.append(f"{self.user}@{self.host}")
        return command

    def _scp_base(self) -> list[str]:
        command = [
            "scp",
            "-P",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        if self.identity_file:
            command.extend(["-i", self.identity_file])
        return command


def _copy_job_outputs(source: Path, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in ("output", "execution.log"):
        path = source / name
        if path.is_dir():
            target = local_dir / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copyfile(path, local_dir / name)
