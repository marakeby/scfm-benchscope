#!/usr/bin/env python3
"""Validate approval bundles and optional pull-request immutability rules."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scfm_cancer_eval.onboarding import (
    ApprovalError,
    verify_approval_bundle,
)


def _changed_paths(base: str, root: Path) -> list[tuple[str, Path]]:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            f"{base}...HEAD",
            "--",
            str(root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ApprovalError(completed.stderr.strip() or "git diff failed")
    changes = []
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise ApprovalError(f"unsupported approval change: {line}")
        changes.append((fields[0], Path(fields[1])))
    return changes


def validate(root: Path, *, base: str | None = None) -> int:
    request_dirs = {
        path.parent for path in root.rglob("approval-request.json")
    }
    manifest_dirs = {
        path.parent for path in root.rglob("execution-manifest.json")
    }
    if request_dirs != manifest_dirs:
        raise ApprovalError(
            "every approval request and execution manifest must be paired"
        )

    if base is not None:
        changes = _changed_paths(base, root)
        modified = [path for status, path in changes if status != "A"]
        if modified:
            raise ApprovalError(
                "approved bundles are immutable; create a new bundle instead "
                f"of changing {modified[0]}"
            )
        changed_dirs = {
            directory
            for _, path in changes
            for directory in request_dirs
            if path == directory or directory in path.parents
        }
        if len(changed_dirs) != 1:
            raise ApprovalError(
                "an approval pull request must add exactly one bundle"
            )

    fingerprints: set[str] = set()
    for directory in sorted(request_dirs):
        bundle = verify_approval_bundle(directory)
        fingerprint = bundle.manifest.fingerprint
        if fingerprint in fingerprints:
            raise ApprovalError(
                f"duplicate manifest fingerprint in {directory}"
            )
        fingerprints.add(fingerprint)
        print(f"Valid approval bundle: {directory}")
    return len(request_dirs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="approvals")
    parser.add_argument(
        "--base",
        help="Git base revision; enforces one new immutable bundle.",
    )
    args = parser.parse_args()
    try:
        count = validate(Path(args.root), base=args.base)
    except ValueError as exc:
        print(f"Approval validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Validated {count} approval bundle(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
