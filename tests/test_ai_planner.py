from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scfm_cancer_eval import cli
from scfm_cancer_eval.onboarding import (
    PlannerError,
    load_integration_plan,
    load_model_candidate,
    load_model_spec,
    load_planner_provider,
    plan_candidate,
)
from scfm_cancer_eval.onboarding.providers import parse_json_object


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples/models/planning"
CANDIDATE = REPO_ROOT / "examples/models/candidates/scgpt.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ready_proposal() -> dict:
    return {
        "status": "ready",
        "unresolved_fields": [],
        "research_notes": [
            "Static proposal only; no generated code was executed."
        ],
        "model_spec": _json(EXAMPLES / "model-spec.json"),
        "integration_plan": _json(EXAMPLES / "integration-plan.json"),
        "files": [
            {
                "path": "pixi.toml",
                "purpose": "Pixi environment proposal",
                "content": (
                    '[workspace]\nname = "example-cell-model"\n'
                    'channels = ["conda-forge"]\n'
                    'platforms = ["linux-64"]\n'
                ),
            },
            {
                "path": "integrations/example_cell_model.py",
                "purpose": "Model adapter proposal",
                "content": (
                    "class ExampleCellModelAdapter:\n"
                    '    output_key = "X_example"\n'
                    "    def fit_transform(self, loader):\n"
                    "        raise NotImplementedError\n"
                ),
            },
            {
                "path": "experiments/example_cell_model.yaml",
                "purpose": "Evaluation configuration proposal",
                "content": "run_id: example-cell-model\n",
            },
        ],
    }


class FakeProvider:
    name = "fake"
    model = "fake-research-model"

    def __init__(self, proposal: dict):
        self.proposal = proposal
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        return copy.deepcopy(self.proposal)


class AiPlannerTests(unittest.TestCase):
    def test_ready_proposal_creates_valid_reviewable_workspace(self) -> None:
        candidate = load_model_candidate(CANDIDATE)
        provider = FakeProvider(_ready_proposal())

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outcome = plan_candidate(
                candidate,
                provider,
                workspace,
                created_at="2026-07-17T22:00:00Z",
            )

            model_spec = load_model_spec(outcome.model_spec_path)
            integration_plan = load_integration_plan(
                outcome.integration_plan_path
            )
            plan_payload = integration_plan.to_dict()

            self.assertEqual(outcome.status, "ready")
            self.assertEqual(len(provider.prompts), 1)
            self.assertIn(candidate.candidate_id, provider.prompts[0])
            self.assertEqual(
                model_spec.to_dict()["candidate"]["fingerprint"],
                candidate.fingerprint,
            )
            self.assertEqual(
                plan_payload["model_spec_fingerprint"],
                model_spec.fingerprint,
            )
            self.assertEqual(plan_payload["planner"]["agent"], "fake")
            self.assertTrue((workspace / "pixi.toml").is_file())
            self.assertTrue(
                (workspace / "integrations/example_cell_model.py").is_file()
            )
            self.assertFalse((workspace / "execution-manifest.json").exists())

    def test_needs_input_preserves_research_without_generating_code(self) -> None:
        candidate = load_model_candidate(CANDIDATE)
        provider = FakeProvider(
            {
                "status": "needs_input",
                "unresolved_fields": ["weight_file_checksums"],
                "research_notes": ["The weight page lists no checksum."],
                "model_spec": None,
                "integration_plan": None,
                "files": [],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outcome = plan_candidate(candidate, provider, workspace)

            self.assertEqual(outcome.status, "needs_input")
            self.assertTrue(outcome.proposal_path.is_file())
            self.assertIsNone(outcome.model_spec_path)
            self.assertEqual(outcome.generated_files, ())
            self.assertEqual(
                sorted(path.name for path in workspace.iterdir()),
                ["planning-status.json", "proposal.json"],
            )

    def test_rejects_unsafe_or_reserved_generated_paths(self) -> None:
        candidate = load_model_candidate(CANDIDATE)
        proposal = _ready_proposal()
        proposal["files"][0]["path"] = "../proposal.json"
        provider = FakeProvider(proposal)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            with self.assertRaisesRegex(PlannerError, "unsafe path"):
                plan_candidate(candidate, provider, workspace)
            self.assertFalse(workspace.exists())

    def test_rejects_ready_proposal_with_unresolved_fields(self) -> None:
        candidate = load_model_candidate(CANDIDATE)
        proposal = _ready_proposal()
        proposal["unresolved_fields"] = ["license_compatibility"]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                PlannerError,
                "ready proposals cannot have unresolved fields",
            ):
                plan_candidate(
                    candidate,
                    FakeProvider(proposal),
                    Path(tmp) / "workspace",
                )

    def test_provider_loader_supports_builtins_and_custom_classes(self) -> None:
        openai = load_planner_provider("openai", model="openai-test")
        anthropic = load_planner_provider(
            "anthropic",
            model="anthropic-test",
        )

        module = types.ModuleType("test_planner_provider")

        class CustomProvider:
            name = "custom"

            def __init__(self, *, model: str | None = None):
                self.model = model or "custom-default"

            def generate(self, prompt: str) -> dict:
                return {"prompt": prompt}

        module.CustomProvider = CustomProvider
        sys.modules[module.__name__] = module
        try:
            custom = load_planner_provider(
                "test_planner_provider:CustomProvider",
                model="custom-test",
            )
        finally:
            del sys.modules[module.__name__]

        self.assertEqual(openai.model, "openai-test")
        self.assertEqual(anthropic.model, "anthropic-test")
        self.assertEqual(custom.name, "custom")
        self.assertEqual(custom.model, "custom-test")

    def test_provider_json_parser_handles_fenced_output(self) -> None:
        parsed = parse_json_object(
            '```json\n{"status": "needs_input"}\n```'
        )

        self.assertEqual(parsed, {"status": "needs_input"})

    def test_cli_uses_selected_provider_without_knowing_its_sdk(self) -> None:
        provider = FakeProvider(
            {
                "status": "needs_input",
                "unresolved_fields": ["immutable_repository_revision"],
                "research_notes": [],
                "model_spec": None,
                "integration_plan": None,
                "files": [],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "workspace"
            stdout = io.StringIO()
            with patch(
                "scfm_cancer_eval.onboarding.load_planner_provider",
                return_value=provider,
            ), contextlib.redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "plan",
                        str(CANDIDATE),
                        "--provider",
                        "future-provider",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Planning status: needs_input", stdout.getvalue())
            self.assertTrue((output / "proposal.json").is_file())


if __name__ == "__main__":
    unittest.main()
