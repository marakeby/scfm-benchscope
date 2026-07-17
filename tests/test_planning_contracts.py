from __future__ import annotations

import copy
import contextlib
import io
import json
import unittest
from pathlib import Path

from scfm_cancer_eval import cli
from scfm_cancer_eval.onboarding import (
    EXECUTION_MANIFEST_SCHEMA_NAME,
    INTEGRATION_PLAN_SCHEMA_NAME,
    MODEL_SPEC_SCHEMA_NAME,
    REVIEW_DECISION_SCHEMA_NAME,
    ContractValidationError,
    ExecutionManifest,
    IntegrationPlan,
    ModelSpec,
    ReviewDecision,
    load_execution_manifest,
    load_integration_plan,
    load_model_candidate,
    load_model_spec,
    load_review_decision,
    planning_schema,
    validate_execution_manifest,
    validate_integration_plan,
    validate_model_spec,
    validate_planning_chain,
    validate_review_decision,
)


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[1] / "examples/models/planning"
)


def _payload(name: str) -> dict:
    return json.loads(
        (EXAMPLE_ROOT / name).read_text(encoding="utf-8")
    )


class PlanningContractTests(unittest.TestCase):
    def test_packaged_schemas_and_examples_match_runtime_contracts(self) -> None:
        contracts = [
            (
                "model_spec",
                MODEL_SPEC_SCHEMA_NAME,
                load_model_spec,
                "model-spec.json",
            ),
            (
                "integration_plan",
                INTEGRATION_PLAN_SCHEMA_NAME,
                load_integration_plan,
                "integration-plan.json",
            ),
            (
                "execution_manifest",
                EXECUTION_MANIFEST_SCHEMA_NAME,
                load_execution_manifest,
                "execution-manifest.json",
            ),
            (
                "review_decision",
                REVIEW_DECISION_SCHEMA_NAME,
                load_review_decision,
                "review-decision.json",
            ),
        ]

        for schema_key, schema_name, loader, example_name in contracts:
            with self.subTest(contract=schema_key):
                schema = planning_schema(schema_key)
                document = loader(EXAMPLE_ROOT / example_name)
                self.assertEqual(
                    schema["properties"]["schema"]["properties"]["name"]["const"],
                    schema_name,
                )
                self.assertEqual(len(document.fingerprint), 64)

    def test_documents_are_immutable_and_order_independent(self) -> None:
        payload = _payload("model-spec.json")
        reordered = json.loads(json.dumps(payload, sort_keys=True))

        first = ModelSpec.from_payload(payload)
        second = ModelSpec.from_payload(reordered)
        exported = first.to_dict()
        exported["model"]["name"] = "Changed"

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            first.to_dict()["model"]["name"],
            "Example Cell Model",
        )

    def test_model_spec_requires_immutable_verified_sources(self) -> None:
        payload = _payload("model-spec.json")
        payload["repository"]["commit"] = "main"
        payload["weights"][0]["sha256"] = "unknown"
        payload["weights"][0]["url"] = "https://localhost/weights.bin"

        with self.assertRaises(ContractValidationError) as raised:
            validate_model_spec(payload)

        message = str(raised.exception)
        self.assertIn("full lowercase 40-character Git commit", message)
        self.assertIn("lowercase SHA-256 digest", message)
        self.assertIn("public hostname", message)

    def test_integration_plan_rejects_unsafe_generated_paths(self) -> None:
        payload = _payload("integration-plan.json")
        payload["generated_files"][0]["path"] = "../pixi.lock"
        payload["installation"]["package_path"] = "/tmp/model"

        with self.assertRaises(ContractValidationError) as raised:
            validate_integration_plan(payload)

        self.assertIn(
            "must be a safe relative POSIX path",
            str(raised.exception),
        )

    def test_execution_manifest_enforces_order_permissions_and_budget(self) -> None:
        payload = _payload("execution-manifest.json")
        payload["steps"][0], payload["steps"][1] = (
            payload["steps"][1],
            payload["steps"][0],
        )
        payload["permissions"]["dataset_read_only"] = False
        payload["permissions"]["network_hosts"] = ["127.0.0.1"]
        payload["resources"]["max_budget_usd"] = 1
        payload["expected_outputs"] = ["../results.json"]

        with self.assertRaises(ContractValidationError) as raised:
            validate_execution_manifest(payload)

        message = str(raised.exception)
        self.assertIn("approved execution order", message)
        self.assertIn("dataset_read_only must be true", message)
        self.assertIn("private or reserved address", message)
        self.assertIn("worst-case approved cost of 4.00", message)
        self.assertIn("safe relative POSIX path", message)

    def test_execution_manifest_allows_only_bounded_retry_steps(self) -> None:
        payload = _payload("execution-manifest.json")
        payload["retry_policy"]["max_attempts"] = 4
        payload["retry_policy"]["retryable_steps"] = ["collect_results"]

        with self.assertRaises(ContractValidationError) as raised:
            validate_execution_manifest(payload)

        self.assertIn("must be at most 3", str(raised.exception))
        self.assertIn("$.retry_policy.retryable_steps[0]", str(raised.exception))

    def test_review_decision_controls_tuning_and_publication(self) -> None:
        rejected = _payload("review-decision.json")
        rejected["decision"] = "rejected"

        with self.assertRaises(ContractValidationError) as raised:
            validate_review_decision(rejected)
        self.assertIn(
            "publication must remain false",
            str(raised.exception),
        )

        tuning = copy.deepcopy(rejected)
        tuning["decision"] = "needs_tuning"
        tuning["publication"] = {
            "include_in_reports": False,
            "promote_baseline": False,
        }
        tuning["tuning"] = {
            "changes": ["Reduce the learning rate."],
            "expected_improvement": "Stabilize validation metrics.",
            "max_additional_budget_usd": 3,
        }
        tuning["run"]["attempt"] = 2
        tuning["run"]["previous_run_id"] = "example-cell-model-run-0"

        validate_review_decision(tuning)

    def test_runtime_types_are_specific_contract_classes(self) -> None:
        self.assertIsInstance(
            ModelSpec.from_payload(_payload("model-spec.json")),
            ModelSpec,
        )
        self.assertIsInstance(
            IntegrationPlan.from_payload(_payload("integration-plan.json")),
            IntegrationPlan,
        )
        self.assertIsInstance(
            ExecutionManifest.from_payload(
                _payload("execution-manifest.json")
            ),
            ExecutionManifest,
        )
        self.assertIsInstance(
            ReviewDecision.from_payload(_payload("review-decision.json")),
            ReviewDecision,
        )

    def test_cli_validates_contract_and_prints_fingerprint(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = cli.main(
                [
                    "contract",
                    "validate",
                    "execution-manifest",
                    str(EXAMPLE_ROOT / "execution-manifest.json"),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "Valid execution-manifest: example-cell-model-run-v1",
            stdout.getvalue(),
        )
        self.assertIn("Fingerprint: sha256:", stdout.getvalue())

    def test_planning_chain_binds_candidate_plan_and_manifest(self) -> None:
        candidate = load_model_candidate(
            Path(__file__).resolve().parents[1]
            / "examples/models/candidates/scgpt.json"
        )
        model_payload = _payload("model-spec.json")
        model_payload["candidate"] = {
            "candidate_id": candidate.candidate_id,
            "fingerprint": candidate.fingerprint,
        }
        model_spec = ModelSpec.from_payload(model_payload)

        plan_payload = _payload("integration-plan.json")
        plan_payload["candidate_fingerprint"] = candidate.fingerprint
        plan_payload["model_spec_fingerprint"] = model_spec.fingerprint
        integration_plan = IntegrationPlan.from_payload(plan_payload)

        manifest_payload = _payload("execution-manifest.json")
        manifest_payload["model_spec_fingerprint"] = model_spec.fingerprint
        manifest_payload[
            "integration_plan_fingerprint"
        ] = integration_plan.fingerprint
        manifest = ExecutionManifest.from_payload(manifest_payload)

        validate_planning_chain(
            candidate,
            model_spec,
            integration_plan,
            manifest,
        )

        changed_manifest = manifest.to_dict()
        changed_manifest["weights"][0]["filename"] = "weights/other.bin"
        with self.assertRaisesRegex(
            ContractValidationError,
            "weights does not match",
        ):
            validate_planning_chain(
                candidate,
                model_spec,
                integration_plan,
                ExecutionManifest.from_payload(changed_manifest),
            )


if __name__ == "__main__":
    unittest.main()
