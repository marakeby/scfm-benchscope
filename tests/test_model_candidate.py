from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scfm_cancer_eval import cli
from scfm_cancer_eval.onboarding import (
    MODEL_CANDIDATE_SCHEMA_NAME,
    MODEL_CANDIDATE_SCHEMA_VERSION,
    CandidateValidationError,
    ModelCandidate,
    load_model_candidate,
    model_candidate_schema,
    validate_model_candidate,
)


def _candidate_payload() -> dict:
    return {
        "schema": {
            "name": MODEL_CANDIDATE_SCHEMA_NAME,
            "version": MODEL_CANDIDATE_SCHEMA_VERSION,
        },
        "candidate_id": "new-model-v1",
        "discovered_at": "2026-07-17T21:30:00Z",
        "discovery": {
            "agent": "test-discovery",
            "source_type": "scheduled_search",
            "source_url": "https://example.org/search",
            "query": "single-cell foundation model",
            "confidence": 0.75,
        },
        "model": {
            "name": "New Model",
            "summary": "Candidate found during a test search.",
        },
        "sources": {
            "paper": {
                "url": "https://example.org/paper",
                "title": "New Model",
            },
            "repository": {
                "url": "https://github.com/example/new-model",
                "revision_hint": "main",
            },
            "weights": [
                {
                    "url": "https://huggingface.co/example/new-model",
                    "kind": "checkpoint",
                    "access": "public",
                }
            ],
        },
        "suggested_tasks": ["cell_type"],
        "unresolved_fields": ["immutable_revision", "weight_checksums"],
    }


class ModelCandidateTests(unittest.TestCase):
    def test_packaged_schema_and_example_match_runtime_contract(self) -> None:
        schema = model_candidate_schema()
        repo_root = Path(__file__).resolve().parents[1]
        example_path = (
            repo_root / "examples/models/candidates/scgpt.json"
        )

        candidate = load_model_candidate(example_path)

        self.assertEqual(
            schema["properties"]["schema"]["properties"]["name"]["const"],
            MODEL_CANDIDATE_SCHEMA_NAME,
        )
        self.assertEqual(
            schema["properties"]["schema"]["properties"]["version"]["const"],
            MODEL_CANDIDATE_SCHEMA_VERSION,
        )
        self.assertEqual(candidate.candidate_id, "scgpt")
        self.assertEqual(candidate.model_name, "scGPT")
        self.assertEqual(len(candidate.fingerprint), 64)

    def test_candidate_is_immutable_and_fingerprint_is_stable(self) -> None:
        payload = _candidate_payload()
        reordered = json.loads(json.dumps(payload, sort_keys=True))

        first = ModelCandidate.from_payload(payload)
        second = ModelCandidate.from_payload(reordered)
        exported = first.to_dict()
        exported["model"]["name"] = "Changed"

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.model_name, "New Model")
        self.assertEqual(first.to_dict()["model"]["name"], "New Model")

    def test_accepts_sparse_evidence_with_one_public_source(self) -> None:
        payload = _candidate_payload()
        payload["sources"] = {
            "paper": None,
            "repository": {"url": "https://github.com/example/new-model"},
            "weights": [],
        }
        payload["suggested_tasks"] = []

        validate_model_candidate(payload)

    def test_rejects_unsafe_unknown_and_incomplete_evidence(self) -> None:
        payload = _candidate_payload()
        payload["unexpected"] = True
        payload["discovered_at"] = "2026-07-17"
        payload["discovery"]["confidence"] = 2
        payload["sources"] = {
            "paper": None,
            "repository": {
                "url": "https://user:secret@localhost/private"
            },
            "weights": [],
        }

        with self.assertRaises(CandidateValidationError) as raised:
            validate_model_candidate(payload)

        message = str(raised.exception)
        self.assertIn("$.unexpected is not supported", message)
        self.assertIn("$.discovered_at must include a timezone", message)
        self.assertIn("$.discovery.confidence", message)
        self.assertIn("must not contain credentials", message)
        self.assertIn("must use a public hostname", message)

    def test_rejects_duplicate_tasks_and_weight_links(self) -> None:
        payload = _candidate_payload()
        payload["suggested_tasks"] = ["cell_type", "cell_type"]
        payload["sources"]["weights"].append(
            dict(payload["sources"]["weights"][0])
        )

        with self.assertRaises(CandidateValidationError) as raised:
            validate_model_candidate(payload)

        self.assertIn("$.suggested_tasks[1] duplicates", str(raised.exception))
        self.assertIn("$.sources.weights[1].url duplicates", str(raised.exception))

    def test_malformed_enum_values_report_errors_instead_of_crashing(self) -> None:
        payload = _candidate_payload()
        payload["discovery"]["source_type"] = []
        payload["sources"]["weights"][0]["kind"] = {}
        payload["sources"]["weights"][0]["access"] = []

        with self.assertRaises(CandidateValidationError) as raised:
            validate_model_candidate(payload)

        self.assertIn("$.discovery.source_type", str(raised.exception))
        self.assertIn("$.sources.weights[0].kind", str(raised.exception))
        self.assertIn("$.sources.weights[0].access", str(raised.exception))

    def test_cli_validates_candidate_and_prints_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(
                json.dumps(_candidate_payload()),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main(["candidate", "validate", str(path)])

            self.assertEqual(exit_code, 0)
            self.assertIn("Valid candidate: new-model-v1", stdout.getvalue())
            self.assertIn("Fingerprint: sha256:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
