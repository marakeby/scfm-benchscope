from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from scfm_cancer_eval.discovery import (
    catalog_model_to_candidate,
    export_candidate_records,
    safe_json_for_html,
)


def _catalog_model() -> dict:
    return {
        "model_name": "New Cell Model",
        "paper_title": "A new cell model",
        "paper_url": "https://example.org/paper",
        "github_url": "https://github.com/example/new-cell-model",
        "weights_url": "https://huggingface.co/example/new-cell-model",
        "description": "A discovered model awaiting integration planning.",
        "category": "FM",
        "architecture": ["Transformer"],
        "confidence": 0.8,
    }


class DiscoveryAgentBridgeTests(unittest.TestCase):
    def test_translates_catalog_row_without_creating_execution_details(self) -> None:
        candidate = catalog_model_to_candidate(
            _catalog_model(),
            agent="test-discovery",
            discovered_at="2026-07-17T22:00:00Z",
        )
        payload = candidate.to_dict()

        self.assertEqual(candidate.candidate_id, "new-cell-model")
        self.assertEqual(payload["discovery"]["confidence"], 0.8)
        self.assertEqual(
            payload["sources"]["repository"]["url"],
            "https://github.com/example/new-cell-model",
        )
        self.assertEqual(payload["sources"]["weights"][0]["access"], "unknown")
        self.assertIn(
            "immutable_repository_revision",
            payload["unresolved_fields"],
        )
        self.assertIn("weight_file_checksums", payload["unresolved_fields"])
        self.assertNotIn("architecture", payload)
        self.assertNotIn("commands", payload)

    def test_invalid_or_private_links_become_unresolved_evidence(self) -> None:
        model = _catalog_model()
        model["github_url"] = "https://localhost/private"
        model["weights_url"] = "http://example.org/weights"

        candidate = catalog_model_to_candidate(
            model,
            agent="test-discovery",
            discovered_at="2026-07-17T22:00:00Z",
        )
        payload = candidate.to_dict()

        self.assertIsNone(payload["sources"]["repository"])
        self.assertEqual(payload["sources"]["weights"], [])
        self.assertIn("repository_url", payload["unresolved_fields"])
        self.assertIn("weights_url", payload["unresolved_fields"])

    def test_exports_immutable_date_partitioned_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            timestamp = "2026-07-17T22:00:00Z"

            first = export_candidate_records(
                [_catalog_model()],
                tmp,
                agent="test-discovery",
                discovered_at=timestamp,
            )
            second = export_candidate_records(
                [_catalog_model()],
                tmp,
                agent="test-discovery",
                discovered_at=timestamp,
            )

            self.assertEqual(len(first.written), 1)
            self.assertEqual(first.errors, ())
            self.assertEqual(second.written, ())
            self.assertEqual(second.existing, first.written)
            self.assertEqual(first.written[0].parent.name, "2026-07-17")
            payload = json.loads(
                first.written[0].read_text(encoding="utf-8")
            )
            self.assertEqual(payload["candidate_id"], "new-cell-model")

    def test_html_json_escapes_script_terminators(self) -> None:
        encoded = safe_json_for_html(
            [{"description": "</script><script>unsafe()</script>&"}]
        )

        self.assertNotIn("</script>", encoded)
        self.assertIn("\\u003c/script\\u003e", encoded)
        self.assertIn("\\u0026", encoded)

    def test_existing_openai_agent_uses_bridge_without_api_call(self) -> None:
        agent_path = Path(__file__).resolve().parents[1] / "docs/agent.py"
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = object
        spec = importlib.util.spec_from_file_location(
            "scfm_docs_agent_test",
            agent_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)

        previous = sys.modules.get("openai")
        sys.modules["openai"] = fake_openai
        try:
            spec.loader.exec_module(module)
        finally:
            if previous is None:
                del sys.modules["openai"]
            else:
                sys.modules["openai"] = previous

        model = _catalog_model()
        model.pop("architecture")
        with contextlib.redirect_stdout(io.StringIO()):
            merged, count, added = module.merge_models([], [model])
        html = module.render_html(
            [{"model_name": "</script><script>unsafe()</script>"}],
            "now",
        )

        self.assertEqual(count, 1)
        self.assertEqual(added, merged)
        self.assertEqual(merged[0]["architecture"], [])
        self.assertNotIn("</script><script>unsafe()", html)


if __name__ == "__main__":
    unittest.main()
