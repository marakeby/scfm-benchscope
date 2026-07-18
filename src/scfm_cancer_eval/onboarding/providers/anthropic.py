"""Anthropic web-search provider loaded only when selected."""

from __future__ import annotations

import os
from typing import Any, Mapping

from scfm_cancer_eval.onboarding.providers.base import parse_json_object


class AnthropicPlannerProvider:
    name = "anthropic"

    def __init__(self, *, model: str | None = None):
        self.model = (
            model
            or os.environ.get("SCFM_PLANNER_ANTHROPIC_MODEL")
            or "claude-sonnet-4-20250514"
        )

    def generate(self, prompt: str) -> Mapping[str, Any]:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic planner requires the optional 'anthropic' package"
            ) from exc

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=12000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            system=(
                "You research software integrations and return only valid "
                "JSON. Never invent commits, checksums, licenses, or "
                "benchmark results."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(
            block.text
            for block in response.content
            if hasattr(block, "text") and block.text
        )
        return parse_json_object(text)
