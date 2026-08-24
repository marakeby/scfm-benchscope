"""OpenAI web-search provider loaded only when selected."""

from __future__ import annotations

import os
from typing import Any, Mapping

from scfm_cancer_eval.onboarding.providers.base import parse_json_object

# Responses API + web_search (gpt-4o-search-preview is shut down 2026-07-23).
_DEFAULT_MODEL = "gpt-5.5"
_SYSTEM_INSTRUCTIONS = (
    "You research software integrations and return only valid JSON. "
    "Never invent commits, checksums, licenses, or benchmark results."
)


class OpenAIPlannerProvider:
    name = "openai"

    def __init__(self, *, model: str | None = None):
        self.model = (
            model
            or os.environ.get("SCFM_PLANNER_OPENAI_MODEL")
            or _DEFAULT_MODEL
        )

    def generate(self, prompt: str) -> Mapping[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI planner requires the optional 'openai' package"
            ) from exc

        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "high",
                }
            ],
            instructions=_SYSTEM_INSTRUCTIONS,
            input=prompt,
            max_output_tokens=16000,
        )
        return parse_json_object(getattr(response, "output_text", None) or "")
