"""OpenAI web-search provider loaded only when selected."""

from __future__ import annotations

import os
from typing import Any, Mapping

from scfm_cancer_eval.onboarding.providers.base import parse_json_object


class OpenAIPlannerProvider:
    name = "openai"

    def __init__(self, *, model: str | None = None):
        self.model = (
            model
            or os.environ.get("SCFM_PLANNER_OPENAI_MODEL")
            or "gpt-4o-search-preview"
        )

    def generate(self, prompt: str) -> Mapping[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI planner requires the optional 'openai' package"
            ) from exc

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            web_search_options={},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You research software integrations and return only "
                        "valid JSON. Never invent commits, checksums, licenses, "
                        "or benchmark results."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=12000,
        )
        return parse_json_object(response.choices[0].message.content or "")
