"""Provider interface and loader for AI integration planning."""

from __future__ import annotations

import importlib
import json
import re
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class PlannerProvider(Protocol):
    """Minimal boundary required from any AI provider."""

    name: str
    model: str

    def generate(self, prompt: str) -> Mapping[str, Any]:
        """Return one JSON-compatible integration proposal."""


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract one JSON object from a provider response."""
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("AI provider response did not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("AI provider response must be a JSON object")
    return value


def load_planner_provider(
    provider: str,
    *,
    model: str | None = None,
) -> PlannerProvider:
    """Load a built-in provider or ``module:attribute`` custom provider."""
    if provider == "openai":
        from scfm_cancer_eval.onboarding.providers.openai import (
            OpenAIPlannerProvider,
        )

        return OpenAIPlannerProvider(model=model)
    if provider == "anthropic":
        from scfm_cancer_eval.onboarding.providers.anthropic import (
            AnthropicPlannerProvider,
        )

        return AnthropicPlannerProvider(model=model)
    if ":" not in provider:
        raise ValueError(
            "Unknown planner provider. Use openai, anthropic, or "
            "module:attribute."
        )

    module_name, attribute_name = provider.split(":", 1)
    attribute = getattr(importlib.import_module(module_name), attribute_name)
    instance = attribute(model=model) if model is not None else attribute()
    if not isinstance(instance, PlannerProvider):
        raise TypeError(
            f"Custom provider {provider!r} does not implement PlannerProvider"
        )
    return instance
