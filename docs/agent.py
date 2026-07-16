"""
scFM Inspector — Discovery Agent (OpenAI version)
==================================================
Runs on a schedule via GitHub Actions (every 3 days).

What it does each run:
  1. Loads the existing models.json database
  2. Calls GPT-4o with web search to find new models that have both
     a public GitHub repo AND publicly available pre-trained weights
  3. Classifies each new model by architecture, loss function, domain,
     and prior knowledge type
  4. Merges new models into the database (deduplicates by model name)
  5. Renders a fresh static index.html from template.html
  6. Commits and pushes — GitHub Pages auto-deploys

Benchmarked flag is set manually in models.json and never overwritten here.
"""

"""
scFM Inspector — Discovery Agent (OpenAI version)
==================================================
Runs on a schedule via GitHub Actions (every 3 days).
"""

import json
import os
import re
import sys
import datetime
from pathlib import Path

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "gpt-4o"
DB_PATH        = Path("models.json")
OUTPUT_PATH    = Path("index.html")
TEMPLATE_PATH  = Path("template.html")
MAX_NEW_MODELS = 15

SEARCH_KEYWORDS = [
    "single-cell foundation model pretrained weights GitHub 2025",
    "scRNA-seq transformer model HuggingFace pretrained 2025",
    "single-cell ATAC-seq epigenomics foundation model code 2025",
    "spatial transcriptomics foundation model pretrained weights",
    "cell type annotation pretrained model GitHub release",
    "genetic perturbation single-cell foundation model code weights",
    "multimodal single-cell omics foundation model HuggingFace",
    "gene expression large language model single-cell pretrained",
    "single-cell chromatin accessibility foundation model",
    "cross-species single-cell foundation model pretrained",
]

SEARCH_PROMPT_TEMPLATE = """You are a computational biology literature agent specializing in single-cell genomics.

Search the web for recent (2023-2026) single-cell foundation models that have BOTH:
1. A public GitHub code repository
2. Publicly available pre-trained model weights (HuggingFace, Zenodo, Figshare, or similar)

Use these search angles:
{keywords}

Models already in the database - DO NOT include these:
{known_names}

CRITICAL JSON RULES — failure to follow these will break the pipeline:
- Return ONLY a valid JSON array, nothing else
- No markdown fences, no preamble, no explanation
- Every string must be properly closed with a double quote
- URLs must be complete — never truncate a URL mid-string
- If you are unsure of a URL, use null rather than a partial URL
- Escape any double quotes inside strings with backslash
- Every object must have all fields listed below

For each NEW model found, return a JSON object with these exact fields:
- model_name: short model name string
- paper_title: full paper title string
- status: "published" or "preprint"
- journal: journal name or preprint server string
- year: integer
- paper_url: full URL string or null
- github_url: full GitHub URL string (must be verified real URL or null)
- weights_url: full HuggingFace/Zenodo URL string (must be verified or null)
- weights_size: size string like "1.2 GB" or null
- modalities: array of strings e.g. ["scRNA-seq"]
- category: one of "FM" "LLM" "Perturbation" "Spatial" "Multimodal"
- description: 2-3 sentence string (no internal quotes)
- architecture: array e.g. ["Transformer", "BERT"]
- loss_functions: array e.g. ["Masked Gene Modelling", "Contrastive"]
- domain: array e.g. ["Generic", "Multi-tissue"]
- prior_knowledge: array e.g. ["None"] or ["Gene Regulatory Network"]
- prior_knowledge_detail: string or ""
- benchmarked: false

Only include models where github_url is a real confirmed URL.
Find up to {max_models} new models.
Start your response with [ and end with ] with no other text.
"""

FIELD_DEFAULTS = {
    "architecture":           ["Transformer"],
    "loss_functions":         ["Self-supervised"],
    "domain":                 ["Generic"],
    "prior_knowledge":        ["None"],
    "prior_knowledge_detail": "",
    "benchmarked":            False,
    "weights_size":           None,
    "modalities":             ["scRNA-seq"],
    "category":               "FM",
}


def load_database() -> list:
    if DB_PATH.exists():
        with open(DB_PATH) as f:
            return json.load(f)
    return []


def save_database(models: list) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(models, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(models)} models to {DB_PATH}")


def try_repair_json(raw: str) -> list:
    """Try several strategies to recover a JSON array from a malformed string."""

    # Strategy 1 — direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract complete objects one by one using regex
    # Find all {...} blobs and try to parse each independently
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                blob = raw[start:i+1]
                try:
                    obj = json.loads(blob)
                    objects.append(obj)
                except json.JSONDecodeError:
                    # Try to salvage by truncating at last complete key-value pair
                    # Find last comma at depth 1
                    last_comma = blob.rfind(',', 0, len(blob)-1)
                    if last_comma > 0:
                        truncated = blob[:last_comma] + '}'
                        try:
                            obj = json.loads(truncated)
                            objects.append(obj)
                            print(f"  Repaired truncated object: {obj.get('model_name','?')}")
                        except json.JSONDecodeError:
                            print(f"  Could not repair object starting at pos {start}")
                start = None

    if objects:
        print(f"  Extracted {len(objects)} objects via repair strategy")
        return objects

    # Strategy 3 — try truncating at last complete object
    last_close = raw.rfind('}')
    if last_close > 0:
        truncated = raw[:last_close+1] + ']'
        first_open = truncated.find('[')
        if first_open >= 0:
            try:
                result = json.loads(truncated[first_open:])
                if isinstance(result, list):
                    print(f"  Recovered {len(result)} objects by truncating at last }}")
                    return result
            except json.JSONDecodeError:
                pass

    print("  All repair strategies failed")
    return []


def call_agent(known_names: list) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    keywords_str = "\n".join(f"- {kw}" for kw in SEARCH_KEYWORDS)
    known_str    = ", ".join(known_names) if known_names else "(none yet)"

    prompt = SEARCH_PROMPT_TEMPLATE.format(
        keywords=keywords_str,
        known_names=known_str,
        max_models=MAX_NEW_MODELS,
    )

    print("  Calling GPT-4o with web search...")
    response = client.responses.create(
        model=MODEL_ID,
        tools=[{"type": "web_search_preview"}],
        input=prompt,
    )

    # Extract the final text output
    raw = ""
    for block in response.output:
        if hasattr(block, "content"):
            for part in block.content:
                if hasattr(part, "text"):
                    raw += part.text

    raw = raw.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Find JSON array boundaries
    start = raw.find("[")
    end   = raw.rfind("]")

    if start == -1:
        print("  WARNING: No JSON array found in response")
        print("  Raw preview:", raw[:300])
        return []

    # Use the bounded substring if both brackets found, else everything from [
    json_str = raw[start : end+1] if end > start else raw[start:]

    # First try direct parse
    try:
        models = json.loads(json_str)
        if isinstance(models, list):
            print(f"  Parsed {len(models)} models cleanly")
            return models
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e} — attempting repair...")
        return try_repair_json(json_str)


def merge_models(existing: list, new_models: list) -> tuple:
    known = {m["model_name"].lower() for m in existing}
    today = datetime.date.today().isoformat()
    added = 0

    for m in new_models:
        name = (m.get("model_name") or "").strip()
        if not name:
            continue
        if name.lower() in known:
            continue
        if not m.get("github_url"):
            print(f"  Skipping {name}: missing github_url")
            continue

        for field, default in FIELD_DEFAULTS.items():
            m.setdefault(field, default)

        m["benchmarked"] = False
        m["added_date"]  = today
        m.setdefault("source", "web-search-agent")

        existing.append(m)
        known.add(name.lower())
        added += 1
        print(f"  + Added: {name}")

    return existing, added


def render_html(models: list, updated_at: str) -> str:
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    models_json = json.dumps(models, ensure_ascii=True)