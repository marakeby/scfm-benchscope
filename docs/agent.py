"""
scFM Inspector — Discovery Agent (OpenAI version)
==================================================
Uses chat.completions API compatible with openai>=1.0.0
"""

import json
import os
import re
import sys
import datetime
from pathlib import Path

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "gpt-4o-search-preview"   # model with built-in web search
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

CRITICAL JSON RULES:
- Return ONLY a valid JSON array, nothing else
- No markdown fences, no preamble, no explanation before or after the array
- Every string must be properly closed with a double quote
- URLs must be complete — never truncate a URL mid-string
- If unsure of a URL, use null rather than a partial URL
- Start your response with [ and end with ] with no other text

For each NEW model found, return a JSON object with these exact fields:
- model_name: short model name string
- paper_title: full paper title string
- status: "published" or "preprint"
- journal: journal name or preprint server string
- year: integer
- paper_url: full URL string or null
- github_url: full GitHub URL string or null
- weights_url: full HuggingFace/Zenodo URL string or null
- weights_size: size string like "1.2 GB" or null
- modalities: array of strings e.g. ["scRNA-seq"]
- category: one of "FM" "LLM" "Perturbation" "Spatial" "Multimodal"
- description: 2-3 sentence string with no internal double quotes
- architecture: array e.g. ["Transformer", "BERT"]
- loss_functions: array e.g. ["Masked Gene Modelling", "Contrastive"]
- domain: array e.g. ["Generic", "Multi-tissue"]
- prior_knowledge: array e.g. ["None"] or ["Gene Regulatory Network"]
- prior_knowledge_detail: string or ""
- benchmarked: false

Only include models where github_url is a real confirmed URL.
Find up to {max_models} new models.
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
    """Try to recover valid objects from a malformed JSON array."""

    # Strategy 1 — extract complete objects one by one
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
                    last_comma = blob.rfind(',', 0, len(blob)-1)
                    if last_comma > 0:
                        try:
                            obj = json.loads(blob[:last_comma] + '}')
                            objects.append(obj)
                            print(f"  Repaired: {obj.get('model_name','?')}")
                        except json.JSONDecodeError:
                            pass
                start = None

    if objects:
        print(f"  Extracted {len(objects)} objects via repair")
        return objects

    # Strategy 2 — truncate at last complete object
    last_close = raw.rfind('}')
    if last_close > 0:
        try:
            candidate = raw[:last_close+1] + ']'
            first_open = candidate.find('[')
            if first_open >= 0:
                result = json.loads(candidate[first_open:])
                if isinstance(result, list):
                    print(f"  Recovered {len(result)} via truncation strategy")
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

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            web_search_options={},
            messages=[
                {
                    "role": "system",
                    "content": "You are a scientific literature discovery agent. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=4096,
        )
        raw = response.choices[0].message.content or ""

    except Exception as e:
        print(f"  API call failed: {e}")
        print("  Trying fallback without web_search_options...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a scientific literature discovery agent. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
            print("  Fallback succeeded (no web search)")
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return []

    raw = raw.strip()
    print(f"  Raw response length: {len(raw)} chars")
    print(f"  First 100 chars: {raw[:100]}")

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    start = raw.find("[")
    end   = raw.rfind("]")

    if start == -1:
        print("  WARNING: No JSON array found in response")
        print("  Raw preview:", raw[:400])
        return []

    json_str = raw[start : end+1] if end > start else raw[start:]

    try:
        models = json.loads(json_str)
        if isinstance(models, list):
            print(f"  Parsed {len(models)} models cleanly")
            return models
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e} — attempting repair...")
        return try_repair_json(json_str)

    return []


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
    html = template.replace("__MODELS_JSON__", models_json)
    html = html.replace("__UPDATED_AT__", updated_at)
    html = html.replace("__TOTAL_COUNT__", str(len(models)))
    return html


def main():
    print("=" * 60)
    print("scFM Inspector - Discovery Agent (OpenAI)")
    print(f"Run started: {datetime.datetime.utcnow().isoformat()} UTC")
    print("=" * 60)

    print("\n[1/4] Loading database...")
    models = load_database()
    print(f"  Found {len(models)} existing models")
    print(f"  Benchmarked: {sum(1 for m in models if m.get('benchmarked'))}")

    print("\n[2/4] Searching for new models...")
    known_names = [m["model_name"] for m in models]
    new_models  = call_agent(known_names)
    print(f"  Agent returned {len(new_models)} candidates")

    print("\n[3/4] Merging results...")
    models, added = merge_models(models, new_models)
    print(f"  Added {added} new models. Total: {len(models)}")

    save_database(models)

    print("\n[4/4] Rendering index.html...")
    updated_at = datetime.datetime.utcnow().strftime("%B %d, %Y at %H:%M UTC")
    html = render_html(models, updated_at)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"  Wrote {OUTPUT_PATH} ({len(html):,} bytes)")

    print("\n" + "=" * 60)
    print("Run complete")
    print(f"  Total models:   {len(models)}")
    print(f"  Added this run: {added}")
    print(f"  Benchmarked:    {sum(1 for m in models if m.get('benchmarked'))}")
    print("=" * 60)


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    main()