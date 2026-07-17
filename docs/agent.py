"""
scFM Inspector — Discovery Agent (OpenAI version)
==================================================
Runs on a schedule via GitHub Actions (every 3 days).
- Adds all found models regardless of missing GitHub/weights URLs
- Actively searches for GitHub repos and weights separately
- Never skips a model just because URLs are missing
"""

import json
import os
import re
import sys
import datetime
from pathlib import Path

from openai import OpenAI
from scfm_cancer_eval.discovery import (
    export_candidate_records,
    safe_json_for_html,
)

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_ROOT      = Path(__file__).resolve().parent
MODEL_ID       = "gpt-4o-search-preview"
DB_PATH        = DOCS_ROOT / "models.json"
OUTPUT_PATH    = DOCS_ROOT / "models.html"
TEMPLATE_PATH  = DOCS_ROOT / "template.html"
CANDIDATE_PATH = DOCS_ROOT / "candidates"
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
    "single-cell chromatin accessibility foundation model 2025",
    "cross-species single-cell foundation model pretrained 2025",
]

DISCOVERY_PROMPT = """You are a computational biology literature agent specializing in single-cell genomics.

Search the web for recent (2023-2026) single-cell foundation models.
Include ALL models you find, even if you cannot confirm a GitHub repo or weights URL.

Use these search angles:
{keywords}

Models already in the database - DO NOT include these:
{known_names}

For each model found, actively search for:
1. The paper URL (arXiv, bioRxiv, Nature, Cell, PubMed, etc.)
2. The GitHub repository — search GitHub directly: site:github.com <model_name> single-cell
3. Pre-trained weights — search HuggingFace: site:huggingface.co <model_name>
4. Weights on Zenodo, Figshare, or the paper's supplementary data

CRITICAL JSON RULES:
- Return ONLY a valid JSON array, nothing else
- No markdown fences, no preamble, no explanation
- Never truncate a URL mid-string — use null if unsure
- Start with [ and end with ] with no other text

For each model return a JSON object with these exact fields:
- model_name: short model name string
- paper_title: full paper title string
- status: "published" or "preprint"
- journal: journal name or preprint server string
- year: integer
- paper_url: full URL string or null
- github_url: full GitHub URL string or null (search GitHub even if not in paper)
- weights_url: full HuggingFace/Zenodo URL or null (search even if not in paper)
- weights_size: size string like "1.2 GB" or null
- has_github: true if you found a GitHub repo, false otherwise
- has_weights: true if you found pre-trained weights, false otherwise
- confidence: number from 0 to 1 reflecting confidence in the source links
- modalities: array of strings e.g. ["scRNA-seq"]
- category: one of "FM" "LLM" "Perturbation" "Spatial" "Multimodal"
- description: 2-3 sentence string with no internal double quotes
- architecture: array e.g. ["Transformer", "BERT"]
- loss_functions: array e.g. ["Masked Gene Modelling", "Contrastive"]
- domain: array e.g. ["Generic", "Multi-tissue"]
- prior_knowledge: array e.g. ["None"] or ["Gene Regulatory Network"]
- prior_knowledge_detail: string or ""
- benchmarked: false

Find up to {max_models} new models. Include models even if github_url or weights_url is null.
"""

URL_SEARCH_PROMPT = """Search the web to find the GitHub repository and pre-trained model weights for the following single-cell foundation model:

Model name: {model_name}
Paper title: {paper_title}
Paper URL: {paper_url}

Please search:
1. GitHub: site:github.com {model_name} single-cell RNA
2. HuggingFace: site:huggingface.co {model_name}
3. Zenodo or Figshare for model weights

Return ONLY a JSON object with these fields (no markdown, no explanation):
{{
  "github_url": "full GitHub URL or null",
  "weights_url": "full HuggingFace/Zenodo URL or null",
  "weights_size": "size string or null",
  "has_github": true or false,
  "has_weights": true or false
}}
"""

FIELD_DEFAULTS = {
    "architecture":           [],
    "loss_functions":         [],
    "domain":                 [],
    "prior_knowledge":        [],
    "prior_knowledge_detail": "",
    "benchmarked":            False,
    "weights_size":           None,
    "modalities":             ["scRNA-seq"],
    "category":               "FM",
    "has_github":             False,
    "has_weights":            False,
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
    objects = []
    depth, start = 0, None
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

    # Truncate at last complete object
    last_close = raw.rfind('}')
    if last_close > 0:
        try:
            candidate = raw[:last_close+1] + ']'
            first_open = candidate.find('[')
            if first_open >= 0:
                result = json.loads(candidate[first_open:])
                if isinstance(result, list):
                    print(f"  Recovered {len(result)} via truncation")
                    return result
        except json.JSONDecodeError:
            pass

    print("  All repair strategies failed")
    return []


def chat_call(client: OpenAI, prompt: str, system: str = "You are a scientific literature discovery agent. Always respond with valid JSON only.") -> str:
    """Make a chat completion call with fallback."""
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            web_search_options={},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"  Web search model failed ({e}), trying gpt-4o fallback...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=4096,
            )
            return response.choices[0].message.content or ""
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return ""


def search_urls_for_model(client: OpenAI, model: dict) -> dict:
    """Separately search for GitHub and weights URLs for a model that's missing them."""
    if model.get("github_url") and model.get("weights_url"):
        return model  # already has both, skip

    print(f"  Searching URLs for: {model['model_name']}")
    prompt = URL_SEARCH_PROMPT.format(
        model_name=model["model_name"],
        paper_title=model.get("paper_title", ""),
        paper_url=model.get("paper_url", "unknown"),
    )

    raw = chat_call(client, prompt)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return model

    try:
        result = json.loads(raw[start:end])
        # Only update if we found something new
        if result.get("github_url") and not model.get("github_url"):
            model["github_url"]  = result["github_url"]
            model["has_github"]  = True
            print(f"    Found GitHub: {result['github_url']}")
        if result.get("weights_url") and not model.get("weights_url"):
            model["weights_url"] = result["weights_url"]
            model["has_weights"] = True
            if result.get("weights_size"):
                model["weights_size"] = result["weights_size"]
            print(f"    Found weights: {result['weights_url']}")
    except json.JSONDecodeError:
        pass

    return model


def call_agent(known_names: list) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    keywords_str = "\n".join(f"- {kw}" for kw in SEARCH_KEYWORDS)
    known_str    = ", ".join(known_names) if known_names else "(none yet)"

    prompt = DISCOVERY_PROMPT.format(
        keywords=keywords_str,
        known_names=known_str,
        max_models=MAX_NEW_MODELS,
    )

    print("  Calling GPT-4o with web search (discovery)...")
    raw = chat_call(client, prompt)
    raw = raw.strip()
    print(f"  Raw response: {len(raw)} chars, first 120: {raw[:120]}")

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1:
        print("  WARNING: No JSON array found")
        return []

    json_str = raw[start : end+1] if end > start else raw[start:]

    try:
        models = json.loads(json_str)
        if isinstance(models, list):
            print(f"  Parsed {len(models)} models cleanly")
        else:
            models = []
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e} — attempting repair...")
        models = try_repair_json(json_str)

    # For each model missing URLs, do a targeted search
    enriched = []
    for m in models:
        if not m.get("github_url") or not m.get("weights_url"):
            m = search_urls_for_model(client, m)
        # Set has_github / has_weights flags
        m["has_github"]  = bool(m.get("github_url"))
        m["has_weights"] = bool(m.get("weights_url"))
        enriched.append(m)

    return enriched


def merge_models(existing: list, new_models: list) -> tuple:
    known = {m["model_name"].lower() for m in existing}
    today = datetime.date.today().isoformat()
    added = 0
    added_models = []

    for m in new_models:
        name = (m.get("model_name") or "").strip()
        if not name:
            continue
        if name.lower() in known:
            print(f"  Already exists: {name}")
            continue

        # Apply defaults for missing fields
        for field, default in FIELD_DEFAULTS.items():
            m.setdefault(field, default)

        # Ensure flags are consistent with URLs
        m["has_github"]  = bool(m.get("github_url"))
        m["has_weights"] = bool(m.get("weights_url"))
        m["benchmarked"] = False
        m["added_date"]  = today
        m.setdefault("source", "web-search-agent")

        existing.append(m)
        added_models.append(m)
        known.add(name.lower())
        added += 1

        gh  = "✓ GitHub"  if m["has_github"]  else "✗ no GitHub"
        w   = "✓ weights" if m["has_weights"] else "✗ no weights"
        print(f"  + Added: {name} [{gh}] [{w}]")

    return existing, added, added_models


def render_html(models: list, updated_at: str) -> str:
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    models_json = safe_json_for_html(models)
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
    print(f"  Benchmarked:  {sum(1 for m in models if m.get('benchmarked'))}")
    print(f"  Has GitHub:   {sum(1 for m in models if m.get('has_github') or m.get('github_url'))}")
    print(f"  Has weights:  {sum(1 for m in models if m.get('has_weights') or m.get('weights_url'))}")

    print("\n[2/4] Searching for new models...")
    known_names = [m["model_name"] for m in models]
    new_models  = call_agent(known_names)
    print(f"  Agent returned {len(new_models)} candidates")

    print("\n[3/4] Merging results...")
    models, added, added_models = merge_models(models, new_models)
    print(f"  Added {added} new models. Total: {len(models)}")

    save_database(models)

    candidate_export = export_candidate_records(
        added_models,
        CANDIDATE_PATH,
        agent="openai-discovery",
    )
    print(
        "  Candidate records: "
        f"{len(candidate_export.written)} new, "
        f"{len(candidate_export.existing)} existing, "
        f"{len(candidate_export.errors)} invalid"
    )
    for error in candidate_export.errors:
        print(f"  Candidate warning: {error}")

    print("\n[4/4] Rendering models.html...")
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