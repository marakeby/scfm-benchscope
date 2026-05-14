"""
Single-Cell Foundation Model Discovery Agent
=============================================
Runs on a schedule via GitHub Actions (every 3 days).

What it does each run:
  1. Loads the existing models.json database
  2. Calls Claude with web search to find new models that have both
     a public GitHub repo AND publicly available pre-trained weights
  3. Classifies each new model by architecture, loss function, domain,
     and prior knowledge type
  4. Merges new models into the database (deduplicates by model name)
  5. Renders a fresh static index.html from template.html
  6. Commits and pushes both files — GitHub Pages auto-deploys

Default display order: benchmarked models first, then by year descending.
Benchmarked flag is set manually in models.json and never overwritten by this agent.
"""

import anthropic
import json
import os
import datetime
import re
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "claude-sonnet-4-20250514"
DB_PATH        = Path("models.json")
OUTPUT_PATH    = Path("index.html")
TEMPLATE_PATH  = Path("template.html")
MAX_NEW_MODELS = 15   # cap per run to control cost

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

Search the web for recent (2023-2025) single-cell foundation models that have BOTH:
1. A public GitHub code repository
2. Publicly available pre-trained model weights (HuggingFace, Zenodo, Figshare, or similar)

Use these search angles:
{keywords}

Models already in the database - DO NOT include these:
{known_names}

For each NEW model found, return a JSON object with these exact fields:

Basic metadata:
- model_name: short model name (e.g. "scGPT")
- paper_title: full paper title
- status: "published" if in a peer-reviewed journal, "preprint" if bioRxiv/arXiv
- journal: journal name, conference, or preprint server
- year: integer publication year
- paper_url: direct URL to paper
- github_url: GitHub repository URL (must be real and verified)
- weights_url: URL to pre-trained weights (must be real HuggingFace/Zenodo/etc URL)
- weights_size: estimated size string like "1.2 GB" or null if unknown
- modalities: JSON array, e.g. ["scRNA-seq", "scATAC-seq", "Spatial"]
- category: one of "FM", "LLM", "Perturbation", "Spatial", "Multimodal"
- description: 2-3 sentences on what is architecturally or biologically novel

Classification fields:
- architecture: JSON array — use the most specific terms that apply:
    Transformer, BERT, GPT, Graph Neural Network, VAE, MLP, Hyena,
    Long Convolution, State Space Model, CNN, Diffusion, Masked Autoencoder,
    Dual Encoder, Hierarchical Attention, JEPA, Tabular, Metric Learning, LoRA/Adapter

- loss_functions: JSON array — include all that apply:
    Masked Gene Modelling, Autoregressive, Reconstruction, Contrastive,
    Multitask, Denoising, Cross-modal Alignment, Self-supervised,
    Graph Contrastive, ELBO, Triplet Loss, Supervised, In-context Learning,
    JEPA, Generative

- domain: JSON array — use Generic if broadly trained:
    Generic, Multi-tissue, Spatial / Tissue, Epigenomics, Cross-species,
    Cancer, Brain / Neuronal, Immunology, Proteomics, Perturbation,
    Regulatory, Pathology

- prior_knowledge: JSON array — what external biological knowledge is injected:
    None, Gene Regulatory Network, Protein-Protein Interaction,
    Protein Sequence / ESM, DNA Sequence, Genomic Position,
    Peak-to-Gene Links, Spatial Coordinates, Cell Ontology,
    Gene Ontology, Text / Literature, Pathway / Metabolic,
    Cross-species Orthologs, Chromatin Accessibility

- prior_knowledge_detail: one sentence explaining HOW prior knowledge is used,
  or empty string if prior_knowledge is ["None"].

- benchmarked: always false (set manually by the research team)

STRICT RULES:
- Only include models where BOTH github_url AND weights_url are real, verified URLs
- Do not hallucinate URLs - only include links you confirmed during web search
- Skip any model that lacks either a code repo or pre-trained weights
- Return ONLY a valid JSON array. No markdown, no preamble, no explanation.
- Find up to {max_models} new models.
"""


def load_database() -> list:
    if DB_PATH.exists():
        with open(DB_PATH) as f:
            return json.load(f)
    return []


def save_database(models: list) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(models, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(models)} models to {DB_PATH}")


def call_agent(known_names: list) -> list:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    keywords_str = "\n".join(f"- {kw}" for kw in SEARCH_KEYWORDS)
    known_str    = ", ".join(known_names) if known_names else "(none yet)"

    prompt = SEARCH_PROMPT_TEMPLATE.format(
        keywords=keywords_str,
        known_names=known_str,
        max_models=MAX_NEW_MODELS,
    )

    print("  Calling Claude with web search...")
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text blocks (model final answer after all tool-use rounds)
    text_parts = [
        block.text
        for block in response.content
        if hasattr(block, "text") and block.text
    ]
    raw = "\n".join(text_parts).strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Extract JSON array
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        print("  WARNING: No JSON array found in response")
        print("  Raw response preview:", raw[:400])
        return []

    try:
        models = json.loads(raw[start : end + 1])
        return models if isinstance(models, list) else []
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error: {e}")
        print("  Raw snippet:", raw[start : start + 300])
        return []


# Sensible defaults for any fields a newly discovered model might be missing
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
        if not m.get("github_url") or not m.get("weights_url"):
            print(f"  Skipping {name}: missing github_url or weights_url")
            continue

        # Apply defaults for any missing classification fields
        for field, default in FIELD_DEFAULTS.items():
            m.setdefault(field, default)

        # Safety: agent must never set benchmarked=True
        m["benchmarked"] = False

        m["added_date"] = today
        m.setdefault("source", "web-search-agent")

        existing.append(m)
        known.add(name.lower())
        added += 1
        print(f"  + Added: {name}")

    return existing, added


def render_html(models: list, updated_at: str) -> str:
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    # ensure_ascii=True keeps special chars as safe \uXXXX sequences in JS
    models_json = json.dumps(models, ensure_ascii=True)
    html = template.replace("__MODELS_JSON__", models_json)
    html = html.replace("__UPDATED_AT__", updated_at)
    html = html.replace("__TOTAL_COUNT__", str(len(models)))
    return html


def main():
    print("=" * 60)
    print("Single-Cell Foundation Model Discovery Agent")
    print(f"Run started: {datetime.datetime.utcnow().isoformat()} UTC")
    print("=" * 60)

    # 1. Load existing database
    print("\n[1/4] Loading database...")
    models = load_database()
    print(f"  Found {len(models)} existing models")
    print(f"  Benchmarked: {sum(1 for m in models if m.get('benchmarked'))}")

    # 2. Search for new models
    print("\n[2/4] Searching for new models...")
    known_names = [m["model_name"] for m in models]
    new_models  = call_agent(known_names)
    print(f"  Agent returned {len(new_models)} candidates")

    # 3. Merge
    print("\n[3/4] Merging results...")
    models, added = merge_models(models, new_models)
    print(f"  Added {added} new models. Total: {len(models)}")

    # 4. Save
    save_database(models)

    # 5. Render
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
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    main()
