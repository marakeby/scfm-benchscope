"""
Embedding extraction with **scConcept** (theislab/scConcept).

Use the ``scconcept`` Pixi environment: ``pixi run -e scconcept ...``.

Upstream API: ``concept.scConcept.load_config_and_model`` then ``extract_embeddings`` (see
``src/concept/get_embs.py`` and ``src/concept/api.py``).

**Option A — Hugging Face hub (recommended for released weights)**

- ``model_name``: e.g. ``Corpus-30M`` (files are cached under ``cache_dir``).

**Option B — Local checkpoint + config**

- ``config_path``: YAML config (same schema as upstream ``config.yaml``).
- ``checkpoint_path``: Lightning ``.ckpt``.
- ``gene_mapping_path``: ``pc_gene_token_mapping.pkl`` (or compatible pickle).
- ``panels_dir``: optional; directory of panel CSVs if your setup requires it.

**Common**

- ``embedding_kind``: ``"mean"`` (default) or ``"cls"`` — maps to upstream
  ``mean_cell_emb`` vs ``cls_cell_emb``.
- ``batch_size``, ``max_tokens``, ``gene_sampling_strategy``, ``gene_id_column`` — passed to
  ``extract_embeddings``.

**Gene IDs (important)**

The model vocabulary is keyed by **Ensembl gene IDs** (``ENSG…``). If ``adata.var_names`` are
**gene symbols**, you must either set ``gene_id_column`` to a ``adata.var`` column that already
holds Ensembl IDs, or set ``map_symbols_to_ensembl: true`` (uses mygene; writes
``ensembl_id`` and sets ``gene_id_column`` unless you override it).
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, FrozenSet, Optional

import numpy as np

import anndata as ad

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger


def _symbols_to_ensembl_var_column(
    adata: ad.AnnData,
    symbol_column: Optional[str],
    out_column: str,
    species: str,
) -> ad.AnnData:
    """Map gene symbols to Ensembl IDs via mygene; store in ``adata.var[out_column]``."""
    import mygene

    mg = mygene.MyGeneInfo()
    if symbol_column is None:
        symbols = adata.var_names.astype(str).tolist()
    else:
        if symbol_column not in adata.var.columns:
            raise ValueError(f"symbol_column {symbol_column!r} not in adata.var")
        symbols = adata.var[symbol_column].astype(str).tolist()

    query_res = mg.querymany(
        symbols,
        scopes="symbol",
        fields="ensembl.gene",
        species=species,
    )
    mapping: Dict[str, str] = {}
    for r in query_res:
        if r.get("notfound"):
            continue
        if "ensembl" in r:
            if isinstance(r["ensembl"], list):
                mapping[r["query"]] = r["ensembl"][0]["gene"]
            else:
                mapping[r["query"]] = r["ensembl"]["gene"]

    adata.var[out_column] = [mapping.get(s) for s in symbols]
    return adata


class ScConceptExtractor(EmbeddingExtractor):
    """Load scConcept and store cell embeddings in ``adata.obsm``."""

    # Local checkpoints (optional if ``model_name`` uses Hugging Face). Omit ``cache_dir`` so
    # relative ``./cache`` stays CWD-based unless you set it explicitly under MODELS.
    MODELS_PATH_KEYS: ClassVar[FrozenSet[str]] = frozenset(
        {"config_path", "checkpoint_path", "gene_mapping_path", "panels_dir"}
    )

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.log = get_logger()
        self.log.info("ScConceptExtractor (%s)", self.params)

    def fit_transform(self, data_loader):
        from concept import scConcept

        p = self.params
        model_name = p.get("model_name")
        cache_dir = str(p.get("cache_dir", "./cache"))
        repo_id = p.get("repo_id", "theislab/scConcept")

        concept = scConcept(cfg=None, repo_id=repo_id, cache_dir=cache_dir)

        if model_name:
            concept.load_config_and_model(model_name=model_name)
        else:
            required = ("config_path", "checkpoint_path", "gene_mapping_path")
            for k in required:
                if k not in p:
                    raise ValueError(
                        f"scConcept: set embedding.params.model_name for HF loading, or provide "
                        f"{', '.join(required)} for local paths."
                    )
            panels = p.get("panels_dir")
            concept.load_config_and_model(
                config=p["config_path"],
                model_path=p["checkpoint_path"],
                gene_mapping_path=p["gene_mapping_path"],
                panels_dir=panels,
            )

        batch_size = int(p.get("batch_size", 32))
        max_tokens = p.get("max_tokens")
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        gene_sampling_strategy = p.get("gene_sampling_strategy")
        gene_id_column = p.get("gene_id_column")

        adata = data_loader.adata
        if p.get("map_symbols_to_ensembl"):
            out_col = str(p.get("ensembl_id_column", "ensembl_id"))
            species = str(p.get("mygene_species", "human"))
            sym_col = p.get("symbol_column")
            if sym_col is not None:
                sym_col = str(sym_col)
            self.log.info(
                "scConcept: mapping gene symbols to Ensembl (%s → %s)",
                sym_col or "var_names",
                out_col,
            )
            adata = _symbols_to_ensembl_var_column(
                adata, sym_col, out_col, species=species
            )
            if gene_id_column is None:
                gene_id_column = out_col

        if gene_id_column is not None and gene_id_column in adata.var.columns:
            # Subset to genes with a usable Ensembl ID (avoids all-NOT_FOUND masks).
            col = adata.var[gene_id_column]
            valid = col.notna() & col.astype(str).str.startswith("ENSG")
            if not valid.any():
                raise ValueError(
                    f"scConcept: no genes with Ensembl IDs (ENSG…) in var[{gene_id_column!r}]. "
                    "Use map_symbols_to_ensembl: true if var_names are gene symbols, or fix IDs."
                )
            n_drop = int((~valid).sum())
            if n_drop:
                self.log.info(
                    "scConcept: dropping %s genes without Ensembl IDs in %r",
                    n_drop,
                    gene_id_column,
                )
                adata = adata[:, valid].copy()

        data_loader.adata = adata

        result = concept.extract_embeddings(
            adata=adata,
            gene_id_column=gene_id_column,
            batch_size=batch_size,
            max_tokens=max_tokens,
            gene_sampling_strategy=gene_sampling_strategy,
        )

        kind = str(p.get("embedding_kind", "mean")).lower().strip()
        if kind == "cls":
            emb = np.asarray(result["cls_cell_emb"], dtype=np.float32)
        elif kind == "mean":
            emb = np.asarray(result["mean_cell_emb"], dtype=np.float32)
        else:
            raise ValueError(
                f"embedding_kind must be 'mean' or 'cls', got {kind!r}"
            )

        data_loader.adata.obsm[self.output_key] = emb

        self.log.info(
            "scConcept: wrote %s shape %s to obsm[%r]",
            kind,
            emb.shape,
            self.output_key,
        )
        return emb
