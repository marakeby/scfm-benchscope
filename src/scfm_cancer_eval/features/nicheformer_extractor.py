"""
Embedding extraction with Nicheformer via the HuggingFace Transformers API.

Requires the ``nicheformer`` Pixi environment (``pixi run -e nicheformer ...``).

Model hub: https://huggingface.co/theislab/Nicheformer

Expected params (under ``embedding.params``):

- ``hf_model_id`` (default ``"theislab/Nicheformer"``): HuggingFace repo id **or** local
  directory with a snapshot (``config.json``, ``model.safetensors``, ``tokenization_nicheformer.py``).
- ``technology_mean_path``: ``.npy`` per-gene scaling means (e.g. ``dissociated_mean_script.npy``,
  length = full vocabulary, typically 20,310 genes). Passed to the HF tokenizer.
- ``gene_name_id_dict``: Geneformer symbol→Ensembl pickle; applied via ``H5ADLoader.map_ensembl``
  when ``var_names`` are symbols (same as CellPLM / scSimilarity).

Optional:

- ``trust_remote_code`` (default ``true``), ``batch_size`` (default 32),
  ``embedding_layer`` (default -1), ``use_gpu`` (default true),
  ``split_value`` (default ``"train"`` for ``nicheformer_split``),
  ``gene_id_column`` (default ``"ensembl_id"`` after mapping),
  ``align_var_names_to_ensembl`` (default ``true``): set ``var_names`` from Ensembl IDs before
  tokenization so they match the hub ``model.h5ad`` reference panel,
  ``modality`` / ``specie`` / ``assay``: obs columns or literal values for special tokens
  (defaults for dissociated human scRNA-seq: ``dissociated``, ``human``, unset assay).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from tqdm import tqdm

from scfm_cancer_eval.data.data_loader import _normalize_ensembl_stable_id
from scfm_cancer_eval.features.extractor import EmbeddingExtractor
from scfm_cancer_eval.utils.logs_ import get_logger


def _prepare_adata_for_nicheformer(adata, split_value: str):
    """Ensure ``nicheformer_split`` exists (used when filtering by split)."""
    adata = adata.copy()
    if "nicheformer_split" not in adata.obs.columns:
        adata.obs["nicheformer_split"] = split_value
    return adata


def _align_var_names_to_ensembl(adata, params: Dict[str, Any], log) -> Any:
    """Use Ensembl stable IDs as ``var_names`` so HF reference alignment matches ``model.h5ad``."""
    if not bool(params.get("align_var_names_to_ensembl", True)):
        return adata
    col = params.get("gene_id_column", "ensembl_id")
    if col is not None and str(col) in adata.var.columns:
        names = [
            _normalize_ensembl_stable_id(v) or str(v).strip()
            for v in adata.var[str(col)].astype(str)
        ]
    else:
        names = [
            _normalize_ensembl_stable_id(v) or str(v).strip()
            for v in adata.var_names.astype(str)
        ]
    if any(n is None or not str(n).startswith("ENS") for n in names):
        raise ValueError(
            "Nicheformer (HF): var_names are not Ensembl IDs and gene_id_column has no usable "
            "Ensembl values. Set embedding.params.gene_name_id_dict (Geneformer pickle) so "
            "map_ensembl runs before extraction."
        )
    out = adata.copy()
    out.var_names = [str(n) for n in names]
    out.var_names_make_unique()
    log.info("Nicheformer: aligned var_names to Ensembl (%s genes).", out.n_vars)
    return out


def _load_reference_gene_order(params: Dict[str, Any]) -> Optional[List[str]]:
    """Ensembl gene names in model vocabulary order (same length as technology_mean)."""
    h5ad_p = params.get("technology_gene_reference_h5ad")
    if h5ad_p:
        import anndata as ad

        p = Path(h5ad_p)
        if not p.is_file():
            raise FileNotFoundError(f"technology_gene_reference_h5ad not found: {p}")
        ref = ad.read_h5ad(p, backed="r")
        try:
            return ref.var_names.astype(str).tolist()
        finally:
            if getattr(ref, "file", None) is not None:
                ref.file.close()
    mean_path = params.get("technology_mean_path")
    if mean_path:
        sibling = Path(mean_path).resolve().parent / "model.h5ad"
        if sibling.is_file():
            return _load_reference_gene_order(
                {"technology_gene_reference_h5ad": str(sibling)}
            )
    return None


def _reference_ensembl_set(gene_order: List[str]) -> Set[str]:
    out: Set[str] = set()
    for g in gene_order:
        ens = _normalize_ensembl_stable_id(g)
        out.add(ens if ens is not None else str(g).strip().upper())
    return out


def _subset_adata_to_reference_genes(
    adata,
    reference_genes: Set[str],
    log,
    *,
    drop: bool = True,
) -> Any:
    """Drop genes not in the Nicheformer reference panel (avoids token ids >= vocab size)."""
    mask = []
    for g in adata.var_names.astype(str):
        key = _normalize_ensembl_stable_id(g)
        mask.append(key in reference_genes if key is not None else False)
    n_keep = sum(mask)
    if n_keep == 0:
        raise ValueError(
            "No genes left after matching to the Nicheformer reference panel. "
            "Check Ensembl mapping (gene_name_id_dict) and species."
        )
    if n_keep < len(mask) and drop:
        log.warning(
            "Nicheformer: dropping %s / %s genes not in the HF reference vocabulary.",
            len(mask) - n_keep,
            len(mask),
        )
        return adata[:, np.array(mask, dtype=bool)].copy()
    if n_keep < len(mask) and not drop:
        bad = [str(g) for g, ok in zip(adata.var_names, mask) if not ok][:20]
        raise ValueError(
            f"{len(mask) - n_keep} genes are not in the reference panel (examples): {bad!r}"
        )
    return adata


def _max_embedding_index(model) -> int:
    """Largest valid token index for Nicheformer embeddings (inclusive)."""
    # NicheformerForMaskedLM does not implement get_input_embeddings().
    nicheformer = getattr(model, "nicheformer", model)
    emb_layer = getattr(nicheformer, "embeddings", None)
    if emb_layer is not None and hasattr(emb_layer, "num_embeddings"):
        return int(emb_layer.num_embeddings) - 1
    n_tokens = int(getattr(model.config, "n_tokens", 20340))
    return n_tokens + 4


def _validate_token_ids(input_ids: torch.Tensor, max_index: int, log) -> None:
    max_tok = int(input_ids.max().item())
    if max_tok > max_index:
        raise ValueError(
            f"Nicheformer token id {max_tok} exceeds embedding table (max index {max_index}). "
            "This usually means genes outside the model vocabulary were tokenized — ensure "
            "adata is subset to the reference panel before calling the HF tokenizer."
        )
    if max_tok < 0:
        raise ValueError(f"Negative token id in input_ids: {max_tok}")


def _ensure_metadata_columns(adata, params: Dict[str, Any], log) -> Any:
    """Set optional modality / species / assay obs for HF special tokens."""
    out = adata.copy()
    defaults = {
        "modality": params.get("modality", "dissociated"),
        "specie": params.get("specie", params.get("species", "human")),
        "assay": params.get("assay"),
    }
    for col, default in defaults.items():
        if default is None:
            continue
        if col not in out.obs.columns:
            out.obs[col] = default
            log.info("Nicheformer: set obs[%r] = %r for all cells.", col, default)
    return out


def _is_local_pretrained_dir(model_id_or_path: str) -> bool:
    """True when ``model_id_or_path`` is an existing directory (not a Hub repo id)."""
    p = Path(model_id_or_path)
    return p.is_dir()


def _load_nicheformer_tokenizer_class(model_id_or_path: str):
    """Load ``NicheformerTokenizer`` from a Hub repo id or a local snapshot directory."""
    path = Path(model_id_or_path)
    if path.is_dir():
        tok_file = path / "tokenization_nicheformer.py"
        if not tok_file.is_file():
            raise FileNotFoundError(
                "Local Nicheformer directory must contain tokenization_nicheformer.py; "
                f"not found: {tok_file}"
            )
        import importlib.util
        import sys

        mod_name = "nicheformer_tokenization_local"
        spec = importlib.util.spec_from_file_location(mod_name, tok_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import NicheformerTokenizer from {tok_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module.NicheformerTokenizer

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(
        "tokenization_nicheformer.NicheformerTokenizer",
        model_id_or_path,
    )


def _load_hf_nicheformer(
    hf_model_id: str,
    technology_mean_path: str,
    *,
    trust_remote_code: bool,
    device: torch.device,
    log,
):
    """Load ``AutoModelForMaskedLM`` + ``NicheformerTokenizer`` and attach technology mean."""
    from transformers import AutoModelForMaskedLM

    local = _is_local_pretrained_dir(hf_model_id)
    source = str(Path(hf_model_id).resolve()) if local else hf_model_id
    log.info(
        "Nicheformer: loading HF model %s (local=%s, trust_remote_code=%s)",
        source,
        local,
        trust_remote_code,
    )

    tokenizer_cls = _load_nicheformer_tokenizer_class(source)
    tokenizer = tokenizer_cls.from_pretrained(
        source,
        local_files_only=local,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        source,
        trust_remote_code=trust_remote_code,
        local_files_only=local,
    )

    mean_path = Path(technology_mean_path)
    if not mean_path.is_file():
        raise FileNotFoundError(f"technology_mean_path not found: {mean_path}")
    technology_mean = np.load(mean_path)
    if hasattr(tokenizer, "_load_technology_mean"):
        tokenizer._load_technology_mean(technology_mean)
    else:
        tokenizer.technology_mean = technology_mean
        log.warning(
            "Nicheformer: tokenizer has no _load_technology_mean; set technology_mean on tokenizer."
        )

    model.eval()
    model.to(device)
    max_len = getattr(tokenizer, "max_length", None)
    log.info(
        "Nicheformer: HF model ready (device=%s, tokenizer max_length=%s, mean len=%s).",
        device,
        max_len,
        len(technology_mean),
    )
    return model, tokenizer


class NicheformerExtractor(EmbeddingExtractor):
    """Extract cell embeddings with ``theislab/Nicheformer`` on HuggingFace."""

    # hf_model_id is a Hub repo id (e.g. theislab/Nicheformer), not under MODELS_PATH.
    MODELS_PATH_KEYS = frozenset(
        {
            "technology_mean_path",
            "technology_gene_reference_h5ad",
            "gene_name_id_dict",
        }
    )

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.log = get_logger()
        self.log.info("NicheformerExtractor (%s)", self.params)

    def fit_transform(self, data_loader):
        if "technology_mean_path" not in self.params:
            raise ValueError("Nicheformer extractor requires embedding.params.technology_mean_path")

        hf_model_id = str(self.params.get("hf_model_id", "theislab/Nicheformer"))
        tech_path = self.params["technology_mean_path"]
        trust_remote_code = bool(self.params.get("trust_remote_code", True))
        batch_size = int(self.params.get("batch_size", 32))
        embedding_layer = int(self.params.get("embedding_layer", -1))
        split_value = str(self.params.get("split_value", "train"))
        use_gpu = bool(self.params.get("use_gpu", True))
        seed = int(self.params.get("seed", 42))

        torch.manual_seed(seed)
        np.random.seed(seed)

        gene_map = self.params.get("gene_name_id_dict")
        if gene_map and hasattr(data_loader, "map_ensembl"):
            self.log.info(
                "Nicheformer: mapping symbols → Ensembl via Geneformer dict (%s)",
                gene_map,
            )
            data_loader.map_ensembl(gene_map)

        adata = _prepare_adata_for_nicheformer(data_loader.adata, split_value=split_value)
        adata = _align_var_names_to_ensembl(adata, self.params, self.log)

        gene_order = _load_reference_gene_order(self.params)
        if gene_order is None:
            raise ValueError(
                "Nicheformer requires a reference gene list matching technology_mean. "
                "Set technology_gene_reference_h5ad or place model.h5ad next to the .npy file."
            )
        technology_mean = np.load(tech_path)
        if len(gene_order) != technology_mean.shape[0]:
            raise ValueError(
                f"Reference genes ({len(gene_order)}) != technology_mean length "
                f"({technology_mean.shape[0]})."
            )
        ref_set = _reference_ensembl_set(gene_order)
        drop = bool(self.params.get("drop_genes_not_in_reference", True))
        adata = _subset_adata_to_reference_genes(adata, ref_set, self.log, drop=drop)
        self.log.info(
            "Nicheformer: %s genes overlap the reference vocabulary (%s total).",
            adata.n_vars,
            len(gene_order),
        )

        adata = _ensure_metadata_columns(adata, self.params, self.log)
        data_loader.adata = adata

        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            if use_gpu:
                self.log.warning(
                    "Nicheformer: use_gpu=true but torch.cuda.is_available() is false "
                    "(CPU-only PyTorch in this env?). Reinstall with CUDA: "
                    "rm -rf .pixi/envs/nicheformer && pixi install -e nicheformer. "
                    "Verify: pixi run -e nicheformer verify-gpu"
                )

        model, tokenizer = _load_hf_nicheformer(
            hf_model_id,
            tech_path,
            trust_remote_code=trust_remote_code,
            device=device,
            log=self.log,
        )
        max_token_index = _max_embedding_index(model)

        n_obs = adata.n_obs
        embeddings: List[np.ndarray] = []
        self.log.info(
            "Nicheformer: extracting embeddings (%s cells, batch_size=%s, device=%s)",
            n_obs,
            batch_size,
            device,
        )

        with torch.inference_mode():
            for start in tqdm(range(0, n_obs, batch_size), desc="Nicheformer embed"):
                end = min(start + batch_size, n_obs)
                chunk = adata[start:end].copy()
                inputs = tokenizer(chunk)
                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                _validate_token_ids(inputs["input_ids"], max_token_index, self.log)
                emb = model.get_embeddings(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    layer=embedding_layer,
                    with_context=False,
                )
                embeddings.append(emb.detach().cpu().numpy())
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        out = np.concatenate(embeddings, axis=0)
        if out.shape[0] != adata.n_obs:
            raise RuntimeError(
                f"Embedding rows {out.shape[0]} != adata.n_obs {adata.n_obs} after HF extraction."
            )
        data_loader.adata.obsm[self.output_key] = out
        return out
