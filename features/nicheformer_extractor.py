"""
Embedding extraction with the Nicheformer foundation model (theislab/nicheformer).

Requires the ``nicheformer`` Pixi environment (``pixi run -e nicheformer ...``).

Expected params (under ``embedding.params``):

- ``checkpoint_path``: Lightning checkpoint (``.ckpt``) for the pretrained ``Nicheformer`` module.
- ``technology_mean_path``: path to a ``.npy`` vector of per-gene scaling means for the **full**
  model vocabulary (e.g. 20,310 genes; see ``nicheformer/data/model_means/*.npy``).
- If ``adata.n_vars`` is smaller (e.g. after HVG), set **one** of:
  ``technology_gene_reference_h5ad`` (path to upstream ``model.h5ad`` — same folder as the means), or
  ``technology_gene_order_path`` (text file: one gene ID per line, same order as the ``.npy``).
  The extractor subsets the mean using the **same ID space as** ``model.h5ad`` (Ensembl). If
  ``var_names`` are **symbols**, set ``map_symbols_to_ensembl: true`` (mygene) or provide
  ``gene_id_column`` with Ensembl IDs per gene.

Optional:

- ``batch_size``: if omitted, chosen from ``context_length`` (long sequences use small batches;
  e.g. ~2 for 5k tokens to avoid CUDA OOM). Override explicitly if needed.
- ``num_workers`` (default 0),
  ``max_seq_len`` (defaults to the loaded model's ``context_length`` — must match it),
  ``aux_tokens`` (30), ``chunk_size`` (1000),
  ``embedding_layer`` (-1 = last transformer layer),
  ``use_gpu`` (default True when CUDA is available),
  ``precision`` (32 or ``"bf16-mixed"``; passed to Lightning Trainer if you extend this),
  ``split_value`` (default ``"train"``): value used for ``adata.obs["nicheformer_split"]`` when
  that column is missing (upstream ``NicheformerDataset`` filters on this column).
- ``gene_id_column``: Ensembl IDs in ``adata.var`` (when symbols are in ``var_names``).
- ``map_symbols_to_ensembl``: use mygene to map symbols → Ensembl for reference matching.
- ``mygene_species`` (default ``"human"``).
- ``drop_genes_not_in_reference`` (default ``true``): drop genes mygene could not map or that are
  absent from ``model.h5ad`` (e.g. many lncRNA symbols); set ``false`` to fail instead.
- ``model_init_params``: optional dict **or** path to JSON/YAML with ``Nicheformer`` constructor
  kwargs. If unset, ``config.yaml`` in the **same directory as the checkpoint** is loaded when
  present (BiEncoder-style training YAML is mapped to ``Nicheformer`` args). ``n_tokens`` and
  ``context_length`` are overridden from the checkpoint ``state_dict`` when possible.
- ``model_init_params_config_path``: explicit path to a YAML/JSON (overrides auto ``config.yaml``).
"""

from __future__ import annotations

import inspect
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger


def _prepare_adata_for_nicheformer(adata, split_value: str):
    """Ensure ``nicheformer_split`` exists (required by ``NicheformerDataset``)."""
    adata = adata.copy()
    if "nicheformer_split" not in adata.obs.columns:
        adata.obs["nicheformer_split"] = split_value
    return adata


def _load_full_gene_order(params: Dict[str, Any]) -> Optional[List[str]]:
    """Gene names in the same order as ``technology_mean`` (full panel)."""
    h5ad_p = params.get("technology_gene_reference_h5ad")
    txt_p = params.get("technology_gene_order_path")
    if h5ad_p and txt_p:
        raise ValueError(
            "Set only one of technology_gene_reference_h5ad or technology_gene_order_path."
        )
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
    if txt_p:
        p = Path(txt_p)
        if not p.is_file():
            raise FileNotFoundError(f"technology_gene_order_path not found: {p}")
        lines = p.read_text().splitlines()
        out = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.split(",")[0].strip())
        return out
    return None


def _ensembl_from_mygene_hit(r: Dict[str, Any]) -> Optional[str]:
    """Parse Ensembl gene ID from a mygene query hit."""
    if not isinstance(r, dict):
        return None
    if r.get("notfound"):
        return None
    if "ensembl" in r:
        e = r["ensembl"]
        if isinstance(e, list) and len(e):
            g = e[0].get("gene") if isinstance(e[0], dict) else None
            if g:
                return str(g)
        if isinstance(e, dict) and e.get("gene"):
            return str(e["gene"])
    _id = r.get("_id")
    if isinstance(_id, str) and _id.startswith("ENS"):
        return _id
    return None


def _mygene_map_symbols_to_ensembl(
    symbols: List[str], species: str, log
) -> List[Optional[str]]:
    """Map gene names to Ensembl IDs; returns ``None`` where mygene has no hit."""
    import mygene

    mg = mygene.MyGeneInfo()
    sym_to_ens: Dict[str, str] = {}

    def consume(res):
        for r in res:
            if not isinstance(r, dict):
                continue
            q = r.get("query")
            if not q:
                continue
            ens = _ensembl_from_mygene_hit(r)
            if ens and str(ens).startswith("ENS"):
                sym_to_ens.setdefault(str(q), str(ens))

    # Broad first pass (HGNC symbols, aliases, Ensembl-like accessions, lncRNA names).
    res1 = mg.querymany(
        symbols,
        scopes="symbol,alias,ensembl.gene,name,retired",
        fields="ensembl.gene,symbol",
        species=species,
        as_dataframe=False,
    )
    consume(res1)

    missing = [s for s in symbols if s not in sym_to_ens]
    if missing:
        res2 = mg.querymany(
            missing,
            scopes="ensembl.gene",
            fields="ensembl.gene",
            species=species,
            as_dataframe=False,
        )
        consume(res2)

    out: List[Optional[str]] = [sym_to_ens.get(s) for s in symbols]
    n_ok = sum(1 for x in out if x)
    log.info(
        "Nicheformer: mygene mapped %s/%s symbols to Ensembl (species=%s).",
        n_ok,
        len(symbols),
        species,
    )
    return out


def _per_column_reference_gene_ids(
    adata, params: Dict[str, Any], log
) -> List[Optional[str]]:
    """One entry per ``adata`` column: Ensembl ID or raw ``var_names`` string for lookup."""
    col = params.get("gene_id_column")
    if col is not None:
        col = str(col)
        if col not in adata.var.columns:
            raise ValueError(f"gene_id_column {col!r} not found in adata.var")
        log.info("Nicheformer: reference matching uses adata.var[%r]", col)
        out: List[Optional[str]] = []
        for v in adata.var[col].astype(str):
            v = v.strip()
            if not v or v.lower() == "nan":
                out.append(None)
            else:
                out.append(v)
        return out
    if params.get("map_symbols_to_ensembl"):
        species = str(params.get("mygene_species", "human"))
        return _mygene_map_symbols_to_ensembl(
            adata.var_names.astype(str).tolist(), species=species, log=log
        )
    return [str(x) for x in adata.var_names.astype(str)]


def _subset_adata_to_genes_in_reference(
    adata,
    col_ids: List[Optional[str]],
    reference_genes: Set[str],
    log,
    *,
    drop: bool,
) -> Tuple[Any, List[str]]:
    """Keep columns whose ID maps and is in the model reference. Returns ``(adata, filtered_ids)``."""
    mask = []
    for gid in col_ids:
        if gid is None:
            mask.append(False)
        else:
            mask.append(gid in reference_genes)
    n_keep = sum(mask)
    if n_keep == 0:
        raise ValueError(
            "No genes left after matching to the Nicheformer reference panel. "
            "Check species, ID types, or use full gene space without HVG."
        )
    if n_keep < len(mask) and drop:
        log.warning(
            "Nicheformer: dropping %s / %s genes (mygene unmapped or not in model vocabulary).",
            len(mask) - n_keep,
            len(mask),
        )
        m = np.array(mask, dtype=bool)
        adata = adata[:, m].copy()
        filt = [str(gid) for gid, ok in zip(col_ids, mask) if ok]
        return adata, filt
    if n_keep < len(mask) and not drop:
        bad = [gid for gid, ok in zip(col_ids, mask) if not ok][:25]
        raise ValueError(
            f"{len(mask) - n_keep} genes are unmapped or not in the reference panel "
            f"(examples): {bad!r}. Set drop_genes_not_in_reference: true or fix IDs."
        )
    return adata, [str(g) for g in col_ids if g is not None]


def _resolve_full_gene_order(params: Dict[str, Any], log) -> Optional[List[str]]:
    """Load gene order from explicit params, or ``model.h5ad`` beside ``technology_mean_path``."""
    order = _load_full_gene_order(params)
    if order is not None:
        return order
    mean_path = params.get("technology_mean_path")
    if not mean_path:
        return None
    sibling = Path(mean_path).resolve().parent / "model.h5ad"
    if sibling.is_file():
        log.info(
            "Nicheformer: found %s next to technology mean — using for gene order (set "
            "technology_gene_reference_h5ad to override).",
            sibling,
        )
        return _load_full_gene_order(
            {"technology_gene_reference_h5ad": str(sibling)}
        )
    return None


def _subset_technology_mean(
    technology_mean: np.ndarray,
    adata,
    full_gene_order: List[str],
    log,
    gene_ids_for_columns: Optional[List[str]] = None,
) -> np.ndarray:
    """Pick entries from the full mean vector for each column of ``adata.X`` (HVG / subset)."""
    n_full = len(full_gene_order)
    if technology_mean.shape[0] != n_full:
        raise ValueError(
            f"technology_mean length {technology_mean.shape[0]} != reference gene list length "
            f"{n_full}. Check technology_mean_path and gene reference."
        )
    order_to_idx = {g: i for i, g in enumerate(full_gene_order)}
    out = np.empty(adata.n_vars, dtype=np.float64)
    missing: List[str] = []
    for j in range(adata.n_vars):
        g = (
            str(gene_ids_for_columns[j])
            if gene_ids_for_columns is not None
            else str(adata.var_names[j])
        )
        idx = order_to_idx.get(g)
        if idx is None:
            missing.append(g)
        else:
            out[j] = float(technology_mean[idx])
    if missing:
        raise ValueError(
            f"{len(missing)} genes are not in the Nicheformer reference panel "
            f"(first 20): {missing[:20]!r}. The reference uses Ensembl IDs — set "
            f"``map_symbols_to_ensembl: true`` or ``gene_id_column``, or use Ensembl in "
            f"``adata.var_names``."
        )
    log.info(
        "Nicheformer: subset technology_mean from %s genes to n_vars=%s (matched to reference order).",
        n_full,
        adata.n_vars,
    )
    return out.astype(technology_mean.dtype, copy=False)


def _coerce_model_init_params(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(f"model_init_params path not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def _infer_nicheformer_dims_from_state_dict(
    sd: Dict[str, Any],
) -> Tuple[Optional[int], Optional[int]]:
    """Infer ``n_tokens`` and ``context_length`` from ``embeddings`` / ``positional_embedding``."""
    n_tokens = None
    context_length = None
    if not sd:
        return None, None
    token_keys = [
        k
        for k in sd
        if k.endswith("embeddings.weight") and "positional" not in k.lower()
    ]
    token_keys.sort(key=lambda x: -sd[x].shape[0])
    for k in token_keys:
        w = sd[k]
        if hasattr(w, "shape") and w.ndim == 2 and w.shape[0] > 5:
            n_tokens = int(w.shape[0]) - 5
            break
    for k, w in sd.items():
        if "positional_embedding.weight" in k and hasattr(w, "shape") and w.ndim == 2:
            context_length = int(w.shape[0])
            break
    return n_tokens, context_length


def _training_yaml_to_nicheformer_kwargs(
    cfg: Dict[str, Any], log
) -> Optional[Dict[str, Any]]:
    """Map training-repo YAML (e.g. BiEncoder + GeneToken) to ``Nicheformer.__init__`` kwargs."""
    if not isinstance(cfg, dict):
        return None
    # Flat sweep_config-style (all keys at top level).
    if (
        cfg.get("model") is None
        and "dim_model" in cfg
        and "n_tokens" in cfg
        and "nlayers" in cfg
    ):
        if log:
            log.info("Nicheformer: using flat sweep-style keys from config YAML.")
        return {k: cfg[k] for k in cfg if not k.startswith("_")}

    m = cfg.get("model")
    if not isinstance(m, dict):
        return None
    name = str(m.get("name", ""))
    if "BiEncoder" not in name and "Contrastive" not in name:
        return None

    tr = m.get("training") or {}
    dm = cfg.get("datamodule") or {}
    dl_train = (dm.get("dataloader") or {}).get("train") or {}

    kwargs: Dict[str, Any] = {
        "dim_model": int(m["dim_model"]),
        "nheads": int(m.get("num_head", 8)),
        "dim_feedforward": int(m.get("dim_hid", 1024)),
        "nlayers": int(m["nlayers"]),
        "dropout": float(m.get("dropout", 0.0)),
        "batch_first": True,
        "masking_p": float(tr.get("masking_rate", 0.15)),
        "context_length": int(m.get("pe_max_len", 5000)),
        "lr": float(tr.get("lr", 1e-4)),
        "warmup": int(tr.get("warmup", 20000)),
        "batch_size": int(dl_train.get("batch_size", 512)),
        "max_epochs": int(tr.get("max_steps", 100_000_000)),
        "n_tokens": int(m.get("n_tokens", 20340)),
        "learnable_pe": True,
        "specie": bool(m.get("specie", False)),
        "assay": bool(m.get("assay", False)),
        "modality": bool(m.get("modality", False)),
        "contrastive": "contrast" in name.lower(),
        "supervised_task": None,
    }
    if log:
        log.info(
            "Nicheformer: mapped training config (model.name=%s) to Nicheformer kwargs.",
            name,
        )
    return kwargs


def _resolve_model_init_params_for_checkpoint(
    params: Dict[str, Any], ckpt_path: str, log
) -> Optional[Dict[str, Any]]:
    """Explicit ``model_init_params``, then ``model_init_params_config_path``, then ``config.yaml``."""
    if params.get("model_init_params") is not None:
        return _coerce_model_init_params(params["model_init_params"])
    import yaml

    cfg_path = params.get("model_init_params_config_path")
    candidates: List[Path] = []
    if cfg_path:
        p = Path(str(cfg_path)).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"model_init_params_config_path not found: {p}")
        candidates.append(p)
    candidates.append(Path(ckpt_path).resolve().parent / "config.yaml")

    explicit_only = bool(cfg_path)
    for p in candidates:
        if not p.is_file():
            continue
        cfg = yaml.safe_load(p.read_text())
        if not isinstance(cfg, dict):
            continue
        kwargs = _training_yaml_to_nicheformer_kwargs(cfg, log)
        if kwargs:
            if log and p.name == "config.yaml" and not explicit_only:
                log.info(
                    "Nicheformer: loaded model_init_params from %s (beside checkpoint).",
                    p,
                )
            return kwargs
    return None


def _default_batch_size_for_context(seq_len: int) -> int:
    """Conservative batch size for transformer self-attn (memory ~ batch × seq_len²)."""
    if seq_len <= 1024:
        return 32
    if seq_len <= 2048:
        return 8
    if seq_len <= 4096:
        return 4
    return 2


def _load_nicheformer_module(
    ckpt_path: str,
    map_location: torch.device,
    *,
    model_init_params: Optional[Dict[str, Any]] = None,
    log=None,
):
    """Load ``Nicheformer`` from Lightning checkpoint, with fallback for plain ``state_dict``."""
    from nicheformer.models import Nicheformer

    ckpt_path = str(ckpt_path)
    try:
        return Nicheformer.load_from_checkpoint(ckpt_path, map_location=map_location)
    except (TypeError, KeyError, RuntimeError) as e:
        if log:
            log.info(
                "Nicheformer: Lightning load_from_checkpoint failed (%s); loading state manually.",
                e,
            )

    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected checkpoint type {type(raw)}: {ckpt_path!r}")

    sd = raw.get("state_dict") or {}
    n_inf, ctx_inf = _infer_nicheformer_dims_from_state_dict(sd)

    hp = raw.get("hyper_parameters")
    if hp is not None:
        if isinstance(hp, Namespace):
            hp = vars(hp)
        elif not isinstance(hp, dict):
            try:
                hp = dict(hp)
            except Exception:
                hp = None

    merged: Dict[str, Any] = {}
    if isinstance(hp, dict):
        merged.update(hp)
    if model_init_params:
        merged.update(model_init_params)

    if not merged:
        raise ValueError(
            "This checkpoint has no Lightning hyper_parameters and no usable config.yaml / "
            "model_init_params. Add embedding.params.model_init_params, or place config.yaml next "
            "to the checkpoint, or use a Lightning checkpoint with hyper_parameters."
        )

    if merged.get("supervised_task") is False:
        merged["supervised_task"] = None

    sig = inspect.signature(Nicheformer.__init__)
    allowed = {p for p in sig.parameters if p != "self"}
    kwargs = {k: v for k, v in merged.items() if k in allowed}

    if n_inf is not None:
        kwargs["n_tokens"] = n_inf
        if log:
            log.info("Nicheformer: set n_tokens=%s from checkpoint weights.", n_inf)
    if ctx_inf is not None:
        kwargs["context_length"] = ctx_inf
        if log:
            log.info("Nicheformer: set context_length=%s from checkpoint weights.", ctx_inf)

    missing = [
        p
        for p, param in sig.parameters.items()
        if p != "self" and param.default is inspect.Parameter.empty and p not in kwargs
    ]
    if missing:
        raise ValueError(
            f"Cannot construct Nicheformer: missing required args {missing}. "
            "Add embedding.params.model_init_params or fix config.yaml beside the checkpoint."
        )

    model = Nicheformer(**kwargs)
    if not sd:
        raise ValueError(f"No state_dict in checkpoint: {ckpt_path!r}")
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        if log:
            log.warning("Nicheformer: strict load_state_dict failed (%s); retrying strict=False.", e)
        model.load_state_dict(sd, strict=False)
    return model


class NicheformerExtractor(EmbeddingExtractor):
    """Run Nicheformer pretrained weights and write cell embeddings to ``obsm``."""

    MODELS_PATH_KEYS = frozenset(
        {
            "checkpoint_path",
            "technology_mean_path",
            "technology_gene_reference_h5ad",
            "technology_gene_order_path",
            "model_init_params_config_path",
        }
    )

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.log = get_logger()
        self.log.info("NicheformerExtractor (%s)", self.params)

    def fit_transform(self, data_loader):
        from nicheformer.data.dataset import NicheformerDataset

        required = ("checkpoint_path", "technology_mean_path")
        for k in required:
            if k not in self.params:
                raise ValueError(f"Nicheformer extractor requires embedding.params.{k}")

        ckpt = self.params["checkpoint_path"]
        tech_path = self.params["technology_mean_path"]
        technology_mean = np.load(tech_path)
        num_workers = int(self.params.get("num_workers", 0))
        aux_tokens = int(self.params.get("aux_tokens", 30))
        chunk_size = int(self.params.get("chunk_size", 1000))
        embedding_layer = int(self.params.get("embedding_layer", -1))
        split_value = str(self.params.get("split_value", "train"))
        use_gpu = bool(self.params.get("use_gpu", True))

        pl.seed_everything(int(self.params.get("seed", 42)))

        adata = _prepare_adata_for_nicheformer(data_loader.adata, split_value=split_value)
        data_loader.adata = adata

        col_ids: List[Optional[str]] = _per_column_reference_gene_ids(
            adata, self.params, self.log
        )

        if technology_mean.shape[0] != adata.n_vars:
            gene_order = _resolve_full_gene_order(self.params, self.log)
            if gene_order is None:
                mean_dir = Path(self.params["technology_mean_path"]).resolve().parent
                raise ValueError(
                    f"technology_mean length {technology_mean.shape[0]} != n_vars {adata.n_vars}. "
                    "Set `embedding.params.technology_gene_reference_h5ad` to upstream "
                    "`data/model_means/model.h5ad`, or `technology_gene_order_path` to a gene list, "
                    f"or place `model.h5ad` next to your mean file (tried {mean_dir / 'model.h5ad'})."
                )
            ref_set = set(gene_order)
            drop = bool(self.params.get("drop_genes_not_in_reference", True))
            adata, col_ids_str = _subset_adata_to_genes_in_reference(
                adata, col_ids, ref_set, self.log, drop=drop
            )
            data_loader.adata = adata
            technology_mean = _subset_technology_mean(
                technology_mean,
                adata,
                gene_order,
                self.log,
                gene_ids_for_columns=col_ids_str,
            )

        # Load model before the dataset: ``max_seq_len`` must match ``context_length`` (positional
        # table size) or ``get_embeddings`` adds token [B,T,D] + pos [1,L,D] with T != L.
        map_location = torch.device("cpu")
        mip = _resolve_model_init_params_for_checkpoint(self.params, ckpt, self.log)
        model = _load_nicheformer_module(
            ckpt,
            map_location,
            model_init_params=mip,
            log=self.log,
        )
        model.eval()
        ctx_len = int(model.hparams.context_length)
        max_seq_len = int(self.params.get("max_seq_len", ctx_len))
        if max_seq_len != ctx_len:
            self.log.warning(
                "Nicheformer: max_seq_len=%s != model context_length=%s; using %s (required for "
                "token/position alignment).",
                max_seq_len,
                ctx_len,
                ctx_len,
            )
            max_seq_len = ctx_len

        if "batch_size" in self.params:
            batch_size = int(self.params["batch_size"])
        else:
            batch_size = _default_batch_size_for_context(max_seq_len)
            self.log.info(
                "Nicheformer: batch_size=%s (default for context_length=%s; set batch_size in "
                "YAML to override, or lower it if you hit CUDA OOM).",
                batch_size,
                max_seq_len,
            )

        dataset = NicheformerDataset(
            adata=adata,
            technology_mean=technology_mean,
            split=split_value,
            max_seq_len=max_seq_len,
            aux_tokens=aux_tokens,
            chunk_size=chunk_size,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)

        embeddings: list[np.ndarray] = []
        self.log.info("Nicheformer: extracting embeddings (%s cells, device=%s)", adata.n_obs, device)

        with torch.inference_mode():
            for batch in dataloader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                emb = model.get_embeddings(batch=batch, layer=embedding_layer)
                embeddings.append(emb.detach().cpu().numpy())
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        out = np.concatenate(embeddings, axis=0)
        data_loader.adata.obsm[self.output_key] = out
        return out
