"""
STATE (arc-state) cell embedding extractor.

Uses ``state.emb.Inference`` + ``create_dataloader`` with in-memory ``AnnData`` from
``H5ADLoader``, matching the library's ``encode_adata`` concatenation of cell and
dataset embeddings.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from scfm_cancer_eval.features.extractor import EmbeddingExtractor
from scfm_cancer_eval.utils.logs_ import get_logger


class StateExtractor(EmbeddingExtractor):
    """
    Params (under ``embedding.params``):

    - ``checkpoint_path`` (str, required): PyTorch Lightning checkpoint for ``StateEmbeddingModel``.
    - ``config_path`` (str, optional): OmegaConf YAML to pass to ``Inference`` before loading
      (use when the checkpoint does not embed ``cfg_yaml`` and you must supply training config).
    - ``batch_size`` (int, optional): Overrides ``cfg.model.batch_size`` for inference.
    - ``num_workers`` (int, default 1): DataLoader workers. Kept ``>= 1`` because ``arc-state``
      sets ``persistent_workers=True`` (PyTorch requires ``num_workers > 0``).
    - ``dataset_name`` (str, default ``"scfm_eval"``): Name passed as ``adata_name`` to the dataloader.
    - ``gene_column`` (str | None, default ``"auto"``): ``"auto"`` uses STATE's overlap-based
      detection; otherwise a column name in ``adata.var`` (e.g. ``"gene_name"``), or ``None`` for index.
      After that, identifiers are aligned to ``protein_embeddings`` keys (case-insensitive symbols,
      optional Ensembl→symbol via ``gene_name_id_dict``); when needed, ``adata.var['_state_pe_gene']``
      is written with exact keys for the STATE dataloader.
    - ``concat_dataset_embedding`` (bool, default True): If True, concatenate dataset embedding
      to cell embedding along the last axis (same as ``Inference.encode_adata``).
    - ``protein_embeddings_path`` (str, optional): ``gene_symbol_to_embedding`` ``.pt`` file
      (Hugging Face ``arcinstitute/SE-600M`` ships ``protein_embeddings.pt`` next to the checkpoint).
      If omitted, ``<checkpoint_dir>/protein_embeddings.pt`` is used when that file exists.
      Required when the embedded config still points at machine-specific paths such as
      ``/data/auxillary/...`` that do not exist locally.
    - ``gene_name_id_dict`` (str, optional): Path (under ``SCFM_MODELS_PATH`` if relative) to
      Geneformer's ``gene_name_id_dict.pkl`` or its parent directory — used to map Ensembl stable
      IDs to **symbols** that match ``protein_embeddings.pt`` keys when var names / columns are
      not already HGNC-style symbols.
    """

    STATE_PE_GENE_COL = "_state_pe_gene"

    MODELS_PATH_KEYS = frozenset(
        {
            "checkpoint_path",
            "config_path",
            "protein_embeddings_path",
            "gene_name_id_dict",
        }
    )

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.log = get_logger()
        self.log.info("StateExtractor params: %s", self.params)

    @staticmethod
    def _convert_to_csr(adata):
        from scipy.sparse import csr_matrix, issparse

        if issparse(adata.X) and not isinstance(adata.X, csr_matrix):
            adata.X = csr_matrix(adata.X)
        return adata

    def _load_ensembl_symbol_map_optional(self) -> Optional[Dict[str, str]]:
        """Ensembl stable ID (upper) -> gene symbol, or ``None`` if ``gene_name_id_dict`` unset."""
        from scfm_cancer_eval.data.data_loader import _coerce_to_ensembl_symbol_map
        from scfm_cancer_eval.data.gf_loader import get_gene_name_ensembl_map

        raw = (self.params.get("gene_name_id_dict") or "").strip()
        if not raw:
            return None
        path = raw
        if os.path.isfile(path) and path.lower().endswith(".pkl"):
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, dict):
                raise TypeError(
                    f"STATE gene_name_id_dict pickle {path!r} must contain a dict, got {type(loaded).__name__}."
                )
            return _coerce_to_ensembl_symbol_map(loaded)
        if os.path.isdir(path):
            ens_map, _ = get_gene_name_ensembl_map(path)
            return _coerce_to_ensembl_symbol_map(ens_map)
        raise ValueError(
            f"STATE gene_name_id_dict: expected a .pkl file or directory with gene_name_id_dict.pkl, got {path!r}."
        )

    @staticmethod
    def _upper_to_canonical_pe_keys(protein_embeds: Dict[str, Any]) -> Dict[str, str]:
        """Map stripped upper-case strings to the exact ``protein_embeds`` dict key."""
        m: Dict[str, str] = {}
        for k in protein_embeds.keys():
            ku = str(k).upper().strip()
            m.setdefault(ku, k)
        return m

    @staticmethod
    def _row_label_candidates(adata: Any, i: int, gene_column: Optional[str]) -> List[Any]:
        if gene_column is not None:
            return [adata.var[gene_column].iloc[i]]
        labs: List[Any] = [adata.var_names[i]]
        for c in ("ensembl_id_clean", "original_gene_id", "gene_name", "gene_symbol", "symbol"):
            if c in adata.var.columns:
                labs.append(adata.var[c].iloc[i])
        return labs

    def _map_label_to_pe_key(
        self,
        label: Any,
        protein_keys: Set[str],
        upper_map: Dict[str, str],
        ens_map: Optional[Dict[str, str]],
    ) -> Optional[str]:
        from scfm_cancer_eval.data.data_loader import _normalize_ensembl_stable_id

        if label is None:
            return None
        try:
            if isinstance(label, float) and np.isnan(label):
                return None
        except TypeError:
            pass
        s = str(label).strip()
        if not s or s.lower() == "nan":
            return None
        if s in protein_keys:
            return s
        u = s.upper()
        hit = upper_map.get(u)
        if hit is not None:
            return hit
        sid = _normalize_ensembl_stable_id(s)
        if ens_map and sid:
            sym = ens_map.get(sid)
            if sym:
                s2 = str(sym).strip()
                if s2 in protein_keys:
                    return s2
                u2 = s2.upper()
                h2 = upper_map.get(u2)
                if h2 is not None:
                    return h2
        return None

    def _finalize_state_gene_column(
        self,
        adata: Any,
        protein_embeds: Any,
        gene_column: Optional[str],
    ) -> Optional[str]:
        """
        Return a ``gene_column`` value compatible with STATE's dataloader: exact keys from
        ``protein_embeds``, or ``None`` when ``adata.var_names`` already match those keys.

        When identifiers are Ensembl IDs or differ only by case, synthesizes
        ``adata.var[STATE_PE_GENE_COL]`` with canonical embedding keys.
        """
        if not isinstance(protein_embeds, dict):
            return gene_column

        protein_keys: Set[str] = set(protein_embeds.keys())
        upper_map = self._upper_to_canonical_pe_keys(protein_embeds)

        def n_direct_hits(col: Optional[str]) -> int:
            if col is None:
                ser = adata.var_names.astype(str)
            else:
                ser = adata.var[col].astype(str)
            return int(sum(1 for v in ser if str(v).strip() in protein_keys))

        if gene_column is None and n_direct_hits(None) > 0:
            return None
        if gene_column is not None and n_direct_hits(gene_column) > 0:
            return gene_column

        ens_map = self._load_ensembl_symbol_map_optional()
        n = int(adata.n_vars)
        mapped: List[str] = []
        for i in range(n):
            hit: Optional[str] = None
            for lab in self._row_label_candidates(adata, i, gene_column):
                hit = self._map_label_to_pe_key(lab, protein_keys, upper_map, ens_map)
                if hit is not None:
                    break
            mapped.append(hit if hit is not None else "")

        n_hit = sum(1 for x in mapped if x)
        if n_hit == 0:
            raise ValueError(
                "STATE gene IDs: no overlap with protein_embeddings keys after symbol case-match"
                + (" and Ensembl→symbol mapping" if ens_map else "")
                + ". Ensure var names or a column carry gene symbols or Ensembl IDs, or set "
                "embedding.params.gene_name_id_dict to Geneformer's gene_name_id_dict.pkl "
                f"(see other model YAMLs). Columns: {list(adata.var.columns)}."
            )

        adata.var[self.STATE_PE_GENE_COL] = mapped
        self.log.info(
            "STATE: wrote adata.var[%r] for embedding lookup (%s/%s genes matched to protein keys).",
            self.STATE_PE_GENE_COL,
            n_hit,
            n,
        )
        return self.STATE_PE_GENE_COL

    def _resolve_protein_embeddings(self, checkpoint: str) -> Any | None:
        """Load ESM2 gene embeddings so ``Inference.load_model`` skips cfg ``/data/...`` paths."""
        import torch

        explicit = (self.params.get("protein_embeddings_path") or "").strip()
        if explicit:
            p = Path(explicit).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(
                    f"STATE protein_embeddings_path does not exist: {explicit!r} "
                    "(download arcinstitute/SE-600M and point to protein_embeddings.pt, "
                    "or place that file next to your .ckpt)."
                )
            self.log.info("STATE: loading protein embeddings from %s", p)
            return torch.load(p, map_location="cpu", weights_only=False)

        candidate = Path(checkpoint).resolve().parent / "protein_embeddings.pt"
        if candidate.is_file():
            self.log.info("STATE: using protein embeddings next to checkpoint: %s", candidate)
            return torch.load(candidate, map_location="cpu", weights_only=False)

        self.log.warning(
            "STATE: no protein_embeddings.pt beside checkpoint (%s). "
            "If load fails with a missing /data/... path, download arcinstitute/SE-600M "
            "or set embedding.params.protein_embeddings_path.",
            candidate.parent,
        )
        return None

    def fit_transform(self, data_loader: Any) -> np.ndarray:
        from state.emb import Inference
        from state.emb.data import create_dataloader
        from state.emb.utils import get_precision_config

        import torch

        checkpoint = self.params["checkpoint_path"]
        config_path: Optional[str] = self.params.get("config_path")
        batch_size = self.params.get("batch_size")
        num_workers = int(self.params.get("num_workers", 1))
        dataset_name = self.params.get("dataset_name", "scfm_eval")
        gene_column_param = self.params.get("gene_column", "auto")
        concat_ds = bool(self.params.get("concat_dataset_embedding", True))

        cfg = OmegaConf.load(config_path) if config_path else None
        protein_embeds = self._resolve_protein_embeddings(checkpoint)
        inf = Inference(cfg=cfg, protein_embeds=protein_embeds)
        try:
            inf.load_model(checkpoint)
        except FileNotFoundError as e:
            if protein_embeds is None:
                raise FileNotFoundError(
                    "STATE could not load protein embeddings: the checkpoint config points at "
                    "paths from the original training environment (e.g. /data/auxillary/...). "
                    "Download arcinstitute/SE-600M from Hugging Face so `protein_embeddings.pt` is "
                    "beside your .ckpt, or set embedding.params.protein_embeddings_path to that file."
                ) from e
            raise

        adata = data_loader.adata
        adata = self._convert_to_csr(adata)

        dataloader_cfg = inf._vci_conf
        if batch_size is not None:
            try:
                dataloader_cfg = OmegaConf.create(OmegaConf.to_container(dataloader_cfg, resolve=True))
                if not hasattr(dataloader_cfg, "model"):
                    dataloader_cfg["model"] = {}
                dataloader_cfg.model.batch_size = int(batch_size)
            except Exception as e:
                self.log.warning("Could not override batch_size: %s", e)

        if gene_column_param == "auto":
            gene_column = inf._auto_detect_gene_column(adata)
        else:
            gene_column = gene_column_param

        gene_column = self._finalize_state_gene_column(adata, inf.protein_embeds, gene_column)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)

        workers = max(1, num_workers)

        dl = create_dataloader(
            dataloader_cfg,
            workers=workers,
            adata=adata,
            adata_name=dataset_name,
            shuffle=False,
            protein_embeds=inf.protein_embeds,
            precision=precision,
            gene_column=gene_column,
        )

        all_cell: list[np.ndarray] = []
        all_ds: list[np.ndarray] = []
        for embeddings, ds_embeddings in tqdm(
            inf.encode(dl), total=len(dl), desc="STATE encode"
        ):
            all_cell.append(embeddings.astype(np.float32))
            if ds_embeddings is not None:
                all_ds.append(ds_embeddings.astype(np.float32))

        out = np.concatenate(all_cell, axis=0)
        if concat_ds and len(all_ds) > 0:
            ds_cat = np.concatenate(all_ds, axis=0)
            out = np.concatenate([out, ds_cat], axis=-1)

        out_key = self.get_output_key()
        data_loader.adata.obsm[out_key] = out
        return out
