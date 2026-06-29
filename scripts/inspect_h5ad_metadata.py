#!/usr/bin/env python3
"""Inspect AnnData (.h5ad) files: X/raw ranges, expression kind, gene IDs, shape."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

import numpy as np
import anndata as ad
import scipy.sparse as sp

try:
    from anndata._core.sparse_dataset import BaseCompressedSparseDataset
except ImportError:  # pragma: no cover
    BaseCompressedSparseDataset = type(None)  # isinstance(..., type(None)) only for None


def _is_sparse_like(X: Any) -> bool:
    """True for in-memory scipy.sparse or AnnData HDF5-backed compressed sparse."""
    if sp.issparse(X):
        return True
    return isinstance(X, BaseCompressedSparseDataset)


def _sparse_data_min_max(sub: Any) -> tuple[float, float]:
    """Min/max of sparse `.data` (works for numpy arrays and h5py-backed datasets)."""
    if sub.nnz == 0:
        return float("inf"), float("-inf")
    data = np.asarray(sub.data, dtype=np.float64)
    return float(data.min()), float(data.max())


def _matrix_min_max(X: Any, row_chunk: int = 65536) -> tuple[float, float]:
    """Min/max over a matrix or sparse matrix without densifying the full object."""
    if X is None:
        return float("nan"), float("nan")

    if _is_sparse_like(X):
        x = X
        n, m = x.shape
        if n == 0 or m == 0:
            return 0.0, 0.0
        mn, mx = float("inf"), float("-inf")
        has_any = False
        for i in range(0, n, row_chunk):
            sub = x[i : i + row_chunk]
            if sub.nnz:
                has_any = True
                dmn, dmx = _sparse_data_min_max(sub)
                mn = min(mn, dmn)
                mx = max(mx, dmx)
            if sub.nnz < sub.shape[0] * sub.shape[1]:
                mn = min(mn, 0.0)
        if not has_any:
            return 0.0, 0.0
        return mn, mx

    arr = np.asarray(X, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.min(arr)), float(np.max(arr))


def _sample_X_values(X: Any, max_nnz: int = 500_000) -> np.ndarray:
    """Draw up to max_nnz non-missing values from X for heuristics (sparse-aware)."""
    if X is None:
        return np.array([])

    if _is_sparse_like(X):
        x = X
        n = x.shape[0]
        if n == 0:
            return np.array([0.0])
        # Backed sparse has no `.data`; sample from row slices (materialized scipy matrices).
        rng = np.random.default_rng(0)
        parts: list[np.ndarray] = []
        total = 0
        row_chunk = 4096
        max_passes = 80
        for _ in range(max_passes):
            if total >= max_nnz:
                break
            r0 = int(rng.integers(0, max(1, n - 1)))
            sub = x[r0 : min(n, r0 + row_chunk)]
            if sub.nnz == 0:
                continue
            parts.append(np.asarray(sub.data, dtype=np.float64))
            total += sub.nnz
        if not parts:
            return np.array([0.0])
        data = np.concatenate(parts)
        if data.size <= max_nnz:
            return data
        idx = rng.choice(data.size, size=max_nnz, replace=False)
        return data[idx]

    flat = np.asarray(X, dtype=np.float64).ravel()
    if flat.size <= max_nnz:
        return flat
    rng = np.random.default_rng(0)
    idx = rng.choice(flat.size, size=max_nnz, replace=False)
    return flat[idx]


def _infer_expression_type(
    adata: ad.AnnData,
    x_min: float,
    x_max: float,
    path_hint: str = "",
) -> str:
    """Heuristic: raw counts vs normalized / log-scaled (not definitive)."""
    hints: list[str] = []

    lp = path_hint.lower()
    if "tpm" in lp or "fpkm" in lp or "cpm" in lp:
        hints.append("filename suggests TPM/FPKM/CPM-style normalized values")

    uns = adata.uns or {}
    for key in ("log1p", "norm_factor", "normalization", "scaled", "hvg", "seurat"):
        if key in uns:
            hints.append(f"adata.uns contains {key!r}")

    layers = list(adata.layers.keys()) if adata.layers is not None else []
    if layers:
        lower = [str(k).lower() for k in layers]
        if any("count" in k for k in lower):
            hints.append("layers include a '*count*' key (often raw counts in layers)")
        if any(k in ("logcounts", "log1p") for k in lower):
            hints.append("layers suggest log counts")

    sample = _sample_X_values(adata.X)
    if sample.size == 0:
        base = "empty X"
        return base + ("; " + "; ".join(hints) if hints else "")

    x_dtype = getattr(adata.X, "dtype", None)
    is_int_dtype = bool(
        (x_dtype is not None and np.issubdtype(x_dtype, np.integer))
        or np.issubdtype(sample.dtype, np.integer)
    )
    frac_int_like = float(np.mean(np.isclose(sample, np.round(sample)))) if sample.size else 0.0

    # Typical log1p normalized max is often < 15–20 for scRNA; raw counts can be huge.
    med = float(np.median(sample))
    p99 = float(np.percentile(sample, 99))

    parts: list[str] = []

    if is_int_dtype and frac_int_like > 0.99:
        parts.append("integer dtype with ~integer values → likely raw or rounded counts")
    elif x_max > 50 and frac_int_like > 0.85:
        parts.append("large values with mostly integer-like entries → likely counts (or rounded)")
    elif x_max <= 20 and med < 8 and frac_int_like < 0.5:
        parts.append("moderate range with mostly non-integers → likely log-normalized or scaled")
    elif 0 <= x_min and x_max <= 30 and p99 < 20:
        parts.append("bounded positive range → consistent with log1p / normalized expression")
    else:
        parts.append("ambiguous: use layers/uns/provenance to confirm")

    if hints:
        parts.append("extra hints: " + "; ".join(hints))

    return " ".join(parts)


_ENSEMBL_VAR = re.compile(
    r"^(ENSG|ENST|ENSMUSG|ENSRNOG|ENSDARG)\d{11}(\.\d+)?$", re.IGNORECASE
)


def _infer_gene_id_type(var_names: Any, sample: int = 2000) -> str:
    names = list(var_names)
    if not names:
        return "no var_names"
    rng = np.random.default_rng(0)
    k = min(sample, len(names))
    pick = rng.choice(len(names), size=k, replace=False)
    chosen = [str(names[i]) for i in pick]

    ens = sum(1 for n in chosen if _ENSEMBL_VAR.match(n))
    frac = ens / k

    # Symbols: short tokens, not starting with ENS...
    symbolish = sum(
        1
        for n in chosen
        if not n.upper().startswith("ENS")
        and len(n) <= 20
        and re.match(r"^[A-Za-z0-9._\-]+$", n)
    )

    if frac >= 0.85:
        return f"mostly Ensembl-style IDs (~{frac:.0%} of {k} sampled var_names match ENS*...)"
    if symbolish / k >= 0.85:
        return f"mostly gene symbols / short IDs (~{symbolish / k:.0%} of {k} sampled; not ENSG-prefixed)"
    return f"mixed or non-standard (~{frac:.0%} Ensembl-like in {k} sampled var_names)"


def get_loader_effective_X(adata: ad.AnnData, *, load_raw: bool, layer_name: str) -> Any:
    """
    Match ``H5ADLoader.load``: optional ``.raw`` swap, then optional ``layers[layer_name]`` as ``X``.
    """
    if load_raw:
        if adata.raw is None:
            raise ValueError("load_raw=True but adata.raw is None")
        adata = adata.raw.to_adata()
    layer = (layer_name or "X").strip()
    if layer != "X":
        if adata.layers is None or layer not in adata.layers:
            raise KeyError(f"layer_name={layer!r} not in adata.layers keys={list((adata.layers or {}).keys())}")
        return adata.layers[layer]
    return adata.X


def infer_matrix_in_x(
    path: str,
    *,
    load_raw: bool = False,
    layer_name: str = "X",
    backed: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Suggest ``dataset.matrix_in_x`` for ``yaml/constraints`` validation: one of
    ``raw_counts`` | ``log_normalized`` | ``other_processed`` | ``unspecified``.

    Uses the same effective matrix as ``H5ADLoader`` (``load_raw``, ``layer_name``).
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    size_gb = os.path.getsize(path) / (1024**3)
    if backed is None:
        backed = size_gb > 4.0

    adata = ad.read_h5ad(path, backed="r" if backed else False)
    Xeff = get_loader_effective_X(adata, load_raw=bool(load_raw), layer_name=str(layer_name or "X"))
    x_min, x_max = _matrix_min_max(Xeff)
    sample = _sample_X_values(Xeff)
    lp = os.path.basename(path).lower() + " " + path.lower()

    stats = {
        "path": path,
        "load_raw": load_raw,
        "layer_name": layer_name,
        "Xeff_min": x_min,
        "Xeff_max": x_max,
        "backed_mode": backed,
    }

    if any(k in lp for k in ("tpm", "fpkm", "cpm", "_tpm", "tpm_")):
        return "other_processed", stats

    if sample.size == 0:
        return "unspecified", stats

    frac_int_like = float(np.mean(np.isclose(sample, np.round(sample))))
    med = float(np.median(sample))
    p99 = float(np.percentile(sample, 99))

    # Raw / count-like: large dynamic range, mostly integer-valued.
    if x_max > 80 and frac_int_like > 0.92:
        return "raw_counts", stats
    if x_max >= 500 and frac_int_like > 0.85 and med > 2.0:
        return "raw_counts", stats

    # Log-normalized / scaled scRNA: bounded positive, mostly non-integers.
    if 0 <= x_min and x_max <= 22.0 and med < 12.0 and frac_int_like < 0.65:
        return "log_normalized", stats
    if x_max <= 30.0 and p99 < 18.0 and frac_int_like < 0.55:
        return "log_normalized", stats

    # TPM / scaled / other continuous processed matrices (not log1p scale, not raw counts)
    if x_max <= 85.0 and med > 0.5 and frac_int_like < 0.75 and "count" not in lp:
        return "other_processed", stats

    return "unspecified", stats


def inspect_h5ad(path: str, *, backed: bool | None = None) -> dict[str, Any]:
    """
    Load an .h5ad and summarize X, raw, expression heuristics, gene IDs, and shape.

    Parameters
    ----------
    path
        Path to .h5ad file.
    backed
        If True, open in backed (read-only) mode. If None, use backed for files > 4 GiB.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    size_gb = os.path.getsize(path) / (1024**3)
    if backed is None:
        backed = size_gb > 4.0

    adata = ad.read_h5ad(path, backed="r" if backed else False)

    x_min, x_max = _matrix_min_max(adata.X)

    raw_min = raw_max = float("nan")
    has_raw = adata.raw is not None
    if has_raw:
        raw_min, raw_max = _matrix_min_max(adata.raw.X)

    out: dict[str, Any] = {
        "path": path,
        "backed_mode": backed,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "X_min": x_min,
        "X_max": x_max,
        "has_raw": has_raw,
        "raw_X_min": raw_min,
        "raw_X_max": raw_max,
        "expression_guess": _infer_expression_type(adata, x_min, x_max, path_hint=path),
        "gene_id_guess": _infer_gene_id_type(adata.var_names),
    }
    return out


def _print_report(d: dict[str, Any]) -> None:
    print("=" * 80)
    print(d["path"])
    print(f"  cells (n_obs): {d['n_obs']:,}  genes (n_vars): {d['n_vars']:,}")
    print(f"  backed read: {d['backed_mode']}")
    print(f"  X min / max: {d['X_min']:.6g} / {d['X_max']:.6g}")
    print(f"  expression (heuristic): {d['expression_guess']}")
    if d["has_raw"]:
        print(f"  raw.X min / max: {d['raw_X_min']:.6g} / {d['raw_X_max']:.6g}")
    else:
        print("  raw: absent")
    print(f"  var_names (heuristic): {d['gene_id_guess']}")


DEFAULT_REL_PATHS = [
    "brca_full/cells_only_13gigs.h5ad",
    "brca_full/cancer_cells_only.h5ad",
    "brca_full/T_cells_only.h5ad",
    "luad_brca/luad/luad_analysis1.h5ad",
    "luad_brca/luad/luad_analysis2.h5ad",
    "melanoma/GSE120575_melanoma_tpm_annotated_final.h5ad",
    "crc/crc_tcell_tumor_MMRd_vs_MMRp_min200cells.h5ad",
    "pancancer/testing_files/combined_data_test_outer.h5ad",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="*",
        help="Paths to .h5ad files (default: bundled list under --base)",
    )
    p.add_argument(
        "--base",
        default="/home/haitham/mnt/DATA",
        help="Base directory when using default relative paths",
    )
    p.add_argument(
        "--no-backed",
        action="store_true",
        help="Always load fully in memory (can OOM large files)",
    )
    p.add_argument(
        "--force-backed",
        action="store_true",
        help="Always use backed mode",
    )
    args = p.parse_args()

    if args.paths:
        targets = [os.path.abspath(x) for x in args.paths]
    else:
        targets = [os.path.join(args.base, rel) for rel in DEFAULT_REL_PATHS]

    backed_default: bool | None
    if args.no_backed:
        backed_default = False
    elif args.force_backed:
        backed_default = True
    else:
        backed_default = None

    for path in targets:
        try:
            d = inspect_h5ad(path, backed=backed_default)
            _print_report(d)
        except Exception as exc:  # noqa: BLE001 — CLI tool: show all failures
            print("=" * 80)
            print(path)
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
