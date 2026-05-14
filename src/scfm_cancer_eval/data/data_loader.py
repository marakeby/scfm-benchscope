"""
data_loader.py

Data loading utilities for single-cell data, supporting CSV and H5AD formats with preprocessing and filtering.
"""

import os
import re
import json
import logging
import pickle

import pandas as pd
import anndata as ad
import scanpy as sc

from scfm_cancer_eval.utils.logs_ import get_logger
from scfm_cancer_eval.setup_path import BASE_PATH
import numpy as np

# GENCODE / Cell Ranger-style names, e.g. ENSG00000237491.8_4 → ENSG00000237491
_ENSEMBL_IN_VAR = re.compile(r"(ENS[A-Z]{0,10}G\d{6,})", re.IGNORECASE)


def _stable_ensembl_from_var_label(label) -> str | None:
    """Extract Ensembl gene stable ID from var names that embed ENSG… (with version / suffix)."""
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    m = _ENSEMBL_IN_VAR.search(s)
    if not m:
        return None
    return m.group(1).upper()


def _normalize_ensembl_stable_id(label) -> str | None:
    """Map a dict key or ``var`` label to Ensembl stable ID (no version), or ``None``."""
    if label is None:
        return None
    parsed = _stable_ensembl_from_var_label(label)
    if parsed is not None:
        return parsed
    s = str(label).strip().upper()
    if s.startswith("ENSG") and len(s) >= 15:
        return s.split(".", 1)[0]
    return None


def _coerce_to_ensembl_symbol_map(raw: dict) -> dict[str, str]:
    """
    Build ``{ensembl_stable_upper: gene_symbol}`` from either orientation.

    If the sample looks like Geneformer-style **symbol → Ensembl**, the dict is inverted
    (last symbol wins on duplicate Ensembl keys; a warning is logged elsewhere).
    """
    sample = list(raw.items())[: min(50, len(raw))]
    if not sample:
        return {}

    def row_looks_ensembl(x) -> bool:
        return _normalize_ensembl_stable_id(x) is not None

    n_key_ens = sum(1 for k, _ in sample if row_looks_ensembl(k))
    n_val_ens = sum(1 for _, v in sample if row_looks_ensembl(v))
    invert = n_val_ens > n_key_ens and n_val_ens >= max(3, len(sample) // 2)

    out: dict[str, str] = {}
    collisions = 0
    if invert:
        for sym, ens in raw.items():
            ek = _normalize_ensembl_stable_id(ens)
            if ek is None:
                continue
            sym_s = str(sym).strip()
            if ek in out and out[ek] != sym_s:
                collisions += 1
            out[ek] = sym_s
    else:
        for ens, sym in raw.items():
            ek = _normalize_ensembl_stable_id(ens)
            if ek is None:
                continue
            out[ek] = str(sym).strip()

    if invert and collisions > 0:
        get_logger().warning(
            "ensembl→symbol invert: %s Ensembl IDs had conflicting symbols (last wins).",
            collisions,
        )
    return out


class DataLoader:
    """Base class for loading and preprocessing single-cell data."""
    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for data loading.
        """
        self.params = params
        self.path = params['path']
        self.dataset_name = os.path.basename(self.path).split(".")[0]
        self.layer = params['layer_name']
        self.label_key = params['label_key']
        self.batch_key = params['batch_key']
        self.train_test_split = params['train_test_split'] if 'train_test_split' in params else None
        self.cv_splits = params['cv_splits'] if 'cv_splits' in params else None
        self.log = get_logger()
        self.adata = None

    @staticmethod
    def validate_config(params):
        """Validate that required parameters are present.

        Args:
            params (dict): Configuration parameters.
        Raises:
            AssertionError: If required parameters are missing.
        """
        assert 'path' in params, "Missing required parameter: 'path'"

    def prepare_data(self, process_dir):
        """Prepare data for downstream analysis. To be implemented in subclasses.

        Args:
            process_dir (str): Directory for processing outputs.
        """
        pass

    def load(self):
        """Load data and return AnnData object. To be implemented in subclasses.

        Returns:
            ad.AnnData: Loaded data object.
        """
        print(f"Loading data from {self.path}")
        adata = ad.AnnData()
        self.adata = adata
        self.log.info(f'Data Loaded, {self.adata.X.shape}')
        self.log.info(f'min {np.min(self.adata.X)}, max {np.max(self.adata.X)}')
        return adata

    def _filter(self, adata, filter_dict):
        """Filter AnnData object based on filter_dict.

        Args:
            adata (ad.AnnData): AnnData object to filter.
            filter_dict (dict): Dictionary of {column: [values]} to filter by.
        Returns:
            ad.AnnData: Filtered AnnData object.
        """
        for col, values in filter_dict.items():
            adata = adata[adata.obs[col].isin(values)]
        # AnnData slicing returns a *view*; downstream steps (e.g. HVG) write to .uns/.var
        # and will emit ImplicitModificationWarning if run on a view.
        if getattr(adata, "is_view", False):
            adata = adata.copy()
        return adata

    def qc(self, min_genes, min_cells):
        """Apply quality control filters to the data.

        Args:
            min_genes (int): Minimum number of genes per cell.
            min_cells (int): Minimum number of cells per gene.
        """
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.log.info(f"Applying QC with min_genes={self.min_genes}, min_cells={self.min_cells}")
        self.log.info(f'obs shape before filtering, {self.adata.X.shape}')

        sc.pp.calculate_qc_metrics(self.adata, percent_top=None, log1p=False, inplace=True)
        sc.pp.filter_cells(self.adata, min_genes=int(self.min_genes))
        sc.pp.filter_genes(self.adata, min_cells=int(self.min_cells))
        self.log.info(f'obs shape after filtering, {self.adata.X.shape}')

    def scale(self, normalize, target_sum, apply_log1p):
        """Normalize and/or log-transform the data.

        Args:
            normalize (bool): Whether to normalize total counts per cell.
            target_sum (float): Target sum for normalization.
            apply_log1p (bool): Whether to apply log1p transformation.
        """
        
        # if layer_key == "X":
        #     adata.layers["counts"] = adata.X
        # elif layer_key != "counts":
        #     adata.layers["counts"] = adata.layers[layer_key]
    
        self.normalize = normalize
        self.target_sum = target_sum
        self.apply_log1p = apply_log1p

        if self.normalize:
            sc.pp.normalize_total(self.adata, target_sum=self.target_sum)
        if self.apply_log1p:
            sc.pp.log1p(self.adata)
        self.log.info(f"Applied LogScalerPreprocessor normalization to data.X, apply_log1p = {self.apply_log1p}, target_sum = {self.target_sum}")
        self.log.info(f'obs shape after scaling, {self.adata.X.shape}')

    def hvg(
        self,
        n_top_genes: int,
        flavor: str,
        batch_key: str | None = None,
        subset: bool = True,
        layer: str | None = None,
        preserve_counts: bool = False,
        counts_layer: str = "counts",
        normalize_before_hvg: bool = False,
        normalize_target_sum: float = 1e4,
        log1p_before_hvg: bool = True,
        hvg_working_layer: str = "hvg_working",
        **scanpy_kwargs,
    ):
        """Select highly variable genes (HVGs).

        By default, Scanpy can subset the AnnData in-place (dropping non-HVG genes).
        This method supports:

        - Computing HVGs from a specific `adata.layers[layer]`
        - Avoiding in-place subsetting (`subset=False`)
        - Preserving the original counts currently in `adata.X` by copying them to a layer

        Args:
            n_top_genes: Number of top HVGs to select.
            flavor: HVG selection method (e.g., "seurat", "seurat_v3").
            batch_key: Optional batch key for batch-aware HVG selection.
            subset: If True, subset the AnnData to HVGs (drops non-HVG genes).
            layer: If provided, compute HVGs using this layer instead of `X`.
            preserve_counts: If True, copy the current `X` into `layers[counts_layer]`
                before HVG selection. Useful when `X` contains raw counts and you
                later normalize/log1p `X` or compute HVGs from counts.
            counts_layer: Name of layer to store preserved counts when `preserve_counts=True`.
        """
        if preserve_counts:
            if counts_layer not in self.adata.layers:
                # Keep a snapshot of the current X (assumed counts) before any downstream transforms.
                self.adata.layers[counts_layer] = self.adata.X.copy()
                self.log.info("Preserved counts: copied adata.X -> adata.layers[%r].", counts_layer)
            else:
                self.log.info(
                    "Preserve counts requested, but adata.layers[%r] already exists; leaving it unchanged.",
                    counts_layer,
                )

        # Optionally normalize/log1p into a working layer for HVG computation.
        # This is especially useful for `flavor="seurat"` which expects log1p-normalized values.
        # We do this in a separate layer to avoid mutating raw counts.
        hvg_layer = layer
        if normalize_before_hvg:
            source_layer = layer or "X"
            if source_layer == "X":
                self.adata.layers[hvg_working_layer] = self.adata.X.copy()
            else:
                if source_layer not in self.adata.layers:
                    raise KeyError(
                        f"hvg: layer={source_layer!r} not found in adata.layers; "
                        f"available={list(self.adata.layers.keys())}"
                    )
                self.adata.layers[hvg_working_layer] = self.adata.layers[source_layer].copy()

            sc.pp.normalize_total(
                self.adata,
                target_sum=float(normalize_target_sum),
                layer=hvg_working_layer,
            )
            if log1p_before_hvg:
                sc.pp.log1p(self.adata, layer=hvg_working_layer)

            hvg_layer = hvg_working_layer
            self.log.info(
                "HVG: normalized/log1p into layers[%r] from %s (target_sum=%s, log1p=%s).",
                hvg_working_layer,
                source_layer,
                normalize_target_sum,
                log1p_before_hvg,
            )

        sc.pp.highly_variable_genes(
            self.adata,
            batch_key=batch_key,
            flavor=flavor,
            subset=bool(subset),
            n_top_genes=int(n_top_genes),
            layer=hvg_layer,
            **scanpy_kwargs,
        )
        self.log.info(
            "Applied HVG: n_top_genes=%s flavor=%s batch_key=%s subset=%s layer=%s preserve_counts=%s counts_layer=%s",
            n_top_genes,
            flavor,
            batch_key,
            subset,
            hvg_layer,
            preserve_counts,
            counts_layer,
        )
        self.log.info("Data shape after HVG, %s", self.adata.X.shape)


class CSVDataLoader(DataLoader):
    """Loader for CSV-formatted single-cell data."""
    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for CSV loading.
        """
        super().__init__(params)
        self.path = params['path']

    def load(self):
        """Load data from a CSV file.

        Returns:
            ad.AnnData: Loaded data object.
        """
        self.log.info(f"Loading CSV data from {self.path}")
        df = pd.read_csv(self.path)
        labels = df['label'].values
        features = df.drop(columns=['label']).values
        adata = ad.AnnData(X=features, obs=pd.DataFrame({'label': labels}))
        self.adata = adata
        return adata


class H5ADLoader(DataLoader):
    """Loader for H5AD-formatted single-cell data."""
    def __init__(self, params):
        """
        Args:
            params (dict): Configuration parameters for H5AD loading.
        """
        super().__init__(params)
        self.path = params['path']
        self.load_raw = bool(params['load_raw'])
        self.label_key = params['label_key']
        self.batch_key = params['batch_key']
        self.filter = params.get('filter', None)

    def load(self):
        """Load data from an H5AD file, with optional filtering and splitting.

        Returns:
            ad.AnnData: Loaded data object.
        """
        logging.info(f"Loading H5AD data from {self.path}")
        adata = ad.read_h5ad(self.path)
        adata.obs['label'] = adata.obs[self.label_key]
        adata.obs['batch'] = adata.obs[self.batch_key]
        if self.load_raw:
            adata = adata.raw.to_adata()
        if self.layer != 'X':
            adata.layers['original_X'] = adata.X
            adata.X = adata.layers[self.layer]
            
        # if self.layer == "X" and 'counts' not in adata.layers:
        #     adata.layers["counts"] = adata.X
        
        self.log.info(f'Data Loaded, {adata.X.shape}')
        self.log.info(f'X min {np.min(adata.X)}, X max {np.max(adata.X)}')

        if self.filter is not None:
            self.log.info(self.filter)
            filter_dict = {entry["column"]: entry["values"] for entry in self.filter}
            adata = self._filter(adata, filter_dict)

        self.adata = adata

        if self.train_test_split:
            fname = os.path.join(BASE_PATH, self.train_test_split)
            with open(fname, "r", encoding="utf-8") as f:
                self.train_test_split_dict = json.load(f)
            fname = os.path.join(BASE_PATH, self.cv_splits)
            with open(fname, "r", encoding="utf-8") as f:
                self.cv_split_dict = json.load(f)
        else:
            self.train_test_split_dict = None
            self.cv_split_dict = None

        return adata

    def map_ensembl(self, gene_to_ensembl_dict):
        """Map ``var`` index to Ensembl IDs and drop genes with no mapping (Geneformer / CellPLM).

        ``gene_to_ensembl_dict`` may be:

        - ``dict``: keys = gene symbols (as in ``adata.var_names``), values = Ensembl gene IDs.
        - ``str`` path to a ``.pkl`` file containing such a dict (e.g. Geneformer
          ``gene_name_id_dict_gc104M.pkl``).
        - ``str`` path to a **directory** that contains ``gene_name_id_dict.pkl`` in the
          layout expected by :func:`data.gf_loader.get_gene_name_ensembl_map`.
        """
        from scfm_cancer_eval.data.gf_loader import get_gene_name_ensembl_map

        if isinstance(gene_to_ensembl_dict, str):
            path = gene_to_ensembl_dict
            if os.path.isfile(path) and path.lower().endswith(".pkl"):
                with open(path, "rb") as f:
                    gene_to_ensembl_dict = pickle.load(f)
                if not isinstance(gene_to_ensembl_dict, dict):
                    raise TypeError(
                        "map_ensembl: pickle %r must load to a dict (symbol -> Ensembl ID)."
                        % (path,)
                    )
            else:
                _, gene_to_ensembl_dict = get_gene_name_ensembl_map(path)

        # Use Series (not Index): Index.fillna rejects a non-scalar fill value in recent pandas.
        vm = pd.Series(
            self.adata.var_names.astype(str).str.strip(),
            index=self.adata.var_names,
        )
        ens_direct = vm.map(gene_to_ensembl_dict)
        # Geneformer-style dicts use uppercase symbols; many h5ads use mixed case.
        upper_map = {
            str(k).strip().upper(): v
            for k, v in gene_to_ensembl_dict.items()
            if k is not None
        }
        ens_upper = vm.str.upper().map(upper_map)
        ens_from_symbols = ens_direct.fillna(ens_upper)
        # var_names may already be Ensembl IDs with version / extra tags (not in symbol dict).
        parsed = vm.map(_stable_ensembl_from_var_label)
        ens = ens_from_symbols.fillna(parsed)
        n_upper = int(ens_from_symbols.notna().sum() - ens_direct.notna().sum())
        if n_upper > 0:
            self.log.info(
                "map_ensembl: matched %d genes via uppercase symbol keys (after direct lookup).",
                n_upper,
            )
        n_parse = int((parsed.notna() & ens_from_symbols.isna()).sum())
        if n_parse > 0:
            self.log.info(
                "map_ensembl: matched %d genes by parsing Ensembl IDs from var_names (e.g. GENCODE suffixes).",
                n_parse,
            )

        self.adata.var["ensembl_id"] = ens
        nan_idx = self.adata.var.ensembl_id.isna()
        n = int(nan_idx.sum())
        sample_vars = list(self.adata.var_names[: min(8, self.adata.n_vars)])
        sample_keys = list(gene_to_ensembl_dict.keys())[:8]
        self.adata = self.adata[:, ~nan_idx]
        if self.adata.n_vars == 0:
            raise ValueError(
                "map_ensembl: every gene was dropped (no symbol→Ensembl match and no parseable ENSG… in var_names). "
                "If var_names are already Ensembl IDs, remove gene_name_id_dict from the CellPLM YAML or use "
                "ensembl_auto_conversion: false. Example var_names=%r, example dict keys=%r"
                % (sample_vars, sample_keys)
            )
        self.log.warning(
            "Genes without Ensembl IDs: %s (removed). Remaining shape %s", n, self.adata.shape
        )
        self.log.info(self.adata.var.head())

    def map_ensembl_to_gene_names(self, ensembl_to_symbol_dict):
        """Rename ``adata.var_names`` from Ensembl IDs to gene symbols (e.g. scFoundation / HGNC panels).

        Complement to :meth:`map_ensembl`, which sets ``var['ensembl_id']`` and keeps symbol names.
        This method resolves each gene's Ensembl stable ID, looks up a **symbol**, assigns
        ``var['ensembl_id']`` to that stable ID (or the parsed ID), renames ``var_names`` to
        symbols, drops genes with no mapping, then runs :meth:`anndata.AnnData.var_names_make_unique`.

        ``ensembl_to_symbol_dict`` may be:

        - ``dict``: Ensembl stable gene ID -> gene symbol (values may be mixed case; keys are normalized).
        - ``str`` path to a ``.pkl`` file containing such a dict, **or** a Geneformer-style
          **symbol -> Ensembl** dict (same as for :meth:`map_ensembl`); orientation is inferred
          from whether keys vs values look like Ensembl IDs.
        - ``str`` path to a **directory** that contains ``gene_name_id_dict.pkl`` in the layout
          expected by :func:`data.gf_loader.get_gene_name_ensembl_map` (uses the inverted Ensembl→symbol map).
        """
        from scfm_cancer_eval.data.gf_loader import get_gene_name_ensembl_map

        if isinstance(ensembl_to_symbol_dict, str):
            path = ensembl_to_symbol_dict
            if os.path.isfile(path) and path.lower().endswith(".pkl"):
                with open(path, "rb") as f:
                    raw = pickle.load(f)
                if not isinstance(raw, dict):
                    raise TypeError(
                        "map_ensembl_to_gene_names: pickle %r must load to a dict."
                        % (path,)
                    )
                ens_map = _coerce_to_ensembl_symbol_map(raw)
            elif os.path.isdir(path):
                ens_map, _ = get_gene_name_ensembl_map(path)
                ens_map = _coerce_to_ensembl_symbol_map(ens_map)
            else:
                raise ValueError(
                    "map_ensembl_to_gene_names: expected a .pkl file or a directory with "
                    "gene_name_id_dict.pkl, got %r."
                    % (path,)
                )
        elif isinstance(ensembl_to_symbol_dict, dict):
            ens_map = _coerce_to_ensembl_symbol_map(ensembl_to_symbol_dict)
        else:
            raise TypeError(
                "map_ensembl_to_gene_names: expected dict or str path, got %s"
                % type(ensembl_to_symbol_dict).__name__
            )

        if not ens_map:
            raise ValueError(
                "map_ensembl_to_gene_names: empty mapping after load/normalize."
            )

        orig_names = list(self.adata.var_names.astype(str))
        stable_ids: list[str | None] = []
        symbols: list[str | None] = []
        for rn in orig_names:
            sid = _normalize_ensembl_stable_id(rn)
            sym = ens_map.get(sid) if sid is not None else None
            stable_ids.append(sid)
            symbols.append(sym)

        self.adata.var["ensembl_id"] = stable_ids

        ok_mask = np.array(
            [(not pd.isna(s)) and str(s).strip() != "" for s in symbols],
            dtype=bool,
        )
        n_drop = int((~ok_mask).sum())
        sample_vars = orig_names[: min(8, len(orig_names))]
        sample_keys = list(ens_map.keys())[:8]
        self.adata = self.adata[:, ok_mask]
        if self.adata.n_vars == 0:
            raise ValueError(
                "map_ensembl_to_gene_names: every gene was dropped (no Ensembl→symbol match). "
                "Example var_names=%r, example map keys=%r"
                % (sample_vars, sample_keys)
            )
        if n_drop:
            self.log.warning(
                "map_ensembl_to_gene_names: dropped %s genes with no Ensembl→symbol mapping. "
                "Remaining shape %s",
                n_drop,
                self.adata.shape,
            )

        new_names = [str(symbols[i]).strip() for i in range(len(symbols)) if ok_mask[i]]
        self.adata.var_names = new_names
        # AnnData API (Scanpy no longer exposes ``sc.pp.var_names_make_unique`` in recent versions).
        self.adata.var_names_make_unique()
        self.log.info("map_ensembl_to_gene_names: renamed var_names; head:\n%s", self.adata.var.head())

    def prepare_data(self, save_dir=None, save_ext="loom"):
        """Write processed AnnData to loom or h5ad for Geneformer ``TranscriptomeTokenizer`` (was ``GFLoader``-only)."""
        self.processed_dir = os.path.join(save_dir, "processed_data")
        self.adata.obs["n_counts"] = self.adata.obs["total_counts"]
        self.adata.obs["adata_order"] = self.adata.obs.index.tolist()
        self.log.info(self.adata.shape)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if save_ext == "loom":
            self.procssed_file = os.path.join(self.processed_dir, f"{self.dataset_name}.loom")
            self.adata.write_loom(self.procssed_file)
            self.log.info("Saving loom file to %s", self.procssed_file)
        elif save_ext == "h5ad":
            self.procssed_file = os.path.join(self.processed_dir, f"{self.dataset_name}.h5ad")
            self.adata.write_h5ad(self.procssed_file)
            self.log.info("Saving h5ad file to %s", self.procssed_file)

