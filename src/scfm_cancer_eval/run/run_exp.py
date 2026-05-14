"""
Main experiment runner for scFM_eval.
Handles configuration, data loading, preprocessing, feature extraction, model training, and evaluation.
"""
import os

# Cap BLAS/OpenMP thread pools before NumPy/PyTorch load (setdefault preserves user-set exports).
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

# ---- warnings policy (must run before importing scanpy/scib) ----
# We suppress a couple of noisy *third-party* FutureWarnings that are not actionable
# inside this repo. Keep the filters narrow so other warnings still surface.
import warnings

# Allow users to disable suppression if they want to see everything.
_suppress = os.environ.get("SCFM_EVAL_SUPPRESS_WARNINGS", "1").lower() not in ("0", "false", "no")
if _suppress:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*pandas\.value_counts is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*Argument `use_highly_variable` is deprecated.*",
    )

# import os
# os.environ.setdefault("PYTHONHASHSEED", "0")
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")      # if needed
# # os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")      # if TF
# # os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")       # if JAX

# # 1) Warnings policy
# import warnings
# warnings.simplefilter("ignore", category=FutureWarning)

# # 2) Seed the world early
# import random, numpy as np
# random.seed(0)
# np.random.seed(0)

# # 3) Now import frameworks and lock determinism
# import torch
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# # torch.use_deterministic_algorithms(True, warn_only=True)

# # 4) Only now import project modules (which may have import-time side effects)
# import sys
# from os.path import dirname, abspath
# dir_path = dirname(dirname(abspath(__file__)))
# if dir_path not in sys.path:
#     sys.path.insert(0, dir_path)

# # Core libs you rely on explicitly
# import yaml, pandas as pd, anndata as ad
# from functools import wraps
# from os.path import dirname, abspath, join, basename, exists
# import shutil
# import time
# import importlib

# # Project modules last
# from viz.visualization import EmbeddingVisualizer
# from evaluation.eval import EmbeddingEvaluator
# from utils.logs_ import set_logging, get_logger
# from scfm_cancer_eval.setup_path import BASE_PATH, OUTPUT_PATH, PARAMS_PATH, DATA_PATH

#-------good list --
import copy
import hashlib
import json
import yaml
from typing import Optional
from pathlib import Path
import importlib
from os.path import join, basename, exists
import shutil

import logging
# logger = logging.getLogger('ml_logger')
from scfm_cancer_eval.viz.visualization import EmbeddingVisualizer
from scfm_cancer_eval.evaluation.eval import EmbeddingEvaluator
from scfm_cancer_eval.utils.logs_ import set_logging, get_logger
from scfm_cancer_eval.setup_path import BASE_PATH, OUTPUT_PATH, PARAMS_PATH, DATA_PATH
from scfm_cancer_eval.utils.sampling import sample_adata

embedding_method_map= dict(PCA='X_pca', HVG='X_hvg', scVI='X_scVI', geneformer='X_geneformer', scgpt='X_scGPT')

import random
import numpy as np
import torch
import time
from functools import wraps
import pandas as pd
import anndata as ad

# New unified results record (schema v1.1.0)
from scfm_cancer_eval.utils.results_json import build_results_json, write_results_json

_LEGACY_EXPERIMENT_MODULE_ROOTS = frozenset(
    {
        "data",
        "features",
        "models",
        "viz",
        "evaluation",
        "utils",
        "run",
        "data_splits",
    }
)


def _resolve_experiment_module(module_path: str) -> str:
    """Map YAML ``module: data.foo`` style paths to the installable package name."""
    if module_path.startswith("scfm_cancer_eval."):
        return module_path
    root = module_path.split(".", 1)[0]
    if root in _LEGACY_EXPERIMENT_MODULE_ROOTS:
        return f"scfm_cancer_eval.{module_path}"
    return module_path


def _resolve_experiment_config_path(config_path: str) -> str:
    """Absolute path to the entry YAML; relative paths are under :data:`PARAMS_PATH`."""
    if os.path.isabs(config_path):
        return os.path.normpath(config_path)
    return os.path.normpath(join(PARAMS_PATH, config_path))


def _output_subdir_for_config(cli_config: str, resolved_config_path: str) -> str:
    """
    Directory under OUTPUT_PATH for this run.

    Relative CLI paths (resolved under PARAMS_PATH) use the same path-shaped layout
    as before, e.g. ``exp/geneformer/.../brca_cell_type`` — *not* ``external/...``.

    Only when the user passes an **absolute** path on the CLI do we use ``external/``,
    because ``join(OUTPUT_PATH, "/abs/path")`` would otherwise ignore OUTPUT_PATH.
    """
    if os.path.isabs(cli_config):
        key = hashlib.sha256(
            os.path.normpath(resolved_config_path).encode("utf-8")
        ).hexdigest()[:16]
        stem = os.path.splitext(os.path.basename(resolved_config_path))[0]
        return join("external", key, stem)
    return os.path.splitext(cli_config)[0]


# RNG / determinism snapshot
# import numpy as np, torch, os
# print("np first rand:", np.random.RandomState(0).rand())
# print("torch cudnn:", torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
# print("matmul precision:", getattr(torch, "get_float32_matmul_precision", lambda:"n/a")())

# # Threads & BLAS
# print("OMP:", os.environ.get("OMP_NUM_THREADS"), "MKL:", os.environ.get("MKL_NUM_THREADS"))
# try:
#     import mkl
#     print("MKL threads:", mkl.get_max_threads())
# except Exception:
#     pass

# # Who imported numpy/torch first?
# import sys
# print("Imported by:", sys.modules['numpy'].__loader__)
# print("Torch loaded from:", sys.modules['torch'].__file__)

# --------bad list ----
# import os
# import sys
# import shutil
# import logging
# import random
# import time
# from functools import wraps
# from pathlib import Path
# import importlib

# from os.path import dirname, abspath, join, basename, exists

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# dir_path = dirname(dirname(abspath(__file__)))
# sys.path.insert(0,dir_path)

# import yaml
# import numpy as np
# import torch
# import pandas as pd
# import anndata as ad

# from viz.visualization import EmbeddingVisualizer
# from evaluation.eval import EmbeddingEvaluator
# from utils.logs_ import set_logging, get_logger
# from scfm_cancer_eval.setup_path import BASE_PATH, OUTPUT_PATH, PARAMS_PATH, DATA_PATH
# RNG / determinism snapshot
# import numpy as np, torch, os
# print("np first rand:", np.random.RandomState(0).rand())
# print("torch cudnn:", torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
# print("matmul precision:", getattr(torch, "get_float32_matmul_precision", lambda:"n/a")())

# # Threads & BLAS
# print("OMP:", os.environ.get("OMP_NUM_THREADS"), "MKL:", os.environ.get("MKL_NUM_THREADS"))
# try:
#     import mkl
#     print("MKL threads:", mkl.get_max_threads())
# except Exception:
#     pass

# # Who imported numpy/torch first?
# import sys
# print("Imported by:", sys.modules['numpy'].__loader__)
# print("Torch loaded from:", sys.modules['torch'].__file__)

# ------------

# Legacy mapping kept for backward compatibility only.
# New code derives the key from the extractor (`extractor.output_key`).
embedding_method_map = dict(PCA='X_pca', HVG='X_hvg', scVI='X_scVI', geneformer='X_geneformer', scgpt='X_scGPT', scfoundation='X_scfoundation', scimilarity='X_scimilarity', cellplm='X_CellPLM', nicheformer='X_nicheformer', scconcept='X_scconcept', state='X_state', mock='X_mock')

# List to store timing records
def _get_timing_log():
    """Return the global timing log list."""
    global _timing_log
    if '_timing_log' not in globals():
        _timing_log = []
    return _timing_log

_timing_log = _get_timing_log()


def _json_default(o):
    """Serialize numpy scalars/arrays and pathlib paths for json.dump."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _value_for_metrics_csv(v):
    """Scalar values safe for pandas CSV cells (numpy / pathlib)."""
    if v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def timing(func):
    """
    Decorator to measure and store execution time of a function.
    Appends timing info to the global _timing_log list.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        _timing_log.append({
            'function': func.__name__,
            'time_seconds': elapsed
        })
        return res
    return wrapper

def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
        deterministic (bool): If True, sets PyTorch to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        # Required for deterministic cuBLAS reductions on CUDA; harmless on CPU-only runs.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except (RuntimeError, AttributeError):
                pass
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    print(f"Random seed set to: {seed}")

def get_configs(config_path):
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Tuple containing run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, classification_config.
    """
    def deep_merge_dicts(base: dict, override: dict) -> dict:
        """
        Recursively merge dictionaries.
        - dict + dict -> recursive merge
        - otherwise -> override wins (lists/scalars are replaced)
        """
        out = copy.deepcopy(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge_dicts(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    def resolve_base_path(base_path: str, cfg_path: str) -> str:
        # Allow absolute paths, relative-to-config paths (./ or ../), or relative to PARAMS_PATH.
        if os.path.isabs(base_path):
            return base_path
        if base_path.startswith("."):
            return os.path.normpath(join(os.path.dirname(cfg_path), base_path))
        return os.path.normpath(join(PARAMS_PATH, base_path))

    def load_config_with_bases(cfg_path: str, visited: set[str]) -> dict:
        cfg_path = os.path.abspath(cfg_path)
        if cfg_path in visited:
            raise ValueError(f"Cyclic YAML bases detected: {cfg_path}")
        visited.add(cfg_path)

        with open(cfg_path, "r") as file:
            config = yaml.safe_load(file) or {}

        # Optional, more readable composition keys:
        # - dataset: list of dataset fragments (or single string)
        # - model: list of model (embedding / extractor) fragments (or single string)
        # - classification: list of classifier fragments (or single string)
        # Backward compatible with:
        # - datasets / classifications
        # - bases / defaults
        dataset = config.get("dataset", None)
        model = config.get("model", None)
        classification = config.get("classification", None)
        datasets = config.pop("datasets", None)  # deprecated alias
        classifications = config.pop("classifications", None)  # deprecated alias
        models = config.pop("models", None)  # optional plural alias for model includes
        bases = config.pop("bases", None)
        if bases is None:
            bases = config.pop("defaults", None)

        def _as_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return x
            return [x]

        # Interpret singular keys as *include lists* only when they are paths (str/list),
        # not when they are actual config dicts (which should remain in the merged config).
        dataset_includes = None
        if isinstance(dataset, (str, list)):
            dataset_includes = config.pop("dataset", None)

        model_includes = None
        if isinstance(model, (str, list)):
            model_includes = config.pop("model", None)

        classification_includes = None
        if isinstance(classification, (str, list)):
            classification_includes = config.pop("classification", None)

        # Merge order: dataset -> model (embedding) -> classification -> bases -> local overrides
        bases_list = (
            _as_list(dataset_includes)
            + _as_list(datasets)
            + _as_list(model_includes)
            + _as_list(models)
            + _as_list(classification_includes)
            + _as_list(classifications)
            + _as_list(bases)
        )

        merged: dict = {}
        for base in bases_list:
            base_cfg_path = resolve_base_path(str(base), cfg_path)
            base_cfg = load_config_with_bases(base_cfg_path, visited)
            merged = deep_merge_dicts(merged, base_cfg)

        merged = deep_merge_dicts(merged, config)
        visited.remove(cfg_path)
        return merged

    config = load_config_with_bases(config_path, visited=set())

    run_id = config.get("run_id", "")
    # Optional: allow a dedicated `task` block for task metadata (label_key, label_map, etc.)
    task_config = config.get("task", None)

    data_config = config["dataset"]
    qc_config = config["qc"]
    preproc_config = config["preprocessing"]
    feat_config = config["embedding"]
    classification_config = config["classification"]
    hvg_config = config.get("hvg", None)
    return run_id, data_config, qc_config, preproc_config, hvg_config, feat_config, classification_config, config, task_config

class Experiment:
    """
    Handles the execution of a machine learning experiment defined by a YAML config.
    Handles loading data, preprocessing, feature extraction, model training, and evaluation.
    """
    def __init__(
        self,
        config_path,
        seed: int = 42,
        max_cells: Optional[int] = None,
        max_cells_stratify: Optional[str] = None,
    ):
        """
        Initialize the Experiment with a given config path.
        Sets up directories, logging, and loads configuration.

        Args:
            config_path: Path to the entry YAML (relative to PARAMS_PATH, or absolute).
            seed: Global RNG seed used when config does not override (e.g. subsampling, classifier).
            max_cells: If set and positive, cap the number of cells after load (overrides dataset.max_cells
                in YAML). Useful for quick tests without editing the config.
            max_cells_stratify: If set (including empty string), ``obs`` column used for stratified
                subsampling when ``max_cells`` applies (overrides dataset.max_cells_stratify in YAML).
                Empty string forces uniform random subsampling (no stratification).
        """
        self.rng_seed = int(seed)
        self._cli_max_cells: Optional[int] = None
        if max_cells is not None:
            mc = int(max_cells)
            if mc > 0:
                self._cli_max_cells = mc
        self._cli_max_cells_stratify: Optional[str] = max_cells_stratify
        self._cli_config_path = config_path
        self.config_path = _resolve_experiment_config_path(config_path)
        (
            self.run_id,
            self.data_config,
            self.qc_config,
            self.preproc_config,
            self.hvg,
            self.feat_config,
            self.classification_config,
            self.resolved_config,
            self.task_config,
        ) = get_configs(self.config_path)

        self.vis_embedding = bool(self.feat_config['viz'])
        self.eval_embedding = bool(self.feat_config['eval'])

        # Prepare saving dir
        relative_sav_dir = _output_subdir_for_config(self._cli_config_path, self.config_path)
        config_filename = os.path.basename(self.config_path)
        save_dir = join(OUTPUT_PATH, relative_sav_dir)
        self.save_dir = save_dir + f'_{self.run_id}' if self.run_id else save_dir
        print(f'save_dir: {self.save_dir}')
        if not exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Copy wrapper YAML and also save the fully resolved config
        shutil.copyfile(self.config_path, join(self.save_dir, config_filename))
        with open(join(self.save_dir, "resolved_config.yaml"), "w") as f:
            yaml.safe_dump(self.resolved_config, f, sort_keys=False)

        # Set logging format
        self.log = set_logging(self.save_dir)

        if os.environ.get("SCFM_VALIDATE_EXP", "").lower() in ("1", "true", "yes"):
            from scfm_cancer_eval.utils.validate_exp_constraints import validate_merged_config

            verrs, vwarns = validate_merged_config(self.resolved_config)
            for w in vwarns:
                self.log.warning("SCFM_VALIDATE_EXP: %s", w)
            if verrs:
                for e in verrs:
                    self.log.error("SCFM_VALIDATE_EXP: %s", e)
                raise ValueError(
                    "Experiment config failed embedding data constraint validation: "
                    + "; ".join(verrs)
                )

        # Placeholders for data, embeddings, model, and results
        self.data = None
        self.embedding = None
        self.embedding_key = None
        self.model = None
        self.results = {}
        self.classification_metrics = None
        self.embedding_metrics = None

        # Lightweight run summary (finalized at end)
        self.run_summary = dict(
            run_id=self.run_id,
            config_path=self._cli_config_path,
            resolved_entry_yaml=self.config_path,
            save_dir=self.save_dir,
            random_seed=self.rng_seed,
            dataset_path=self.data_config.get("path") if isinstance(self.data_config, dict) else None,
            label_key=self.data_config.get("label_key") if isinstance(self.data_config, dict) else None,
            batch_key=self.data_config.get("batch_key") if isinstance(self.data_config, dict) else None,
            embedding_method=self.feat_config.get("method") if isinstance(self.feat_config, dict) else None,
            max_cells_cli=self._cli_max_cells,
            max_cells_stratify_cli=self._cli_max_cells_stratify,
        )
        # best-effort run timing (RFC3339); set finished at report time
        from datetime import datetime, timezone
        self.run_summary["started_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @timing
    def run(self):
        """
        Run the complete experiment workflow: data loading, QC, preprocessing, feature extraction, visualization, evaluation, and classification.
        """
        self.load_data()
        self.qc_data()
        self.preprocess_data()
        if self.hvg:
            self.filter_hvg()
        self.extract_embeddings()
        if self.vis_embedding:
            self.visualize_embedding()
        if self.eval_embedding:
            self.evaluate_embedding()
        self.train_classifier()

    def load_class(self, module_path, class_name):
        """
        Dynamically import and return a class by module and name.

        Args:
            module_path (str): Python module path.
            class_name (str): Name of the class to import.

        Returns:
            type: The class object.
        """
        module = importlib.import_module(_resolve_experiment_module(module_path))
        return getattr(module, class_name)

    @timing
    def load_data(self):
        """
        Load the dataset using the loader specified in the config.
        Optional: set ``max_cells`` (and optionally ``max_cells_stratify``) on the dataset YAML block.
        """
        loader_config = self.data_config
        loader_config['path'] = join(DATA_PATH, loader_config['path'])
        LoaderClass = self.load_class(loader_config['module'], loader_config['class'])
        # Pretty-print config for readability in logs.
        try:
            cfg_pretty = yaml.safe_dump(loader_config, sort_keys=False, default_flow_style=False).rstrip()
        except Exception:
            cfg_pretty = repr(loader_config)
        self.log.info("Data Loader config:\n%s", cfg_pretty)
        self.loader = LoaderClass(loader_config)
        self.data = self.loader.load()

        max_cells = self._cli_max_cells if self._cli_max_cells is not None else loader_config.get('max_cells')
        if self._cli_max_cells is not None:
            self.log.info(
                "Using --max-cells CLI override: %s (YAML dataset.max_cells was %s)",
                self._cli_max_cells,
                loader_config.get('max_cells'),
            )
        if self._cli_max_cells_stratify is not None:
            self.log.info(
                "Using --max-cells-stratify CLI override: %r (YAML dataset.max_cells_stratify was %s)",
                self._cli_max_cells_stratify,
                loader_config.get('max_cells_stratify'),
            )
        if max_cells is not None and int(max_cells) > 0:
            max_cells = int(max_cells)
            n_obs = self.loader.adata.n_obs
            if n_obs > max_cells:
                stratify = (
                    self._cli_max_cells_stratify
                    if self._cli_max_cells_stratify is not None
                    else loader_config.get('max_cells_stratify')
                )
                if stratify == "":
                    stratify = None
                rs = int(loader_config.get('max_cells_random_state', self.rng_seed))
                self.log.info(
                    "Subsampling cells: n_obs=%s -> max_cells=%s, stratify_by=%s, random_state=%s",
                    n_obs,
                    max_cells,
                    stratify,
                    rs,
                )
                self.loader.adata = sample_adata(
                    self.loader.adata,
                    sample_size=max_cells,
                    stratify_by=stratify,
                    random_state=rs,
                )
                self.data = self.loader.adata
                self.run_summary['max_cells'] = max_cells
                self.run_summary['max_cells_stratify'] = stratify
            else:
                self.log.info("max_cells=%s not applied (n_obs=%s).", max_cells, n_obs)

    @timing
    def qc_data(self):
        """
        Apply quality control filters to the dataset.
        """
        qc_config = self.qc_config
        if ('skip' in qc_config) and (qc_config['skip'] is True):
            return
        self.loader.qc(**qc_config)

    @timing
    def preprocess_data(self):
        """
        Apply preprocessing steps such as normalization.
        """
        preproc_config = self.preproc_config
        if ('skip' in preproc_config) and (preproc_config['skip'] is True):
            return
        self.loader.scale(**preproc_config)

    @timing
    def filter_hvg(self):
        """
        Select High Variant Genes only, if specified in config.
        """
        hvg_config = self.hvg
        if ('skip' in hvg_config) and (hvg_config['skip'] is True):
            self.log.info('Skipping HVG. Analyzing all genes')
            return
        self.loader.hvg(**hvg_config)

    @timing
    def extract_embeddings(self):
        """
        Extract features or embeddings from the dataset using the specified extractor.
        """
        self.log.info('Extract features or embeddings from the dataset')
        feat_config = self.feat_config
        if ('skip' in feat_config) and (feat_config['skip'] is True):
            return
        # Ensure params exists
        feat_config.setdefault("params", {})
        feat_config["params"]["save_dir"] = self.save_dir
        ExtractorClass = self.load_class(feat_config['module'], feat_config['class'])
        extractor = ExtractorClass(feat_config)
        # Prefer extractor-defined key; fall back to legacy mapping.
        self.embedding_key = getattr(extractor, "output_key", None) or (
            extractor.get_output_key() if hasattr(extractor, "get_output_key") else None
        )
        if not self.embedding_key:
            self.embedding_key = embedding_method_map.get(feat_config["method"], f"X_{feat_config['method']}")
        
        #loaded pre-calculated embeddings h5ad
        fname= join(self.save_dir, 'data.h5ad')
        if ('load_cached' in feat_config) and (feat_config['load_cached'] is True):
            try: 
                extractor.load_cashed(self.loader, fname)
                self.log.info(f'Loaded cached embeddings {fname}')
                self.log.info(f'Shape {self.loader.adata.shape}')
                self.embedding = self.loader.adata.obsm[self.embedding_key]
                return 
            except: 
                self.log.info('Failed to load cached embeddings. Recalculating the embeddings')

        self.embedding = extractor.fit_transform(self.loader)
        #TODO: save space, save embeddings only
        adata_embedding = ad.AnnData(X=self.embedding, obs =  self.loader.adata.obs)
        adata_embedding.obsm[self.embedding_key] = adata_embedding.X
        adata_embedding.write_h5ad(fname, compression='gzip')

    @timing
    def train_classifier(self):
        """
        Train the machine learning model as specified in the config.
        """
        if ('skip' in self.classification_config) and (self.classification_config['skip'] is True):
            return
        clf_config = self.classification_config
        # Allow label maps to live under dataset config (shared per task/dataset).
        # If not explicitly set in classifier params, inherit from dataset.label_map.
        if 'params' in clf_config and isinstance(clf_config['params'], dict):
            clf_config['params'].setdefault('random_state', self.rng_seed)
            if 'label_map' not in clf_config['params']:
                dataset_label_map = None
                if isinstance(self.data_config, dict):
                    dataset_label_map = self.data_config.get('label_map', None)
                # Prefer explicit task label map if present
                if isinstance(self.task_config, dict) and self.task_config.get("label_map") is not None:
                    dataset_label_map = self.task_config.get("label_map")
                if dataset_label_map is not None:
                    clf_config['params']['label_map'] = dataset_label_map
        clf_config['params']['save_dir'] = self.save_dir
        viz = bool(self.classification_config['viz'])
        eval_ = bool(self.classification_config['eval'])
        clf_config['viz'] = viz
        clf_config['eval'] = eval_
        clf_config['params']['embedding_col'] = self.embedding_key
        ClfClass = self.load_class(clf_config['module'], clf_config['class'])
        clf = ClfClass(clf_config)
        clf.train(self.loader)
        # Try to collect a compact summary from saved CSVs (if produced).
        self.classification_metrics = self._collect_classification_metrics()

    def evaluate_model(self):
        """
        Run additional evaluation modules as specified in the config (if any).
        """
        evaluations = self.config.get('evaluations', [])
        for eval_cfg in evaluations:
            EvaluatorClass = self.load_class(eval_cfg['module'], eval_cfg['class'])
            evaluator = EvaluatorClass(eval_cfg)
            result = evaluator.evaluate(self.model, self.embedding, self.data)
            self.results[eval_cfg['name']] = result
        print("Evaluation Results:", self.results)

    def visualize_embedding(self):
        """
        Generate and save embedding visualizations using EmbeddingVisualizer.
        """
        visualizer = EmbeddingVisualizer(self.embedding, self.loader.adata.obs, save_dir=self.save_dir)
        visualizer.plot()

    def evaluate_embedding(self):
        """
        Evaluate the quality of the learned embeddings using EmbeddingEvaluator.
        """
        evaluator = EmbeddingEvaluator(
            self.loader.adata,
            embedding_key=self.embedding_key,
            save_dir=self.save_dir,
            random_state=self.rng_seed,
        )
        self.embedding_metrics = evaluator.evaluate()

    def _collect_classification_metrics(self):
        """
        Best-effort parser for classifier metrics written by ClassifierPipeline.
        Returns a dict (may be empty).
        """
        out = {}
        try:
            import pandas as pd
        except Exception:
            return out

        # Prefer CV metrics if present
        cv_dir = join(self.save_dir, "cv")
        if os.path.isdir(cv_dir):
            for fname in sorted(os.listdir(cv_dir)):
                if fname.endswith("_cv_metrics.csv"):
                    df = pd.read_csv(join(cv_dir, fname), index_col=0)
                    # expected columns: ['Metrics', <model_name>, 'fold'] or similar
                    # try common patterns
                    if "Metrics" in df.columns:
                        # pivot-like format
                        pass
                    # If it's in a tidy format: Metrics, <model>, fold
                    if "fold" in df.columns:
                        metric_cols = [c for c in df.columns if c not in ("fold", "Metrics")]
                        if metric_cols:
                            model_col = metric_cols[0]
                            # mean over folds
                            means = df.groupby("Metrics")[model_col].mean().to_dict() if "Metrics" in df.columns else {}
                            if means:
                                out["cv_mean"] = means
                                out["cv_file"] = join("cv", fname)
                                return out

        # Fall back to per-run metrics files in root save_dir
        for fname in sorted(os.listdir(self.save_dir)):
            if fname.startswith("cls_metrics_") and fname.endswith(".csv"):
                df = pd.read_csv(join(self.save_dir, fname), index_col=0)
                # format: index Metrics, single model column
                if df.shape[1] >= 1:
                    col = df.columns[0]
                    out[fname.replace(".csv", "")] = df[col].to_dict()
        return out

    def _write_standard_reports(self):
        metrics = dict(
            embedding_metrics=self.embedding_metrics,
            classification_metrics=self.classification_metrics,
        )
        with open(join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)

        # Update run summary
        self.run_summary["embedding_key"] = self.embedding_key
        from datetime import datetime, timezone
        self.run_summary["finished_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with open(join(self.save_dir, "run_summary.json"), "w") as f:
            json.dump(self.run_summary, f, indent=2, default=_json_default)

        # New: schema'd results.json (multi-evaluation, folds+aggregate when available)
        try:
            results_payload = build_results_json(
                run_id=self.run_id,
                save_dir=self.save_dir,
                repo_root=BASE_PATH,
                config_path=self.run_summary.get("config_path"),
                resolved_config_path="resolved_config.yaml",
                resolved_config=self.resolved_config if isinstance(self.resolved_config, dict) else None,
                dataset_cfg=self.data_config if isinstance(self.data_config, dict) else None,
                embedding_cfg=self.feat_config if isinstance(self.feat_config, dict) else None,
                task_cfg=self.task_config if isinstance(self.task_config, dict) else None,
                embedding_key=self.embedding_key,
                embedding_metrics=self.embedding_metrics if isinstance(self.embedding_metrics, dict) else None,
                rng_seed=self.rng_seed,
                started_at=self.run_summary.get("started_at"),
                finished_at=self.run_summary.get("finished_at"),
                status="success",
            )
            write_results_json(join(self.save_dir, "results.json"), results_payload)
            self.log.info("Wrote results.json: %s", join(self.save_dir, "results.json"))
        except Exception:
            self.log.exception("Failed to write results.json (non-fatal).")

        # Append one row to a global CSV in OUTPUT_PATH
        metrics_csv = join(OUTPUT_PATH, "metrics_runs.csv")
        try:
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            row = dict(
                run_id=self.run_id,
                config_path=self.run_summary.get("config_path"),
                resolved_entry_yaml=self.run_summary.get("resolved_entry_yaml"),
                save_dir=self.save_dir,
                dataset_path=self.run_summary.get("dataset_path"),
                label_key=self.run_summary.get("label_key"),
                embedding_method=self.run_summary.get("embedding_method"),
                embedding_key=self.embedding_key,
            )
            # Add a few common metrics if available
            if isinstance(self.embedding_metrics, dict):
                for k, v in self.embedding_metrics.items():
                    row[f"emb__{k}"] = _value_for_metrics_csv(v)
            if isinstance(self.classification_metrics, dict):
                cv_mean = self.classification_metrics.get("cv_mean")
                if isinstance(cv_mean, dict):
                    for k, v in cv_mean.items():
                        row[f"clf_cv__{k}"] = _value_for_metrics_csv(v)
            row = {k: _value_for_metrics_csv(v) for k, v in row.items()}
            df_row = pd.DataFrame([row])
            if os.path.exists(metrics_csv):
                df_existing = pd.read_csv(metrics_csv)
                df_out = pd.concat([df_existing, df_row], ignore_index=True, sort=False)
            else:
                df_out = df_row
            df_out.to_csv(metrics_csv, index=False)
            self.log.info("Appended run to metrics ledger: %s", metrics_csv)
        except Exception:
            self.log.exception(
                "Failed to append row to metrics_runs.csv (%s). "
                "Check permissions, disk space, or whether the file is open in another program.",
                metrics_csv,
            )


def main():
    """
    Main entry point for running an experiment from the command line.
    Sets random seed, parses config path, runs experiment, and saves timing log.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run scFM_eval experiment from YAML.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="experiment.yaml",
        help=(
            "Experiment YAML: path relative to PARAMS_PATH (bundled or SCFM_PARAMS_PATH), "
            "or an absolute path. Includes in YAML resolve relative to each file's directory."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global RNG seed (default: SCFM_EVAL_SEED environment variable or 42).",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        metavar="N",
        help=(
            "After loading, keep at most N cells (random or stratified subsample). Overrides dataset.max_cells "
            "when set to a positive integer. Omit (default) to leave dataset YAML max_cells unchanged; "
            "omit with no YAML cap to keep all loaded cells. Use --max-cells-stratify to override the stratify column."
        ),
    )
    parser.add_argument(
        "--max-cells-stratify",
        type=str,
        default=None,
        metavar="COL",
        help=(
            "When subsampling runs (dataset.max_cells and/or --max-cells), stratify by this obs column. "
            "Overrides dataset.max_cells_stratify. Use an empty argument (e.g. --max-cells-stratify '') to force "
            "uniform random sampling instead of stratifying."
        ),
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(os.environ.get("SCFM_EVAL_SEED", "42"))
    set_random_seed(seed)
    experiment = Experiment(
        args.config_path,
        seed=seed,
        max_cells=args.max_cells,
        max_cells_stratify=args.max_cells_stratify,
    )
    experiment.run()
    experiment._write_standard_reports()
    timing_df = pd.DataFrame(_timing_log)
    timing_df.to_csv(join(experiment.save_dir, 'timing.csv'))

if __name__ == '__main__':
    main()