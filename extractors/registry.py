"""
Extractor Registry - Central dispatcher for all embedding extractors.

This module provides a unified interface for:
1. Getting extractor instances (handles both inline and subprocess execution)
2. Managing model metadata (env requirements, checkpoint paths, etc.)
3. Dispatching to separate environments when needed

Usage:
    from extractors.registry import ExtractorRegistry
    
    # Get extractor (handles env switching automatically)
    extractor = ExtractorRegistry.get_extractor("stack", config)
    embeddings = extractor.fit_transform(data_loader)
    
    # Or use the high-level API
    embeddings = ExtractorRegistry.extract_embeddings(
        model_name="stack",
        adata_path="/path/to/data.h5ad",
        params=config,
    )
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Base paths (CHECKPOINTS_BASE from setup_path so it reads .env)
import os as _os
from setup_path import CHECKPOINTS_BASE

EXTRACTORS_DIR = Path(__file__).parent


# =============================================================================
# Model Registry Configuration
# =============================================================================
# Each model entry contains:
#   - class_name: The extractor class name
#   - module: Python module path (relative to extractors/)
#   - needs_separate_env: Whether model requires its own conda/venv environment
#   - needs_training: Whether model trains on data (vs loading checkpoint)
#   - default_params: Default parameters for the model
#   - checkpoint_dir: Subdirectory under CHECKPOINTS_BASE (if applicable)

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # =========================================================================
    # Foundation Models (checkpoint-based, some need separate envs)
    # =========================================================================
    "stack": {
        "class_name": "STACKExtractor",
        "module": "stack.extract",
        "needs_separate_env": True,
        "needs_training": False,
        "checkpoint_dir": "STACK",
        "default_params": {
            "checkpoint_path": str(CHECKPOINTS_BASE / "STACK" / "bc_large_aligned.ckpt"),
            "genelist_path": str(CHECKPOINTS_BASE / "STACK" / "basecount_1000per_15000max.pkl"),
            "batch_size": 32,
            "device": "auto",
        },
    },
    "scgpt": {
        "class_name": "scGPTExtractor",
        "module": "scgpt.extract",
        "needs_separate_env": True,
        "needs_training": False,
        "checkpoint_dir": "scGPT",
        "default_params": {
            "model_dir": str(CHECKPOINTS_BASE / "scGPT"),
            "batch_size": 32,
            "device": "auto",
        },
    },
    "geneformer": {
        "class_name": "GeneformerExtractor",
        "module": "geneformer.extract",
        "needs_separate_env": True,
        "needs_training": False,
        "checkpoint_dir": "Geneformer-V2-104M",
        "default_params": {
            "model_dir": str(CHECKPOINTS_BASE / "Geneformer-V2-104M"),
            "model_name": "Geneformer-V2-104M",
            "batch_size": 32,
        },
    },
    "state": {
        "class_name": "STATEExtractor",
        "module": "state.extract",
        "needs_separate_env": True,
        "needs_training": False,
        "checkpoint_dir": "STATE",
        "default_params": {
            "model_dir": str(CHECKPOINTS_BASE / "STATE" / "SE-600M"),
            "batch_size": 32,
        },
    },
    "uce": {
        "class_name": "UCEExtractor",
        "module": "uce.extract",
        "needs_separate_env": True,
        "needs_training": False,
        "checkpoint_dir": "UCE",
        "default_params": {
            "model_dir": str(CHECKPOINTS_BASE / "UCE"),
            "batch_size": 32,
        },
    },
    "aido": {
        "class_name": "AIDOExtractor",
        "module": "aido.extract",
        "needs_separate_env": True,
        "needs_training": False,
        "checkpoint_dir": "AIDO",
        "default_params": {
            "model_dir": str(CHECKPOINTS_BASE / "AIDO"),
        },
    },
    "scconcept": {
        "class_name": "scConceptExtractor",
        "module": "scconcept.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "default_params": {
            "model_name": "Corpus-30M",
            "cache_dir": "./cache/",
        },
    },
    
    # =========================================================================
    # Foundation Models (checkpoint-based, base env compatible)
    # =========================================================================
    "scfoundation": {
        "class_name": "scFoundationExtractor",
        "module": "scfoundation.extract",
        "needs_separate_env": True,
        "needs_training": False,
        "checkpoint_dir": "scFoundation",
        "default_params": {
            "model_path": str(CHECKPOINTS_BASE / "scFoundation" / "models.ckpt"),
            "gene_index_path": str(CHECKPOINTS_BASE / "scFoundation" / "OS_scRNA_gene_index.19264.tsv"),
            "batch_size": 32,
        },
    },
    "scimilarity": {
        "class_name": "SCimilarityExtractor",
        "module": "scimilarity.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "checkpoint_dir": "SCSimilarity",
        "default_params": {
            "model_path": str(CHECKPOINTS_BASE / "SCSimilarity" / "model_v1.1"),
        },
    },
    "laplacian_ae": {
        "class_name": "LaplacianAEExtractor",
        "module": "laplacian_ae.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "checkpoint_dir": "LaplacianAE",
        "default_params": {
            "checkpoint_path": str(CHECKPOINTS_BASE / "LaplacianAE" / "model.ckpt"),
            "gene_order_file": str(CHECKPOINTS_BASE / "SCSimilarity" / "model_v1.1" / "gene_order.tsv"),
        },
    },
    "cellplm": {
        "class_name": "CellPLMExtractor",
        "module": "cellplm.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "checkpoint_dir": "CellPLM",
        "default_params": {
            "pretrain_prefix": "CellPLM",
            "pretrain_directory": str(CHECKPOINTS_BASE / "CellPLM"),
        },
    },
    "cellfm": {
        "class_name": "CellFMExtractor",
        "module": "cellfm.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "checkpoint_dir": "CellFM",
        "default_params": {
            "model_path": str(CHECKPOINTS_BASE / "CellFM"),
            "model_name": "80m",
        },
    },
    "scbert": {
        "class_name": "scBERTExtractor",
        "module": "scbert.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "checkpoint_dir": "scBERT",
        "default_params": {
            "checkpoint_path": str(CHECKPOINTS_BASE / "scBERT" / "model.pt"),
        },
    },
    "omnicell": {
        "class_name": "OmnicellExtractor",
        "module": "omnicell.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "checkpoint_dir": "Omnicell",
        "default_params": {
            "checkpoint_path": _os.environ.get("OMNICELL_CHECKPOINT_PATH")
            or str(
                Path(_os.environ.get("OMNICELL_CHECKPOINTS_BASE", str(CHECKPOINTS_BASE)))
                / "Omnicell" / "omnicell_checkpoint_epoch27.pt"
            ),
            "base_dir": _os.environ.get("OMNICELL_BASE_DIR", ""),
            "config": "obs",
            "batch_size": 4096,
            "device": "auto",
            "gene_types": "feature_name",
            "load_avg": False,
        },
    },

    # =========================================================================
    # Trainable Models (train on data, save checkpoints)
    # =========================================================================
    "concord": {
        "class_name": "ConcordExtractor",
        "module": "concord.extract",
        "needs_separate_env": False,
        "needs_training": True,
        "checkpoint_dir": "CONCORD",
        "default_params": {
            "n_epochs": 15,
            "latent_dim": 100,
            "batch_size": 256,
            "lr": 1e-2,
            "variant": "concord_hcl",  # Default variant
        },
    },
    # CONCORD variants (same extractor, different params)
    "concord_hcl": {
        "class_name": "ConcordExtractor",
        "module": "concord.extract",
        "needs_separate_env": False,
        "needs_training": True,
        "checkpoint_dir": "CONCORD_HCL",
        "default_params": {
            "variant": "concord_hcl",  # Hard Contrastive Learning
            "n_epochs": 20,
            "latent_dim": 30,
            "batch_size": 32,
        },
    },
    "concord_knn": {
        "class_name": "ConcordExtractor",
        "module": "concord.extract",
        "needs_separate_env": False,
        "needs_training": True,
        "checkpoint_dir": "CONCORD_KNN",
        "default_params": {
            "variant": "concord_knn",  # k-NN sampling
            "n_epochs": 20,
            "latent_dim": 30,
            "batch_size": 32,
        },
    },
    "contrastive": {
        "class_name": "ConcordExtractor",
        "module": "concord.extract",
        "needs_separate_env": False,
        "needs_training": True,
        "checkpoint_dir": "CONTRASTIVE",
        "default_params": {
            "variant": "contrastive",  # Naive contrastive (no batch)
            "n_epochs": 20,
            "latent_dim": 30,
            "batch_size": 32,
        },
    },
    "scvi": {
        "class_name": "scVIExtractor",
        "module": "scvi.extract",
        "needs_separate_env": False,
        "needs_training": True,
        "checkpoint_dir": "scVI",
        "default_params": {
            "n_layers": 2,
            "n_latent": 10,
            "gene_likelihood": "zinb",
            "max_epochs": 400,
        },
    },
    
    # =========================================================================
    # Baseline Methods (no checkpoint, compute directly)
    # =========================================================================
    "pca": {
        "class_name": "PCAExtractor",
        "module": "pca.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "default_params": {
            "n_components": 50,
        },
    },
    "hvg": {
        "class_name": "HVGExtractor",
        "module": "hvg.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "default_params": {
            "n_top_genes": 2000,
            "flavor": "seurat",
        },
    },
    "baseline": {
        "class_name": "BaselineExtractor",
        "module": "baseline.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "default_params": {
            "baseline_type": "no_integration",
        },
    },
    
    # =========================================================================
    # Integration Baselines (classical batch-correction methods, run as extractors)
    # These wrap evaluation.baseline_embeddings.create_baseline_embedding so that
    # each integration method can be run as a first-class method with its own YAML.
    # =========================================================================
    "harmony": {
        "class_name": "IntegrationBaselineExtractor",
        "module": "integration_baseline.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "default_params": {
            "integration_method": "harmony",
            "n_comps": 50,
        },
    },
    "bbknn": {
        "class_name": "IntegrationBaselineExtractor",
        "module": "integration_baseline.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "default_params": {
            "integration_method": "bbknn",
            "n_comps": 50,
        },
    },
    "scanorama": {
        "class_name": "IntegrationBaselineExtractor",
        "module": "integration_baseline.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "default_params": {
            "integration_method": "scanorama",
            "n_comps": 50,
        },
    },
    "pca_qc": {
        "class_name": "IntegrationBaselineExtractor",
        "module": "integration_baseline.extract",
        "needs_separate_env": False,
        "needs_training": False,
        "default_params": {
            "integration_method": "pca_qc",
            "n_comps": 50,
        },
    },
}


class ExtractorRegistry:
    """Central registry for all embedding extractors.
    
    Handles:
    - Loading extractor classes (inline or via subprocess)
    - Environment switching for models with separate envs
    - Checkpoint management for trainable models
    """
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List all available model names."""
        return list(MODEL_REGISTRY.keys())
    
    @classmethod
    def _resolve_canonical_model_name(cls, model_name: str) -> str:
        """Resolve AB/variant names to the canonical registry key.

        e.g. scconcept_hybrid, scconcept_hyperbolic -> scconcept (same extractor).
        """
        key = model_name.lower()
        if key in MODEL_REGISTRY:
            return key
        if key.startswith("scconcept"):
            return "scconcept"
        if key.startswith("stack_"):
            return "stack"
        return key

    @classmethod
    def get_model_info(cls, model_name: str) -> dict[str, Any]:
        """Get metadata for a model.

        AB-test variants (e.g. scconcept_hybrid) resolve to the base model (scconcept).

        Args:
            model_name: Name of the model (case-insensitive)

        Returns:
            Model metadata dictionary

        Raises:
            ValueError: If model not found
        """
        model_name = model_name.lower()
        canonical = cls._resolve_canonical_model_name(model_name)
        if canonical not in MODEL_REGISTRY:
            available = ", ".join(cls.list_models())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        return MODEL_REGISTRY[canonical].copy()
    
    @classmethod
    def needs_separate_env(cls, model_name: str) -> bool:
        """Check if model requires a separate environment."""
        return cls.get_model_info(model_name).get("needs_separate_env", False)
    
    @classmethod
    def needs_training(cls, model_name: str) -> bool:
        """Check if model requires training on data."""
        return cls.get_model_info(model_name).get("needs_training", False)
    
    @classmethod
    def get_env_python(cls, model_name: str) -> Path | None:
        """Get path to model's environment Python executable.
        
        Uses the canonical model name so that variants (e.g. ``stack_small_data``)
        resolve to the base model's env (``extractors/stack/env``).
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to python executable, or None if using base env
        """
        model_name = model_name.lower()
        if not cls.needs_separate_env(model_name):
            return None
        
        canonical = cls._resolve_canonical_model_name(model_name)
        env_path = EXTRACTORS_DIR / canonical / "env" / "bin" / "python"
        if env_path.exists():
            return env_path
        return None
    
    @classmethod
    def get_checkpoint_path(cls, model_name: str, dataset_name: str | None = None) -> Path | None:
        """Get checkpoint path for a model.
        
        For pretrained models: returns the default checkpoint
        For trainable models: returns dataset-specific checkpoint if it exists
        
        Args:
            model_name: Name of the model
            dataset_name: Dataset name (for trainable models)
            
        Returns:
            Path to checkpoint, or None if not found
        """
        info = cls.get_model_info(model_name)
        checkpoint_dir = info.get("checkpoint_dir")
        
        if not checkpoint_dir:
            return None
        
        base_dir = CHECKPOINTS_BASE / checkpoint_dir
        
        if info.get("needs_training") and dataset_name:
            # Look for dataset-specific checkpoint
            checkpoint_path = base_dir / f"{dataset_name}_best.ckpt"
            if checkpoint_path.exists():
                return checkpoint_path
            # Also check for .pt extension
            checkpoint_path = base_dir / f"{dataset_name}_best.pt"
            if checkpoint_path.exists():
                return checkpoint_path
        
        return base_dir if base_dir.exists() else None
    
    @classmethod
    def get_extractor(
        cls,
        model_name: str,
        config: dict[str, Any] | None = None,
    ):
        """Get an extractor instance for the given model.
        
        For models that need separate environments, this returns a proxy
        that will dispatch to subprocess when fit_transform is called.
        
        Args:
            model_name: Name of the model
            config: Configuration dict (merged with defaults)
            
        Returns:
            Extractor instance
        """
        model_name = model_name.lower()
        info = cls.get_model_info(model_name)
        
        # Merge config with defaults
        params = info["default_params"].copy()
        if config:
            # Handle nested params structure from YAML
            if "params" in config:
                params.update(config["params"])
            else:
                params.update(config)
        
        # Add method name
        params["method"] = model_name.upper()
        
        # Check if we need subprocess dispatch
        if cls.needs_separate_env(model_name):
            env_python = cls.get_env_python(model_name)
            if env_python and env_python.exists():
                logger.info(f"Model {model_name} requires separate env: {env_python}")
                return SubprocessExtractor(model_name, params, env_python)
        
        # Load extractor class inline
        return cls._load_extractor_class(model_name, params)
    
    @classmethod
    def _load_extractor_class(
        cls,
        model_name: str,
        params: dict[str, Any],
    ):
        """Load and instantiate an extractor class.
        
        Args:
            model_name: Name of the model
            params: Configuration parameters
            
        Returns:
            Extractor instance
        """
        import importlib
        
        info = cls.get_model_info(model_name)
        module_path = f"extractors.{info['module']}"
        class_name = info["class_name"]
        
        try:
            module = importlib.import_module(module_path)
            extractor_class = getattr(module, class_name)
            return extractor_class(params)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {class_name} from {module_path}. "
                f"Error: {e}. "
                f"Make sure the extractor module exists at extractors/{info['module'].replace('.', '/')}.py"
            )
    
    @classmethod
    def extract_embeddings(
        cls,
        model_name: str,
        adata_path: str | Path,
        output_path: str | Path | None = None,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """High-level API to extract embeddings.
        
        Handles environment switching, data loading, and caching.
        
        Args:
            model_name: Name of the model
            adata_path: Path to input h5ad file
            output_path: Optional path to save embeddings (.npy)
            params: Additional parameters
            
        Returns:
            Embeddings array
        """
        adata_path = Path(adata_path)
        
        if cls.needs_separate_env(model_name):
            # Use subprocess for separate env
            return cls._extract_subprocess(model_name, adata_path, output_path, params)
        else:
            # Load inline
            import scanpy as sc
            
            extractor = cls.get_extractor(model_name, params)
            adata = sc.read_h5ad(adata_path)
            
            # Create a mock loader with adata and path
            class MockLoader:
                def __init__(self, adata, path):
                    self.adata = adata
                    self.path = str(path)
            
            loader = MockLoader(adata, adata_path)
            embeddings = extractor.fit_transform(loader)
            
            if output_path:
                np.save(output_path, embeddings)
            
            return embeddings
    
    @classmethod
    def _extract_subprocess(
        cls,
        model_name: str,
        adata_path: Path,
        output_path: Path | None,
        params: dict[str, Any] | None,
    ) -> np.ndarray:
        """Extract embeddings using subprocess (for separate env models).
        
        Args:
            model_name: Name of the model
            adata_path: Path to input h5ad file
            output_path: Path to save embeddings
            params: Additional parameters
            
        Returns:
            Embeddings array
        """
        canonical = cls._resolve_canonical_model_name(model_name)
        env_python = cls.get_env_python(model_name)
        if not env_python or not env_python.exists():
            raise RuntimeError(
                f"Model {model_name} requires separate environment but "
                f"env not found at: {EXTRACTORS_DIR / canonical / 'env'}"
            )
        
        # Prepare output path
        if output_path is None:
            temp_output = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
            output_path = Path(temp_output.name)
            temp_output.close()
            cleanup_output = True
        else:
            output_path = Path(output_path)
            cleanup_output = False
        
        # Build command -- use canonical name for the script path (e.g. stack_small_data -> stack)
        extract_script = EXTRACTORS_DIR / canonical / "extract.py"
        cmd = [
            str(env_python),
            str(extract_script),
            "--input", str(adata_path),
            "--output", str(output_path),
        ]
        
        # Add params
        info = cls.get_model_info(model_name)
        merged_params = info["default_params"].copy()
        if params:
            merged_params.update(params.get("params", params))
        
        # Handle model name mapping for scGPT (YAML uses "model" but CLI expects "model_name")
        if model_name == "scgpt" and "model" in merged_params and "model_name" not in merged_params:
            merged_params["model_name"] = merged_params.pop("model")
            logger.info(f"Mapped 'model' param to 'model_name' for scGPT: {merged_params['model_name']}")
        
        # Convert params to CLI args
        for key, value in merged_params.items():
            if key in ("method", "save_dir"):
                continue
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running subprocess: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Subprocess stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Subprocess stderr: {result.stderr}")
            
            # Load embeddings
            embeddings = np.load(output_path)
            
            # Delete temp output file immediately after loading
            if cleanup_output and output_path.exists():
                try:
                    output_path.unlink()
                    logger.debug(f"Deleted temp output file: {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp output file {output_path}: {e}")
            
            return embeddings
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Subprocess failed: {e.stderr}")
            # Clean up temp output file on error
            if cleanup_output and output_path.exists():
                try:
                    output_path.unlink()
                    logger.debug(f"Deleted temp output file after error: {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp output file {output_path}: {e}")
            raise RuntimeError(f"Extraction failed for {model_name}: {e.stderr}")


class SubprocessExtractor:
    """Proxy extractor that dispatches to subprocess.
    
    This class mimics the BaseExtractor interface but runs extraction
    in a subprocess using the model's separate environment.
    """
    
    def __init__(
        self,
        model_name: str,
        params: dict[str, Any],
        env_python: Path,
    ):
        self.model_name = model_name
        self.params = params
        self.env_python = env_python
        self.method = params.get("method", model_name.upper())
    
    def fit_transform(self, loader) -> np.ndarray:
        """Extract embeddings via subprocess.
        
        Args:
            loader: Data loader with .adata and .path attributes
            
        Returns:
            Embeddings array
        """
        # ALWAYS save the modified adata to a temp file
        # This ensures that gene_symbol column (from ensure_both_gene_identifiers())
        # is passed to the subprocess, not just the original file
        temp_input = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
        adata_path = temp_input.name
        temp_input.close()
        
        logger.info(f"Saving modified adata to temp file (preserves gene_symbol column)...")
        loader.adata.write_h5ad(adata_path)
        logger.info(f"Saved adata to temp file: {adata_path}")
        
        try:
            embeddings = ExtractorRegistry._extract_subprocess(
                self.model_name,
                Path(adata_path),
                None,  # temp output
                {"params": self.params},
            )
            return embeddings
        finally:
            # Delete temp file immediately after extraction completes
            temp_path = Path(adata_path)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Deleted temp file: {adata_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {adata_path}: {e}")


# Convenience function
def get_extractor(model_name: str, config: dict[str, Any] | None = None):
    """Convenience function to get an extractor.
    
    Args:
        model_name: Name of the model
        config: Configuration dict
        
    Returns:
        Extractor instance
    """
    return ExtractorRegistry.get_extractor(model_name, config)


def extract_embeddings(
    model_name: str,
    adata_path: str | Path,
    output_path: str | Path | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Convenience function to extract embeddings.
    
    Args:
        model_name: Name of the model
        adata_path: Path to input h5ad file
        output_path: Optional path to save embeddings
        params: Additional parameters
        
    Returns:
        Embeddings array
    """
    return ExtractorRegistry.extract_embeddings(
        model_name, adata_path, output_path, params
    )
