"""
Drug response prediction evaluation module.

Evaluates foundation model embeddings on drug response prediction tasks
using the DepMap PRISM dataset. Supports multiple generalization scenarios:
1. New cell line: Predict response for unseen cell lines (drugs seen in training)
2. New drug: Predict response for unseen drugs (cell lines seen in training)
3. New MoA: Predict response for entire unseen mechanism of action classes
4. New both: Predict response for unseen cell lines AND unseen drugs (hardest)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.logs_ import get_logger

logger = get_logger()


# =============================================================================
# SPLITTING UTILITIES
# =============================================================================

def bin_values(series: pd.Series, n_bins: int = 5) -> pd.Series:
    """Create quantile bins for stratification, handling edge cases."""
    try:
        return pd.qcut(series, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Fall back to equal-width bins if quantiles fail
        return pd.cut(series, bins=n_bins, labels=False)


def get_entity_stats(obs: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Compute per-entity statistics for stratification."""
    stats = obs.groupby(entity_col, observed=True).agg(
        auc_mean=('auc', 'mean'),
        auc_std=('auc', 'std'),
        auc_median=('auc', 'median'),
        n_samples=('auc', 'count'),
        pct_resistant=('auc', lambda x: (x >= 0.9).mean()),
    ).fillna(0)
    
    stats['auc_bin'] = bin_values(stats['auc_mean'], n_bins=5)
    return stats


def split_new_cell_line(
    adata: AnnData,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by_tissue: bool = True,
    cell_line_col: str = 'depmap_id',
    tissue_col: str = 'cell_line_OncotreeLineage',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split so test set contains UNSEEN cell lines.
    All drugs in test were seen during training (on other cell lines).
    
    Args:
        adata: AnnData with drug response data
        test_size: Fraction of cell lines to hold out
        random_state: Random seed for reproducibility
        stratify_by_tissue: Whether to stratify by tissue type
        cell_line_col: Column name for cell line IDs
        tissue_col: Column name for tissue/lineage
    
    Returns:
        train_mask, test_mask: boolean arrays for indexing adata
    """
    obs = adata.obs.copy()
    cell_lines = obs[cell_line_col].unique()
    
    cl_stats = get_entity_stats(obs, cell_line_col)
    
    tissue_map = obs.groupby(cell_line_col, observed=True)[tissue_col].first()
    cl_stats['tissue'] = tissue_map
    
    if stratify_by_tissue:
        cl_stats['strat'] = cl_stats['auc_bin'].astype(str) + '_' + cl_stats['tissue'].astype(str)
        strat_counts = cl_stats['strat'].value_counts()
        rare_strats = strat_counts[strat_counts < 2].index
        cl_stats.loc[cl_stats['strat'].isin(rare_strats), 'strat'] = cl_stats.loc[
            cl_stats['strat'].isin(rare_strats), 'auc_bin'
        ].astype(str)
    else:
        cl_stats['strat'] = cl_stats['auc_bin']

    strat_series = cl_stats.loc[cell_lines, 'strat']
    strat_counts_final = strat_series.value_counts()
    single_member_strats = strat_counts_final[strat_counts_final < 2].index
    orphan_mask = strat_series.isin(single_member_strats) | strat_series.isna()
    orphan_cls = strat_series.index[orphan_mask].values
    rest_mask = ~orphan_mask
    rest_cls = strat_series.index[rest_mask].values

    if len(rest_cls) == 0:
        # All cell lines are in single-member strata; fall back to random split.
        train_cls, test_cls = train_test_split(
            cell_lines,
            test_size=test_size,
            random_state=random_state,
        )
    elif len(orphan_cls) > 0:
        # Stratify the rest; assign single-member strata to train.
        rest_strat = strat_series.loc[rest_cls].values
        train_rest, test_rest = train_test_split(
            rest_cls,
            test_size=test_size,
            stratify=rest_strat,
            random_state=random_state,
        )
        train_cls = np.concatenate([np.atleast_1d(train_rest), np.atleast_1d(orphan_cls)])
        test_cls = test_rest
    else:
        # No single-member strata; standard stratified split.
        train_cls, test_cls = train_test_split(
            cell_lines,
            test_size=test_size,
            stratify=strat_series.values,
            random_state=random_state,
        )

    train_mask = obs[cell_line_col].isin(train_cls).values
    test_mask = obs[cell_line_col].isin(test_cls).values
    
    return train_mask, test_mask


def split_new_drug(
    adata: AnnData,
    test_size: float = 0.2,
    random_state: int = 42,
    drug_col: str = 'broad_id',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split so test set contains UNSEEN drugs.
    All cell lines in test were seen during training (with other drugs).
    """
    obs = adata.obs.copy()
    drugs = obs[drug_col].unique()
    drug_stats = get_entity_stats(obs, drug_col)

    strat_series = drug_stats.loc[drugs, 'auc_bin'].astype(str)
    strat_counts_final = strat_series.value_counts()
    single_member_strats = strat_counts_final[strat_counts_final < 2].index
    orphan_mask = strat_series.isin(single_member_strats) | strat_series.isna()
    orphan_drugs = strat_series.index[orphan_mask].values
    rest_mask = ~orphan_mask
    rest_drugs = strat_series.index[rest_mask].values

    if len(rest_drugs) == 0:
        train_drugs, test_drugs = train_test_split(
            drugs,
            test_size=test_size,
            random_state=random_state,
        )
    elif len(orphan_drugs) > 0:
        rest_strat = strat_series.loc[rest_drugs].values
        train_rest, test_rest = train_test_split(
            rest_drugs,
            test_size=test_size,
            stratify=rest_strat,
            random_state=random_state,
        )
        train_drugs = np.concatenate([np.atleast_1d(train_rest), np.atleast_1d(orphan_drugs)])
        test_drugs = test_rest
    else:
        train_drugs, test_drugs = train_test_split(
            drugs,
            test_size=test_size,
            stratify=strat_series.values,
            random_state=random_state,
        )

    train_mask = obs[drug_col].isin(train_drugs).values
    test_mask = obs[drug_col].isin(test_drugs).values
    
    return train_mask, test_mask


def split_new_moa(
    adata: AnnData,
    test_moas: Optional[List[str]] = None,
    n_test_moas: int = 50,
    min_samples_per_moa: int = 500,
    random_state: int = 42,
    moa_col: str = 'moa',
    drug_col: str = 'broad_id',
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Hold out entire mechanisms of action.
    
    This is a stricter version of "new drug" — the model has never seen
    ANY drug with this mechanism.
    """
    obs = adata.obs.copy()
    
    if test_moas is None:
        moa_stats = obs.groupby(moa_col, observed=True).agg(
            auc_mean=('auc', 'mean'),
            auc_std=('auc', 'std'),
            n_samples=('auc', 'count'),
            n_drugs=(drug_col, 'nunique'),
        )
        
        valid_moas = moa_stats[
            (moa_stats['n_samples'] >= min_samples_per_moa) &
            (moa_stats['auc_std'] > 0.05)
        ].copy()
        
        valid_moas['auc_bin'] = bin_values(valid_moas['auc_mean'], n_bins=3)
        
        np.random.seed(random_state)
        test_moas = []
        for bin_val in valid_moas['auc_bin'].unique():
            bin_moas = valid_moas[valid_moas['auc_bin'] == bin_val].index.tolist()
            n_select = max(1, int(n_test_moas * len(bin_moas) / len(valid_moas)))
            test_moas.extend(np.random.choice(bin_moas, size=min(n_select, len(bin_moas)), replace=False))
    
    train_mask = (~obs[moa_col].isin(test_moas)).values
    test_mask = obs[moa_col].isin(test_moas).values
    
    return train_mask, test_mask, list(test_moas)


def split_new_drug_and_cell_line(
    adata: AnnData,
    test_cell_frac: float = 0.2,
    test_drug_frac: float = 0.2,
    random_state: int = 42,
    cell_line_col: str = 'depmap_id',
    drug_col: str = 'broad_id',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict split: test set has BOTH unseen cell lines AND unseen drugs.
    
    Matrix view:
                        Train Drugs    Test Drugs
        Train CL        [TRAIN]        [excluded]
        Test CL         [excluded]     [TEST]
    """
    obs = adata.obs.copy()

    cell_lines = obs[cell_line_col].unique()
    cl_stats = get_entity_stats(obs, cell_line_col)
    cl_strat = cl_stats.loc[cell_lines, 'auc_bin']
    cl_strat = cl_strat.values if cl_strat.value_counts().min() >= 2 else None

    train_cls, test_cls = train_test_split(
        cell_lines,
        test_size=test_cell_frac,
        stratify=cl_strat,
        random_state=random_state,
    )

    drugs = obs[drug_col].unique()
    drug_stats = get_entity_stats(obs, drug_col)
    drug_strat = drug_stats.loc[drugs, 'auc_bin']
    drug_strat = drug_strat.values if drug_strat.value_counts().min() >= 2 else None

    train_drugs, test_drugs = train_test_split(
        drugs,
        test_size=test_drug_frac,
        stratify=drug_strat,
        random_state=random_state + 1,
    )
    
    train_mask = (
        obs[cell_line_col].isin(train_cls) & obs[drug_col].isin(train_drugs)
    ).values
    test_mask = (
        obs[cell_line_col].isin(test_cls) & obs[drug_col].isin(test_drugs)
    ).values
    
    return train_mask, test_mask


# =============================================================================
# MLP REGRESSOR
# =============================================================================

class MLPRegressor(nn.Module):
    """Simple MLP for drug response regression."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def train_mlp_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    hidden_dims: List[int] = [256, 128, 64],
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 100,
    patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Tuple[MLPRegressor, StandardScaler]:
    """Train MLP regressor with early stopping."""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation data
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_t = torch.FloatTensor(X_val_scaled).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Model
    model = MLPRegressor(X_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_dataset)
        
        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, scaler


# =============================================================================
# DRUG RESPONSE EVALUATOR
# =============================================================================

class DrugResponseEvaluator:
    """
    Evaluates embeddings on drug response prediction tasks.
    
    The DepMap dataset has:
    - ~477 unique cell lines with bulk RNA profiles
    - ~1500 drugs tested across cell lines
    - AUC values measuring drug sensitivity
    
    Since each cell line appears multiple times (once per drug), we:
    1. Extract unique cell line embeddings
    2. Map embeddings back to all (cell line, drug) pairs
    3. Train predictive models on train split
    4. Evaluate on test split
    """
    
    SPLIT_TYPES = ['new_cell_line', 'new_drug', 'new_both']
    
    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        cell_line_col: str = 'depmap_id',
        drug_col: str = 'broad_id',
        target_col: str = 'auc',
        tissue_col: str = 'cell_line_OncotreeLineage',
        moa_col: str = 'moa',
        save_dir: Optional[str] = None,
        plots_dir: Optional[str] = None,
        split_types: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        model_type: Literal['mlp', 'rf', 'both'] = 'both',
        task_type: Literal['regression', 'classification', 'both'] = 'regression',
        classification_threshold: float = 0.9,
        classification_target: Literal['threshold', 'zscore'] = 'zscore',
        z_score_domain: Literal['per_drug', 'per_lineage_drug'] = 'per_lineage_drug',
        z_sensitive_threshold: float = -1.0,
        z_resistant_threshold: float = 1.0,
        z_min_std: float = 1e-6,
        mlp_epochs: int = 100,
        mlp_patience: int = 10,
        mlp_hidden_dims: List[int] = [256, 128, 64],
        rf_n_estimators: int = 100,
        use_drug_features: bool = True,
        smiles_col: str = 'smiles',
        morgan_nbits: int = 1024,
        morgan_radius: int = 2,
    ):
        """
        Initialize drug response evaluator.
        
        Args:
            adata: AnnData with drug response obs and cell embeddings in obsm
            embedding_key: Key in adata.obsm containing cell embeddings
            cell_line_col: Column name for cell line IDs
            drug_col: Column name for drug IDs  
            target_col: Column name for response values (AUC)
            tissue_col: Column name for tissue lineage
            moa_col: Column name for mechanism of action
            save_dir: Directory to save metrics
            plots_dir: Directory to save plots
            split_types: Which splits to evaluate (default: all)
            test_size: Fraction to hold out for test
            random_state: Random seed
            model_type: 'mlp', 'rf', or 'both'
            task_type: 'regression', 'classification', or 'both'
            classification_threshold: AUC threshold for resistant vs sensitive (if classification_target='threshold')
            classification_target: 'threshold' (AUC >= threshold) or 'zscore' (z-based sensitivity)
            z_score_domain: 'per_drug' or 'per_lineage_drug' for z = (AUC - mu) / sigma
            z_sensitive_threshold: z < this → sensitive (class 0)
            z_resistant_threshold: z > this → resistant (class 2); else typical (class 1)
            z_min_std: minimum sigma to avoid div by zero when computing z
            mlp_epochs: Max training epochs for MLP
            mlp_patience: Early stopping patience
            mlp_hidden_dims: MLP hidden layer dimensions
            rf_n_estimators: Number of trees for Random Forest
            use_drug_features: Whether to include Morgan fingerprint drug features
            smiles_col: Column name in obs containing SMILES strings
            morgan_nbits: Number of bits for Morgan fingerprint
            morgan_radius: Radius for Morgan fingerprint
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.cell_line_col = cell_line_col
        self.drug_col = drug_col
        self.target_col = target_col
        self.tissue_col = tissue_col
        self.moa_col = moa_col
        self.save_dir = Path(save_dir) if save_dir else None
        self.plots_dir = Path(plots_dir) if plots_dir else self.save_dir
        self.split_types = split_types or self.SPLIT_TYPES
        self.test_size = test_size
        self.random_state = random_state
        self.model_type = model_type
        self.task_type = task_type
        self.classification_threshold = classification_threshold
        self.classification_target = classification_target
        self.z_score_domain = z_score_domain
        self.z_sensitive_threshold = z_sensitive_threshold
        self.z_resistant_threshold = z_resistant_threshold
        self.z_min_std = z_min_std
        self.mlp_epochs = mlp_epochs
        self.mlp_patience = mlp_patience
        self.mlp_hidden_dims = mlp_hidden_dims
        self.rf_n_estimators = rf_n_estimators
        self.use_drug_features = use_drug_features
        self.smiles_col = smiles_col
        self.morgan_nbits = morgan_nbits
        self.morgan_radius = morgan_radius
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_dir:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate embedding exists
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm")
        
        # Build cell line -> embedding mapping
        self._build_cell_line_embeddings()
        
        # Build drug -> fingerprint mapping
        if self.use_drug_features:
            self._build_drug_fingerprints()
    
    def _build_drug_fingerprints(self) -> None:
        """
        Build mapping from drug ID to Morgan fingerprint.

        Computes Morgan fingerprint from SMILES for each unique drug.
        Drugs with missing or invalid SMILES get a zero vector.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Prefer MorganGenerator (RDKit 2022.03+) to avoid deprecation warnings
        try:
            from rdkit.Chem import rdFingerprintGenerator
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.morgan_radius, fpSize=self.morgan_nbits
            )
            use_morgan_generator = True
        except (ImportError, AttributeError):
            morgan_gen = None
            use_morgan_generator = False

        obs = self.adata.obs
        drug_smiles = obs.groupby(self.drug_col, observed=True)[self.smiles_col].first()
        unique_drugs = drug_smiles.index.values
        n_drugs = len(unique_drugs)

        fps = np.zeros((n_drugs, self.morgan_nbits), dtype=np.float32)
        n_invalid = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for i, drug_id in enumerate(unique_drugs):
                smi = drug_smiles.loc[drug_id]
                if pd.isna(smi) or not isinstance(smi, str) or len(smi.strip()) == 0:
                    n_invalid += 1
                    continue
                # Clean SMILES: strip whitespace and trailing commas (CSV artifact),
                # then take only the first entry if comma-separated.
                smi = smi.strip().rstrip(',').strip()
                if ',' in smi:
                    smi = smi.split(',')[0].strip()
                if len(smi) == 0:
                    n_invalid += 1
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    n_invalid += 1
                    continue
                if use_morgan_generator:
                    fp = morgan_gen.GetFingerprint(mol)
                else:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.morgan_radius, nBits=self.morgan_nbits
                    )
                fps[i] = np.array(fp, dtype=np.float32)

        self.drug_to_idx = {d: i for i, d in enumerate(unique_drugs)}
        self.unique_drug_fps = fps

        logger.info(
            f"Built drug fingerprints: {n_drugs} unique drugs, "
            f"{self.morgan_nbits}-bit Morgan (radius={self.morgan_radius}), "
            f"{n_invalid} invalid/missing SMILES"
        )

    def _build_cell_line_embeddings(self):
        """
        Build mapping from cell line ID to embedding.
        
        Since adata has one row per (cell line, drug) pair but embeddings
        are cell-line specific, we need to deduplicate.
        """
        # Get unique cell lines and their first occurrence indices
        cell_lines = self.adata.obs[self.cell_line_col].values
        unique_cls, first_indices = np.unique(cell_lines, return_index=True)
        
        # Extract embeddings for unique cell lines
        embeddings = self.adata.obsm[self.embedding_key]
        unique_embeddings = embeddings[first_indices]
        
        # Create mapping
        self.cell_line_to_idx = {cl: i for i, cl in enumerate(unique_cls)}
        self.unique_embeddings = unique_embeddings
        
        logger.info(f"Built embedding mapping: {len(unique_cls)} unique cell lines, "
                   f"embedding dim: {unique_embeddings.shape[1]}")
    
    def _get_features_and_targets(
        self, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature matrix and targets for samples specified by mask.

        When use_drug_features is True, X is [cell_emb | drug_fp].
        
        Args:
            mask: Boolean mask for selecting samples
            
        Returns:
            X: Feature matrix (n_samples, cell_dim [+ morgan_nbits])
            y: Target values (n_samples,)
        """
        obs = self.adata.obs.iloc[mask]
        
        # Get cell line embeddings for each sample
        cell_lines = obs[self.cell_line_col].values
        cl_indices = np.array([self.cell_line_to_idx[cl] for cl in cell_lines])
        X_cell = self.unique_embeddings[cl_indices]

        if self.use_drug_features:
            # Get drug fingerprints for each sample
            drugs = obs[self.drug_col].values
            drug_indices = np.array([self.drug_to_idx[d] for d in drugs])
            X_drug = self.unique_drug_fps[drug_indices]
            X = np.hstack([X_cell, X_drug])
        else:
            X = X_cell
        
        # Get targets
        y = obs[self.target_col].values.astype(np.float32)
        
        return X, y

    def _get_zscore_class_labels(
        self,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute binary sensitivity class labels from z-scored AUC (train stats only).

        z = (AUC - mu) / sigma, computed per z_score_domain (per_drug or
        per_lineage_drug). Classes: 0 = sensitive (z < z_sensitive_threshold),
        1 = resistant (z >= z_sensitive_threshold).
        """
        obs = self.adata.obs
        auc_col = self.target_col
        train_obs = obs.iloc[np.flatnonzero(train_mask)].copy()
        test_obs = obs.iloc[np.flatnonzero(test_mask)].copy()

        if self.z_score_domain == 'per_lineage_drug':
            group_cols = [self.tissue_col, self.drug_col]
        else:
            group_cols = [self.drug_col]

        train_stats = train_obs.groupby(group_cols, observed=True)[auc_col].agg(
            ['mean', 'std']
        ).reset_index()
        train_stats.columns = list(group_cols) + ['mu', 'sigma']
        train_stats['sigma'] = train_stats['sigma'].clip(lower=self.z_min_std)
        train_stats.loc[train_stats['sigma'].isna(), 'sigma'] = self.z_min_std

        global_mu = train_obs[auc_col].mean()
        train_std = train_obs[auc_col].std()
        global_std = max(
            np.nan_to_num(train_std, nan=self.z_min_std),
            self.z_min_std,
        )

        # Vectorized: merge train_obs with train_stats to get mu, sigma per row
        train_merged = train_obs[group_cols + [auc_col]].merge(
            train_stats, on=group_cols, how='left'
        )
        train_merged['mu'] = train_merged['mu'].fillna(global_mu)
        train_merged['sigma'] = train_merged['sigma'].fillna(global_std).clip(lower=self.z_min_std)
        z_train = (train_merged[auc_col].values - train_merged['mu'].values) / train_merged['sigma'].values
        y_train_class = (z_train >= self.z_sensitive_threshold).astype(int)  # 0 = sensitive, 1 = resistant

        test_merged = test_obs[group_cols + [auc_col]].merge(
            train_stats, on=group_cols, how='left'
        )
        test_merged['mu'] = test_merged['mu'].fillna(global_mu)
        test_merged['sigma'] = test_merged['sigma'].fillna(global_std).clip(lower=self.z_min_std)
        z_test = (test_merged[auc_col].values - test_merged['mu'].values) / test_merged['sigma'].values
        y_test_class = (z_test >= self.z_sensitive_threshold).astype(int)  # 0 = sensitive, 1 = resistant

        return y_train_class, y_test_class
    
    def _create_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Create all requested train/test splits."""
        splits = {}
        
        for split_type in self.split_types:
            if split_type == 'new_cell_line':
                train_mask, test_mask = split_new_cell_line(
                    self.adata, self.test_size, self.random_state,
                    cell_line_col=self.cell_line_col, tissue_col=self.tissue_col
                )
            elif split_type == 'new_drug':
                train_mask, test_mask = split_new_drug(
                    self.adata, self.test_size, self.random_state,
                    drug_col=self.drug_col
                )
            elif split_type == 'new_both':
                train_mask, test_mask = split_new_drug_and_cell_line(
                    self.adata, self.test_size, self.test_size,
                    self.random_state, self.cell_line_col, self.drug_col
                )
            else:
                logger.warning(f"Unknown split type: {split_type}, skipping")
                continue
            
            splits[split_type] = (train_mask, test_mask)
            logger.info(f"Split '{split_type}': train={train_mask.sum()}, test={test_mask.sum()}")
        
        return splits
    
    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        # Correlation metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                metrics['pearson_r'], metrics['pearson_p'] = pearsonr(y_true, y_pred)
            except:
                metrics['pearson_r'], metrics['pearson_p'] = np.nan, np.nan
            try:
                metrics['spearman_r'], metrics['spearman_p'] = spearmanr(y_true, y_pred)
            except:
                metrics['spearman_r'], metrics['spearman_p'] = np.nan, np.nan
        
        return metrics
    
    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute classification metrics (binary or multiclass)."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        if y_prob is not None:
            n_classes = y_prob.shape[1]
            try:
                if n_classes == 2:
                    metrics['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auprc'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    metrics['auroc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
                    # One-vs-rest AUPRC: binarize and average per class
                    classes = np.arange(n_classes)
                    y_true_bin = label_binarize(y_true, classes=classes)
                    ap_scores = []
                    for c in classes:
                        if y_true_bin[:, c].sum() > 0:
                            ap_scores.append(
                                average_precision_score(
                                    y_true_bin[:, c], y_prob[:, c]
                                )
                            )
                    metrics['auprc'] = np.mean(ap_scores) if ap_scores else np.nan
            except Exception:
                metrics['auroc'] = np.nan
                metrics['auprc'] = np.nan

        return metrics

    def _plot_actual_vs_pred(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        path: Path,
        split_name: str,
        model_name: str,
        pearson_r: float = np.nan,
        r2: float = np.nan,
    ) -> None:
        """Plot actual vs predicted AUC (regression) and save to path."""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, alpha=0.3, s=5)
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, 'k--', lw=1, label='Perfect prediction')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Actual AUC')
        ax.set_ylabel('Predicted AUC')
        ax.set_title(f'{split_name} — {model_name}\nPearson r = {pearson_r:.4f}, R² = {r2:.4f}')
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _train_and_evaluate_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        task: str,
        plot_path: Optional[Path] = None,
        split_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Train a model and evaluate on test set.

        For classification, y_train/y_test are already class labels (from
        threshold or z-score); no binarization here.
        """
        results = {}
        
        if model_name == 'mlp':
            if task == 'regression':
                # Split train into train/val for early stopping
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=self.random_state
                )
                model, scaler = train_mlp_regressor(
                    X_tr, y_tr, X_val, y_val,
                    hidden_dims=self.mlp_hidden_dims,
                    epochs=self.mlp_epochs,
                    patience=self.mlp_patience,
                )
                
                # Predict
                device = next(model.parameters()).device
                X_test_scaled = scaler.transform(X_test)
                X_test_t = torch.FloatTensor(X_test_scaled).to(device)
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_t).cpu().numpy()
                
                results = self._evaluate_regression(y_test, y_pred)
                if plot_path and split_name:
                    self._plot_actual_vs_pred(
                        y_test, y_pred, plot_path, split_name, model_name,
                        pearson_r=results['pearson_r'], r2=results['r2'],
                    )
            else:
                # Classification not implemented for MLP yet
                logger.warning("MLP classification not yet implemented, skipping")
                return {}
        
        elif model_name == 'rf':
            if task == 'regression':
                model = RandomForestRegressor(
                    n_estimators=self.rf_n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results = self._evaluate_regression(y_test, y_pred)
                if plot_path and split_name:
                    self._plot_actual_vs_pred(
                        y_test, y_pred, plot_path, split_name, model_name,
                        pearson_r=results['pearson_r'], r2=results['r2'],
                    )
            else:
                model = RandomForestClassifier(
                    n_estimators=self.rf_n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight='balanced',
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                results = self._evaluate_classification(y_test, y_pred, y_prob)
        
        return results
    
    def evaluate(self) -> Dict[str, pd.DataFrame]:
        """
        Run full evaluation across all splits and models.
        
        Returns:
            Dict mapping split names to DataFrames with metrics
        """
        logger.info("=" * 60)
        logger.info("DRUG RESPONSE PREDICTION EVALUATION")
        logger.info(f"Embedding key: {self.embedding_key}")
        logger.info(f"Splits: {self.split_types}")
        logger.info(f"Models: {self.model_type}")
        logger.info(f"Tasks: {self.task_type}")
        logger.info("=" * 60)
        
        # Create splits
        splits = self._create_splits()
        
        # Determine which models and tasks to run
        models = ['mlp', 'rf'] if self.model_type == 'both' else [self.model_type]
        tasks = ['regression', 'classification'] if self.task_type == 'both' else [self.task_type]
        
        all_results = {}
        
        for split_name, (train_mask, test_mask) in splits.items():
            logger.info(f"\n--- Evaluating split: {split_name} ---")
            
            # Get features and AUC targets
            X_train, y_train_auc = self._get_features_and_targets(train_mask)
            X_test, y_test_auc = self._get_features_and_targets(test_mask)
            
            logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            
            split_results = []
            
            for model_name in models:
                for task in tasks:
                    if task == 'regression':
                        y_train, y_test = y_train_auc, y_test_auc
                    else:
                        if self.classification_target == 'zscore':
                            y_train, y_test = self._get_zscore_class_labels(
                                train_mask, test_mask
                            )
                        else:
                            y_train = (
                                y_train_auc >= self.classification_threshold
                            ).astype(int)
                            y_test = (
                                y_test_auc >= self.classification_threshold
                            ).astype(int)
                    
                    logger.info(f"Training {model_name} for {task}...")
                    
                    try:
                        plot_path = None
                        if (
                            self.plots_dir
                            and task == 'regression'
                        ):
                            plot_path = self.plots_dir / f"actual_vs_pred_{split_name}_{model_name}.png"
                        metrics = self._train_and_evaluate_model(
                            X_train, y_train, X_test, y_test, model_name, task,
                            plot_path=plot_path,
                            split_name=split_name,
                        )
                        
                        if metrics:
                            metrics['model'] = model_name
                            metrics['task'] = task
                            metrics['split'] = split_name
                            metrics['n_train'] = len(y_train)
                            metrics['n_test'] = len(y_test)
                            split_results.append(metrics)
                            
                            # Log key metrics
                            if task == 'regression':
                                logger.info(f"  {model_name}/{task}: R²={metrics['r2']:.4f}, "
                                          f"Pearson r={metrics['pearson_r']:.4f}, "
                                          f"RMSE={metrics['rmse']:.4f}")
                            else:
                                logger.info(f"  {model_name}/{task}: Acc={metrics['accuracy']:.4f}, "
                                          f"AUROC={metrics.get('auroc', np.nan):.4f}")
                    
                    except Exception as e:
                        logger.error(f"Error training {model_name} for {task}: {e}")
            
            if split_results:
                results_df = pd.DataFrame(split_results)
                all_results[split_name] = results_df
                
                # Save per-split results
                if self.save_dir:
                    results_df.to_csv(self.save_dir / f'drug_response_{split_name}.csv', index=False)
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results.values(), ignore_index=True)
            
            if self.save_dir:
                combined_df.to_csv(self.save_dir / 'drug_response_all.csv', index=False)
                
                # Create summary pivot table
                if 'regression' in tasks:
                    summary = combined_df[combined_df['task'] == 'regression'].pivot_table(
                        index='split',
                        columns='model',
                        values=['pearson_r', 'r2', 'rmse'],
                        aggfunc='first'
                    )
                    summary.to_csv(self.save_dir / 'drug_response_summary.csv')
            
            logger.info("\n" + "=" * 60)
            logger.info("EVALUATION COMPLETE")
            logger.info("=" * 60)
            
            # Print summary
            for split_name, df in all_results.items():
                logger.info(f"\n{split_name}:")
                for _, row in df.iterrows():
                    if row['task'] == 'regression':
                        logger.info(f"  {row['model']}: R²={row['r2']:.4f}, "
                                  f"r={row['pearson_r']:.4f}")
        
        return all_results


def validate_split(
    adata: AnnData,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    split_name: str,
    cell_line_col: str = 'depmap_id',
    drug_col: str = 'broad_id',
) -> pd.DataFrame:
    """Validate split quality and check for leakage."""
    obs = adata.obs
    
    train_cls = set(obs.loc[train_mask, cell_line_col].unique())
    test_cls = set(obs.loc[test_mask, cell_line_col].unique())
    train_drugs = set(obs.loc[train_mask, drug_col].unique())
    test_drugs = set(obs.loc[test_mask, drug_col].unique())
    
    results = {
        'split': split_name,
        'n_train': train_mask.sum(),
        'n_test': test_mask.sum(),
        'train_auc_mean': obs.loc[train_mask, 'auc'].mean(),
        'test_auc_mean': obs.loc[test_mask, 'auc'].mean(),
        'train_auc_std': obs.loc[train_mask, 'auc'].std(),
        'test_auc_std': obs.loc[test_mask, 'auc'].std(),
        'train_pct_resistant': (obs.loc[train_mask, 'auc'] >= 0.9).mean(),
        'test_pct_resistant': (obs.loc[test_mask, 'auc'] >= 0.9).mean(),
        'n_train_cell_lines': len(train_cls),
        'n_test_cell_lines': len(test_cls),
        'n_train_drugs': len(train_drugs),
        'n_test_drugs': len(test_drugs),
        'cell_line_overlap': len(train_cls & test_cls),
        'drug_overlap': len(train_drugs & test_drugs),
    }
    
    return pd.DataFrame([results])
