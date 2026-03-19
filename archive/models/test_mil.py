# ===========================
# Testbed for MILExperiment
# ===========================
import numpy as np
import pandas as pd
import torch
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mil_experiment import MILExperiment


# --------- Mini AnnData fallback (used if 'anndata' isn't installed) ----------
try:
    import anndata as ad
    HAVE_ANN = True
except Exception:
    HAVE_ANN = False

class MiniAnnData:
    """
    Minimal AnnData-like container sufficient for MILExperiment:
      - .obs: pandas DataFrame (rows = cells)
      - .obsm: dict of name -> np.ndarray (aligned with obs rows)
      - slicing: adata[mask] where mask is boolean index over obs rows
    """
    def __init__(self, obs: pd.DataFrame, obsm: Dict[str, np.ndarray]):
        self.obs = obs.copy()
        self.obsm = {k: np.asarray(v) for k, v in obsm.items()}
        assert all(v.shape[0] == len(self.obs) for v in self.obsm.values())

    def __getitem__(self, mask):
        if isinstance(mask, (pd.Series, np.ndarray, list)):
            mask = np.asarray(mask)
            new_obs = self.obs.loc[mask].copy()
            idx = new_obs.index.values
            # map integer positions
            if np.issubdtype(idx.dtype, np.integer):
                pos = idx
            else:
                # fallback: align by original order
                pos = self.obs.index.get_indexer(idx)
            new_obsm = {k: v[pos] for k, v in self.obsm.items()}
            # reindex obs to a fresh RangeIndex for safety
            new_obs = new_obs.reset_index(drop=True)
            new_obsm = {k: v for k, v in new_obsm.items()}
            return MiniAnnData(new_obs, new_obsm)
        raise TypeError("Unsupported index type for MiniAnnData.")

# -------------------------------
# Synthetic MIL data generator
# -------------------------------
@dataclass
class SynthConfig:
    n_patients: int = 200
    embed_dim: int = 64
    mean_cells: int = 200
    min_cells: int = 50
    positive_rate: float = 0.5
    signal_frac_pos: float = 0.10   # fraction of cells in positive bags that carry signal
    signal_strength: float = 2.0    # mean shift magnitude for signal cells
    seed: int = 42

def generate_synthetic_mil(cfg: SynthConfig, use_anndata_if_available: bool = True):
    """
    Create a dataset where only a small subset of cells in positive patients carry signal.
    This favors attention MIL over naive mean pooling.
    """
    rng = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)

    # Global signal direction (sparse or dense)
    w = rng.normal(0, 1, cfg.embed_dim)
    w = w / (np.linalg.norm(w) + 1e-8)

    rows = []
    X_all = []

    for pid in range(cfg.n_patients):
        y = 1 if rng.random() < cfg.positive_rate else 0
        n_cells = int(max(cfg.min_cells, rng.poisson(cfg.mean_cells)))

        # background noise cells
        X = rng.normal(0, 1, size=(n_cells, cfg.embed_dim))

        if y == 1:
            # inject a subset of signal cells
            n_sig = max(1, int(cfg.signal_frac_pos * n_cells))
            idx_sig = rng.choice(n_cells, size=n_sig, replace=False)
            X[idx_sig] += cfg.signal_strength * w  # mean shift along w

        # (optional) simple categorical cell type for attention plots
        # We'll assign 3 pseudo-types randomly; signal more likely in type 2
        cell_types = rng.choice(["T", "B", "Myeloid"], size=n_cells, p=[0.4, 0.3, 0.3])
        if y == 1:
            # bias signal cells to, say, "Myeloid"
            # (just for a more interpretable attention plot)
            cell_types[idx_sig] = "Myeloid"

        # accumulate
        X_all.append(X)
        rows.append(pd.DataFrame({
            "patient_id": pid,
            "outcome": y,
            "cell_type": cell_types
        }))

    obs = pd.concat(rows, ignore_index=True)
    X_all = np.vstack(X_all)
    obsm = {"X_embed": X_all.astype(np.float32)}

    if use_anndata_if_available and HAVE_ANN:
        return ad.AnnData(np.zeros((len(obs), 1))), obs, obsm  # dummy X (unused)
    else:
        return MiniAnnData(obs=obs, obsm=obsm), obs, obsm

def build_adata_object(container, obs: pd.DataFrame, obsm: Dict[str, np.ndarray]):
    """
    Place obs/obsm into either AnnData or MiniAnnData.
    """
    if HAVE_ANN and isinstance(container, type):
        # Create true AnnData
        adata = container(X=np.zeros((len(obs), 1)))
        adata.obs = obs.copy()
        for k, v in obsm.items():
            adata.obsm[k] = v
        return adata
    elif isinstance(container, MiniAnnData):
        return container
    else:
        # container is AnnData instance with dummy X
        adata = container
        adata.obs = obs.copy()
        for k, v in obsm.items():
            adata.obsm[k] = v
        return adata

# -------------------------------
# Patient-level splitting
# -------------------------------
def patient_train_val_test_split(obs: pd.DataFrame, patient_key: str, label_key: str,
                                 test_size: float = 0.20, val_size: float = 0.20, seed: int = 42):
    """
    Return lists of patient IDs for train/val/test, stratified by patient label.
    """
    df_pat = obs.groupby(patient_key)[label_key].first().reset_index()
    pids = df_pat[patient_key].values
    y = df_pat[label_key].values

    pids_train, pids_tmp, y_train, y_tmp = train_test_split(
        pids, y, test_size=(test_size + val_size), random_state=seed, stratify=y
    )
    rel_val = val_size / (test_size + val_size)
    pids_val, pids_test, y_val, y_test = train_test_split(
        pids_tmp, y_tmp, test_size=(1 - rel_val), random_state=seed, stratify=y_tmp
    )
    return pids_train, pids_val, pids_test

def subset_by_patient(adata, pids, patient_key="patient_id"):
    mask = adata.obs[patient_key].isin(set(pids)).to_numpy()
    return adata[mask].copy()

# -------------------------------
# Baseline: mean-pooling + LR
# -------------------------------
def mean_pool_by_patient(adata, embedding_col="X_embed", patient_key="patient_id", label_key="outcome"):
    """
    Returns patient-level feature matrix and labels by averaging cell embeddings per patient.
    """
    # build per-patient lists
    grouped = {}
    for pid in adata.obs[patient_key].unique():
        mask = adata.obs[patient_key] == pid
        Xp = adata.obsm[embedding_col][mask.values if isinstance(mask, pd.Series) else mask]
        grouped[pid] = Xp.mean(axis=0)

    # Align to sorted patient IDs for stable correspondence
    pids_sorted = sorted(grouped.keys())
    X_pat = np.vstack([grouped[pid] for pid in pids_sorted])
    y_pat = adata.obs.groupby(patient_key)[label_key].first().reindex(pids_sorted).values.astype(int)
    return np.asarray(pids_sorted), X_pat, y_pat

# -------------------------------
# Run experiment
# -------------------------------
def run_experiment():
    # 1) Generate synthetic MIL dataset
    cfg = SynthConfig(
        n_patients=240,
        embed_dim=64,
        mean_cells=220,
        min_cells=60,
        positive_rate=0.5,
        signal_frac_pos=0.08,   # only 8% of cells carry signal -> mean pooling becomes harder
        signal_strength=2.5,
        seed=7
    )
    if HAVE_ANN:
        dummy_container, obs, obsm = generate_synthetic_mil(cfg, use_anndata_if_available=True)
        adata = build_adata_object(dummy_container.__class__, obs, obsm)
    else:
        adata, obs, obsm = generate_synthetic_mil(cfg, use_anndata_if_available=False)

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    # 2) Train/val/test split at patient level (no leakage)
    ptrain, pval, ptest = patient_train_val_test_split(
        adata.obs, patient_key="patient_id", label_key="outcome",
        test_size=0.20, val_size=0.20, seed=1
    )
    adata_train = subset_by_patient(adata, ptrain)
    adata_val   = subset_by_patient(adata, pval)
    adata_test  = subset_by_patient(adata, ptest)

    # 3) Compute pos_weight for imbalance (based on train patients)
    train_counts = adata_train.obs.groupby("patient_id")["outcome"].first().value_counts()
    n_pos = int(train_counts.get(1, 0))
    n_neg = int(train_counts.get(0, 0))
    pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else None
    print(f"[Info] Train patients: {len(ptrain)} | Pos={n_pos}, Neg={n_neg}, pos_weight={pos_weight}")

    # 4) Run MILExperiment (uses your class defined earlier)
    exp = MILExperiment(
        embedding_col="X_embed",
        label_key="outcome",
        patient_key="patient_id",
        celltype_key="cell_type",
        hidden_dim=128,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.1,
        patience=15,
        min_delta=1e-4,
        monitor="val_loss",
        mode="min",
        pos_weight=pos_weight,
        seed=7
    )
    exp.train(adata_train, adata_val=adata_val)
    pids_mil, y_true_mil, preds_mil, scores_mil, metrics_mil = exp.evaluate(adata_test)
    print("\n=== Attention MIL (bag-level) ===")
    print(f"Test ROC AUC: {metrics_mil.get('roc_auc')}")
    print(f"Test Avg Precision: {metrics_mil.get('avg_precision')}")

    # 5) Baseline: mean-pooling + LogisticRegression
    # Prepare patient-level features for train/val/test using the SAME splits
    _, Xtr_pat, ytr_pat = mean_pool_by_patient(adata_train)
    _, Xva_pat, yva_pat = mean_pool_by_patient(adata_val)
    pids_te, Xte_pat, yte_pat = mean_pool_by_patient(adata_test)

    # Fit scaler + LR on train+val (to mirror MIL early-stopping that used val)
    Xtrv = np.vstack([Xtr_pat, Xva_pat])
    ytrv = np.concatenate([ytr_pat, yva_pat])

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", C=1.0)
    )
    pipe.fit(Xtrv, ytrv)
    scores_lr = pipe.predict_proba(Xte_pat)[:, 1]
    auc_lr = roc_auc_score(yte_pat, scores_lr) if len(set(yte_pat)) > 1 else None
    ap_lr = average_precision_score(yte_pat, scores_lr) if len(set(yte_pat)) > 1 else None

    print("\n=== Mean-Pooling + LogisticRegression (patient-level) ===")
    print(f"Test ROC AUC: {auc_lr}")
    print(f"Test Avg Precision: {ap_lr}")

    # 6) Side-by-side comparison
    print("\n=== Side-by-side ===")
    print(f"ROC AUC ->  MIL: {metrics_mil.get('roc_auc'):.4f} | Mean+LR: {auc_lr:.4f}")
    print(f"AvgPrec -> MIL: {metrics_mil.get('avg_precision'):.4f} | Mean+LR: {ap_lr:.4f}")

    # 7) (Optional) Attention by cell type on the test set
    # This is just for sanity/interpretability; comment out if not needed.
    try:
        summary = exp.visualize_attention(adata_test)
        if summary is not None:
            print("\nTop attention cell types (test set):")
            print(summary.head(5))
    except Exception as e:
        print(f"[Warn] Attention viz failed: {e}")
        
    # ---- Attention diagnostics on the TEST set ----
    attn_df = exp.collect_attention_frame(adata_test, top_k=10)

    # 1) Which cell types get attention/contribution (split by label)?
    exp.plot_attention_by_celltype(attn_df)

    # 2) Is attention too diffuse? (high entropy often correlates with errors)
    per_patient = exp.plot_entropy_vs_prob(attn_df)

    # 3) Quick table: top-k attended cell types per patient
    topk = exp.topk_table(attn_df, k=5)
    print("\nTop-1 attended cell type per patient (from the top-5 attention cells):")
    print(topk.head(20).to_string(index=False))


if __name__ == "__main__":
    # Make runs reproducible
    torch.set_num_threads(1)  # to reduce variability across machines
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)

    run_experiment()
