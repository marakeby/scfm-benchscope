"""
Survival analysis evaluation module.

Evaluates foundation model embeddings on survival prediction tasks using
a linear Cox proportional hazards model with L2 regularization, following
the methodology of:
    "A foundation model for clinical-grade computational pathology and rare
    cancers detection" (Nature Medicine, 2025).

Key design choices:
    - Linear Cox PH model (sksurv.linear_model.CoxPHSurvivalAnalysis)
    - Disease-specific survival (DSS) as the default clinical endpoint
    - 5-fold site-preserved stratification (TCGA tissue source site)
    - No validation fold: 4 folds train, 1 fold test
    - Hyperparameter α searched over 25 log-spaced values in [10^1, 10^5],
      with the L2 coefficient C = α
    - Best C chosen by best average test concordance index across folds
    - Evaluated per cancer type

References:
    - scikit-survival: https://scikit-survival.readthedocs.io/
    - TCGA-CDR: doi.org/10.1016/j.cell.2018.02.052
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from utils.logs_ import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_tcga_tss(sample_ids: pd.Index) -> pd.Series:
    """Extract the Tissue Source Site (TSS) code from TCGA barcodes.

    TCGA barcode format: ``TCGA-XX-XXXX-...`` where ``XX`` is the TSS code.

    Args:
        sample_ids: Index/Series of TCGA sample barcodes.

    Returns:
        Series of TSS codes (same index as input).
    """
    return sample_ids.to_series().str.split("-").str[1]


def site_preserved_cv_splits(
    cancer_obs: pd.DataFrame,
    n_folds: int = 5,
    site_col: str = "tss_code",
    event_col: str = "DSS",
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create site-preserved, event-stratified cross-validation folds.

    All samples from the same tissue source site are assigned to the same
    fold.  We first group sites and sort them by number of events so that
    folds are approximately balanced in terms of event counts.
    When n_folds < n_sites, multiple sites share a fold (greedy by events);
    using fewer folds (e.g. 3) can avoid empty test sets when some sites
    have zero or very few samples.

    Args:
        cancer_obs: DataFrame with at least ``site_col`` and ``event_col``.
        n_folds: Number of folds.
        site_col: Column containing the site/batch identifier.
        event_col: Column containing the binary event indicator (1 = event).
        random_state: Random seed for reproducibility.

    Returns:
        List of ``(train_idx, test_idx)`` tuples (integer indices into
        ``cancer_obs``).
    """
    rng = np.random.RandomState(random_state)

    # Aggregate per site: total samples and total events
    site_stats = cancer_obs.groupby(site_col, observed=True).agg(
        n_samples=(event_col, "size"),
        n_events=(event_col, "sum"),
    )

    # Shuffle sites, then sort by n_events descending for balanced assignment
    sites = site_stats.index.values.copy()
    rng.shuffle(sites)
    site_stats = site_stats.loc[sites]
    sites_sorted = site_stats.sort_values("n_events", ascending=False).index.values

    # Greedy assignment: assign each site to the fold with fewest events
    fold_events = np.zeros(n_folds, dtype=float)
    site_to_fold: Dict[str, int] = {}
    for site in sites_sorted:
        best_fold = int(np.argmin(fold_events))
        site_to_fold[site] = best_fold
        fold_events[best_fold] += site_stats.loc[site, "n_events"]

    # Build index arrays
    fold_assignment = cancer_obs[site_col].map(site_to_fold).values
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(cancer_obs))
    for fold_i in range(n_folds):
        test_mask = fold_assignment == fold_i
        train_idx = all_idx[~test_mask]
        test_idx = all_idx[test_mask]
        splits.append((train_idx, test_idx))

    return splits


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SurvivalEvaluator:
    """Evaluate embeddings on survival prediction via Cox PH regression.

    For each cancer type with enough samples, fits a regularised linear Cox
    model on the embedding space using site-preserved cross-validation and
    reports the concordance index (C-index).

    Attributes:
        ALPHA_GRID: Default 25 log-spaced regularisation strengths [10, 1e5].
    """

    ALPHA_GRID: np.ndarray = np.logspace(1, 5, num=25)
    PCA_DIM_THRESHOLD: int = 150  # default PCA when embedding dim > this
    DEFAULT_PCA_COMPONENTS: int = 100

    def __init__(
        self,
        adata: AnnData,
        embedding_key: str,
        event_col: str = "DSS",
        time_col: str = "DSS.time",
        cancer_type_col: str = "cancer type abbreviation",
        save_dir: Optional[str] = None,
        plots_dir: Optional[str] = None,
        n_folds: int = 5,
        alpha_grid: Optional[np.ndarray] = None,
        min_samples: int = 50,
        min_events: int = 10,
        random_state: int = 42,
        endpoints: Optional[List[str]] = None,
        n_pca_components: Optional[int] = None,
    ) -> None:
        """Initialise the survival evaluator.

        Cox PH is always fitted with scikit-survival. If embedding dim > 150,
        PCA is applied first (default 100 components) so n > p.

        Args:
            adata: AnnData with embeddings in ``obsm`` and survival data in
                ``obs``.
            embedding_key: Key in ``adata.obsm`` for the embedding matrix.
            event_col: Binary event indicator column (1 = event occurred).
            time_col: Time-to-event column (days).
            cancer_type_col: Column specifying the cancer type.
            save_dir: Directory to write metric CSVs.
            plots_dir: Directory to save visualisation plots.
            n_folds: Number of CV folds (default 5).
            alpha_grid: Array of L2 regularisation strengths to search.
                If ``None`` uses the default 25 log-spaced values [10, 1e5].
            min_samples: Minimum number of non-NaN samples per cancer type
                to include it in evaluation.
            min_events: Minimum number of events per cancer type.
            random_state: Seed for reproducibility.
            endpoints: Optional list of ``(event_col, time_col)`` pairs to
                evaluate.  If ``None``, only the primary endpoint is used.
            n_pca_components: PCA on embeddings before Cox. If ``None`` (default),
                use ``DEFAULT_PCA_COMPONENTS`` when embedding dim >
                ``PCA_DIM_THRESHOLD``, else no PCA. If 0, no PCA. If >0, use that
                many components (per-fold capped by n_train - 1 and n_features).
        """
        self.adata = adata
        self.embedding_key = embedding_key
        self.event_col = event_col
        self.time_col = time_col
        self.cancer_type_col = cancer_type_col
        self.save_dir = Path(save_dir) if save_dir else None
        self.plots_dir = Path(plots_dir) if plots_dir else self.save_dir
        self.n_folds = n_folds
        self.alpha_grid = alpha_grid if alpha_grid is not None else self.ALPHA_GRID
        self.min_samples = min_samples
        self.min_events = min_events
        self.random_state = random_state

        # Build endpoint list: each entry is (event_col, time_col, label)
        if endpoints is not None:
            self.endpoints = endpoints
        else:
            self.endpoints = [(event_col, time_col, event_col)]
        self.n_pca_components = n_pca_components

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_dir:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Validate embedding exists
        if embedding_key not in adata.obsm:
            raise ValueError(
                f"Embedding key '{embedding_key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )

    def _resolve_n_pca(self, embedding_dim: int) -> Optional[int]:
        """Resolve number of PCA components: default 100 when dim > 150 else no PCA."""
        if self.n_pca_components is not None and self.n_pca_components == 0:
            return None
        if self.n_pca_components is not None and self.n_pca_components > 0:
            return self.n_pca_components
        if embedding_dim > self.PCA_DIM_THRESHOLD:
            return self.DEFAULT_PCA_COMPONENTS
        return None

    # ------------------------------------------------------------------
    # Single cancer-type evaluation
    # ------------------------------------------------------------------

    def _evaluate_cancer_type(
        self,
        cancer_type: str,
        event_col: str,
        time_col: str,
    ) -> Optional[Dict]:
        """Run Cox PH survival evaluation for a single cancer type.

        Performs site-preserved 5-fold CV with L2 hyperparameter search.

        Args:
            cancer_type: Cancer type abbreviation (e.g. ``"BRCA"``).
            event_col: Event indicator column name.
            time_col: Time-to-event column name.

        Returns:
            Dictionary with per-fold and mean C-index, best alpha, etc.
            ``None`` if the cancer type is skipped.
        """
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored

        obs = self.adata.obs
        mask = obs[self.cancer_type_col] == cancer_type

        # Subset to this cancer type and drop NaN survival data
        sub_obs = obs.loc[mask, [event_col, time_col]].copy()
        sub_obs = sub_obs.dropna(subset=[event_col, time_col])

        # Apply event/time validity filters
        sub_obs[event_col] = sub_obs[event_col].astype(float).astype(int)
        sub_obs[time_col] = sub_obs[time_col].astype(float)
        valid_mask = sub_obs[time_col] > 0
        sub_obs = sub_obs[valid_mask]

        n_samples = len(sub_obs)
        n_events = int(sub_obs[event_col].sum())

        if n_samples < self.min_samples:
            logger.info(
                f"  [Survival] {cancer_type}: skip (n_samples={n_samples} < {self.min_samples})"
            )
            return None
        if n_events < self.min_events:
            logger.info(
                f"  [Survival] {cancer_type}: skip (n_events={n_events} < {self.min_events})"
            )
            return None

        logger.info(
            f"  [Survival] {cancer_type}: n_samples={n_samples} n_events={n_events} "
            f"event_rate={n_events / n_samples:.3f} — building splits"
        )

        # Align embeddings
        idx_in_adata = self.adata.obs.index.get_indexer(sub_obs.index)
        n_missing = (idx_in_adata == -1).sum()
        if n_missing > 0:
            logger.warning(
                f"  [Survival] {cancer_type}: {n_missing} samples not found in adata index"
            )
        X = np.asarray(self.adata.obsm[self.embedding_key][idx_in_adata], dtype=np.float64)
        # Drop embedding dimensions that contain any NaN (source: adata.obsm cache/extractor)
        finite_mask = np.all(np.isfinite(X), axis=0)
        if not np.all(finite_mask):
            n_drop = (~finite_mask).sum()
            X = X[:, finite_mask]
            logger.warning(
                f"  [Survival] {cancer_type}: dropped {n_drop} embedding dims with NaN "
                f"(kept {X.shape[1]} all-finite dims). Source: adata.obsm."
            )
        if X.shape[1] == 0:
            logger.warning(
                f"  [Survival] {cancer_type}: skip (no all-finite embedding dims left)"
            )
            return None
        embedding_dim = X.shape[1]
        effective_pca = self._resolve_n_pca(embedding_dim)
        if effective_pca is not None:
            logger.info(
                f"  [Survival] {cancer_type}: embedding dim={embedding_dim} > "
                f"{self.PCA_DIM_THRESHOLD}, applying PCA (k={effective_pca}) before Cox"
            )
        # Build structured survival array
        y = np.array(
            [(bool(e), t) for e, t in zip(sub_obs[event_col], sub_obs[time_col])],
            dtype=[("event", bool), ("time", float)],
        )

        # Extract TSS codes for site-preserved splitting
        sub_obs_reset = sub_obs.reset_index()
        sub_obs_reset["tss_code"] = extract_tcga_tss(sub_obs.index).values
        n_sites = sub_obs_reset["tss_code"].nunique()

        splits = site_preserved_cv_splits(
            sub_obs_reset,
            n_folds=self.n_folds,
            site_col="tss_code",
            event_col=event_col,
            random_state=self.random_state,
        )

        fold_sizes = [(len(te), int(y[te]["event"].sum())) for _, te in splits]
        logger.info(
            f"  [Survival] {cancer_type}: n_sites={n_sites} "
            f"folds=[n_test, test_events] {fold_sizes}"
        )

        # Hyperparameter search: for each alpha, average C-index across folds
        alpha_scores: Dict[float, List[float]] = {a: [] for a in self.alpha_grid}

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            n_test = len(test_idx)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_events = y_test["event"]
            n_test_events = int(test_events.sum())

            logger.info(
                f"  [Survival] {cancer_type} fold={fold_i} "
                f"n_train={len(train_idx)} n_test={n_test} "
                f"test_events={n_test_events}/{n_test}"
            )

            # Skip fold if test set is empty (0,0), has no events, or all events
            if n_test == 0:
                logger.warning(
                    f"  [Survival] {cancer_type} fold={fold_i} SKIP: empty test set (0,0)"
                )
                for a in self.alpha_grid:
                    alpha_scores[a].append(np.nan)
                continue
            if n_test_events == 0 or n_test_events == n_test:
                logger.warning(
                    f"  [Survival] {cancer_type} fold={fold_i} SKIP: no variability "
                    f"(test_events={n_test_events}/{n_test})"
                )
                for a in self.alpha_grid:
                    alpha_scores[a].append(np.nan)
                continue

            # Optional PCA on embeddings so n_train > p for Cox (avoids singular matrix)
            if effective_pca is not None:
                from sklearn.decomposition import PCA
                n_train = len(train_idx)
                d = X_train.shape[1]
                n_comp = min(effective_pca, n_train - 1, d)
                n_comp = max(1, n_comp)
                pca = PCA(n_components=n_comp, random_state=self.random_state)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            for alpha in self.alpha_grid:
                try:
                    model = CoxPHSurvivalAnalysis(alpha=alpha)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_train, y_train)
                    risk_scores = model.predict(X_test)
                    c_index, *_ = concordance_index_censored(
                        y_test["event"], y_test["time"], risk_scores
                    )
                    alpha_scores[alpha].append(c_index)
                except Exception as e:
                    logger.debug(
                        f"  {cancer_type} fold {fold_i} alpha={alpha:.1f}: {e}"
                    )
                    alpha_scores[alpha].append(np.nan)

        # Select best alpha by mean C-index across folds (ignoring NaN folds)
        def _safe_nanmean(scores: List[float]) -> float:
            arr = np.asarray(scores, dtype=np.float64)
            valid = arr[~np.isnan(arr)]
            return float(np.mean(valid)) if len(valid) > 0 else np.nan

        mean_scores = {
            a: _safe_nanmean(scores) for a, scores in alpha_scores.items()
        }
        best_alpha = max(mean_scores, key=lambda a: mean_scores[a])
        best_mean_ci = mean_scores[best_alpha]

        # Collect per-fold scores at best alpha
        per_fold_ci = alpha_scores[best_alpha]
        per_fold_arr = np.asarray(per_fold_ci, dtype=np.float64)
        valid_ci = per_fold_arr[~np.isnan(per_fold_arr)]
        if len(valid_ci) < 2:
            std_ci = 0.0
        else:
            std_ci = float(np.std(valid_ci))
        result = {
            "cancer_type": cancer_type,
            "n_samples": n_samples,
            "n_events": n_events,
            "event_rate": n_events / n_samples,
            "n_folds": self.n_folds,
            "best_alpha": best_alpha,
            "mean_c_index": best_mean_ci,
            "std_c_index": std_ci,
        }
        # Per-fold details
        for i, ci in enumerate(per_fold_ci):
            result[f"fold_{i}_c_index"] = ci

        logger.info(
            f"  [Survival] {cancer_type}: done C-index={best_mean_ci:.4f} ± "
            f"{result['std_c_index']:.4f} best_alpha={best_alpha:.1f}"
        )
        return result

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, pd.DataFrame]:
        """Run survival evaluation across all cancer types and endpoints.

        Returns:
            Dictionary mapping endpoint labels to DataFrames of per-cancer-type
            results.  Also saves CSVs and summary plots.
        """
        logger.info("=" * 60)
        logger.info("SURVIVAL ANALYSIS EVALUATION (Cox PH)")
        logger.info(f"Embedding key: {self.embedding_key}")
        logger.info(f"Endpoints: {[(e, t) for e, t, _ in self.endpoints]}")
        logger.info(f"CV folds: {self.n_folds}")
        logger.info(f"Alpha grid: {len(self.alpha_grid)} values in "
                     f"[{self.alpha_grid.min():.0f}, {self.alpha_grid.max():.0f}]")
        logger.info("Cox backend: sksurv")
        emb_dim = self.adata.obsm[self.embedding_key].shape[1]
        n_pca = self._resolve_n_pca(emb_dim)
        if n_pca is not None:
            logger.info(f"PCA: {n_pca} components (embedding dim={emb_dim} > {self.PCA_DIM_THRESHOLD})")
        else:
            logger.info(f"PCA: none (embedding dim={emb_dim})")
        logger.info("=" * 60)

        all_results: Dict[str, pd.DataFrame] = {}

        for event_col, time_col, label in self.endpoints:
            logger.info(f"\n--- Endpoint: {label} ({event_col}, {time_col}) ---")

            # Determine which cancer types to evaluate
            obs = self.adata.obs
            cancer_types = sorted(obs[self.cancer_type_col].dropna().unique())
            logger.info(f"Found {len(cancer_types)} cancer types")

            endpoint_results: List[Dict] = []

            for ct in cancer_types:
                result = self._evaluate_cancer_type(ct, event_col, time_col)
                if result is not None:
                    result["endpoint"] = label
                    endpoint_results.append(result)

            if not endpoint_results:
                logger.warning(f"No cancer types passed filters for {label}")
                continue

            results_df = pd.DataFrame(endpoint_results)
            all_results[label] = results_df

            # Compute pan-cancer summary
            mean_ci = results_df["mean_c_index"].mean()
            median_ci = results_df["mean_c_index"].median()
            logger.info(
                f"\n{label} Pan-cancer: mean C-index = {mean_ci:.4f}, "
                f"median = {median_ci:.4f} "
                f"({len(results_df)} cancer types)"
            )

            # Save per-endpoint CSV
            if self.save_dir:
                csv_path = self.save_dir / f"survival_{label}_per_cancer.csv"
                results_df.to_csv(csv_path, index=False)
                logger.info(f"Saved per-cancer results to {csv_path}")

        # Save combined summary
        if all_results and self.save_dir:
            combined = pd.concat(all_results.values(), ignore_index=True)
            combined.to_csv(self.save_dir / "survival_all_endpoints.csv", index=False)

            # Summary: just cancer_type, endpoint, mean_c_index, std_c_index
            summary_cols = [
                "endpoint", "cancer_type", "n_samples", "n_events",
                "mean_c_index", "std_c_index", "best_alpha",
            ]
            summary = combined[
                [c for c in summary_cols if c in combined.columns]
            ]
            summary.to_csv(self.save_dir / "survival_summary.csv", index=False)
            logger.info(f"Saved summary to {self.save_dir / 'survival_summary.csv'}")

        # Generate plots
        if all_results:
            self._plot_results(all_results)

        logger.info("Survival analysis evaluation complete.")
        return all_results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _plot_results(self, all_results: Dict[str, pd.DataFrame]) -> None:
        """Generate a bar chart of C-index per cancer type for each endpoint."""
        if not self.plots_dir:
            return

        for label, df in all_results.items():
            df_sorted = df.sort_values("mean_c_index", ascending=True)

            fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df_sorted))))
            y_pos = np.arange(len(df_sorted))

            bars = ax.barh(
                y_pos,
                df_sorted["mean_c_index"].values,
                xerr=df_sorted["std_c_index"].values,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
            )

            # Reference line at 0.5 (random)
            ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_sorted["cancer_type"].values, fontsize=9)
            ax.set_xlabel("Concordance Index (C-index)", fontsize=11)
            ax.set_title(
                f"Survival Prediction — {label}\n"
                f"Cox PH, {self.n_folds}-fold site-preserved CV",
                fontsize=12,
            )
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlim(0, 1)

            # Annotate bars with values
            for i, (ci, std) in enumerate(
                zip(df_sorted["mean_c_index"], df_sorted["std_c_index"])
            ):
                ax.text(
                    ci + std + 0.01, i, f"{ci:.3f}", va="center", fontsize=8
                )

            fig.tight_layout()
            plot_path = self.plots_dir / f"survival_{label}_c_index_per_cancer.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved plot to {plot_path}")
