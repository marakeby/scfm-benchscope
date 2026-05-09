from pathlib import Path
from typing import Dict, Union, List
from anndata import AnnData
import numpy as np
import scanpy as sc
import scib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from utils.logs_ import get_logger
import multiprocessing

logger = get_logger()

from sklearn.metrics import accuracy_score
from utils.sampling import sample_adata


def _graph_connectivity_no_deprecated_pandas(adata: AnnData, *, label_key: str) -> float:
    """
    scIB's graph_connectivity without deprecated `pd.value_counts`.

    Uses the same definition as `scib.metrics.graph_connectivity`, but replaces
    `pd.value_counts` with `pd.Series(...).value_counts()` to avoid pandas deprecation warnings.
    """
    if "neighbors" not in adata.uns:
        raise KeyError("Please compute the neighborhood graph before running graph connectivity!")
    if "connectivities" not in adata.obsp:
        raise KeyError("Expected adata.obsp['connectivities'] to exist (run neighbors first).")

    labels = adata.obs[label_key]
    cats = labels.cat.categories if hasattr(labels, "cat") else pd.Categorical(labels).categories

    scores: list[float] = []
    for lab in cats:
        mask = (adata.obs[label_key] == lab).to_numpy()
        # Slice produces a view; copy to keep behavior explicit and stable.
        adata_sub = adata[mask].copy()
        _, comp_labels = connected_components(adata_sub.obsp["connectivities"], connection="strong")
        tab = pd.Series(comp_labels).value_counts()
        scores.append(float(tab.max() / tab.sum()))

    return float(np.mean(scores)) if scores else float("nan")


def _pcr_no_scanpy_warning(
    adata: AnnData,
    *,
    covariate: str,
    embed: str | None = None,
    n_comps: int = 50,
    n_threads: int = 1,
) -> float:
    """
    Principal component regression (PCR) without calling `scanpy.tl.pca(use_highly_variable=...)`.

    This mirrors scIB's `pcr` definition but computes PCA via sklearn to avoid
    Scanpy's `use_highly_variable` FutureWarning.
    """
    from scipy import sparse
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    if covariate not in adata.obs:
        raise KeyError(f"covariate={covariate!r} not in adata.obs")
    cov = adata.obs[covariate]

    if embed is None:
        X = adata.X
    else:
        if embed not in adata.obsm:
            raise KeyError(f"embed={embed!r} not in adata.obsm")
        X = adata.obsm[embed]

    # One-hot encode categorical covariate (as scIB does).
    if pd.api.types.is_numeric_dtype(cov):
        C = np.asarray(cov).reshape(-1, 1)
    else:
        C = pd.get_dummies(cov).to_numpy()

    # PCA on dense/sparse matrices.
    if sparse.issparse(X):
        # TruncatedSVD works for sparse; explained_variance_ratio_ approximates PCA on centered data.
        svd = TruncatedSVD(n_components=min(int(n_comps), max(1, min(X.shape) - 1)), random_state=0)
        X_pca = svd.fit_transform(X)
        var = svd.explained_variance_ratio_
    else:
        X_arr = np.asarray(X)
        n = min(int(n_comps), max(1, min(X_arr.shape)))
        pca = PCA(n_components=n, svd_solver="auto", random_state=0)
        X_pca = pca.fit_transform(X_arr)
        var = pca.explained_variance_ratio_

    # Fit multi-target linear regression: C -> PCs, then per-PC R2.
    model = LinearRegression(fit_intercept=True, n_jobs=n_threads if n_threads is not None else None)
    model.fit(C, X_pca)
    X_pred = model.predict(C)
    r2 = np.array([max(0.0, r2_score(X_pca[:, i], X_pred[:, i])) for i in range(X_pca.shape[1])], dtype=float)

    # Weighted sum of R2 by variance contribution.
    var = np.asarray(var, dtype=float)
    var = var / var.sum() if var.sum() > 0 else var
    return float(np.sum(r2 * var))


def _as_2d_array(x, *, dtype=np.float64) -> np.ndarray:
    """
    Coerce an embedding-like object to a real 2D numpy array.

    Notes
    -----
    - `np.asarray(csr_matrix)` produces a 0-d `object` array containing the sparse
      matrix, which breaks downstream sklearn / metric code expecting `x.ndim==2`.
    - AnnData `obsm[...]` may store either dense arrays or scipy sparse matrices.
    """
    if issparse(x):
        return x.toarray().astype(dtype, copy=False)

    arr = np.asarray(x)
    # Handle the pathological case: array(<csr_matrix>, dtype=object) scalar.
    if arr.dtype == object and arr.shape == ():
        item = arr.item()
        if issparse(item):
            return item.toarray().astype(dtype, copy=False)
        arr = np.asarray(item)

    if arr.ndim != 2:
        raise TypeError(f"Expected embedding with 2 dimensions, got shape={arr.shape} dtype={arr.dtype!r}")

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _embedding_copy(embedding):
    """
    Defensive copy for embeddings to avoid in-place mutations during evaluation.
    Supports numpy arrays, scipy sparse matrices, and torch tensors.
    """
    # torch.Tensor -> numpy
    if hasattr(embedding, "detach") and hasattr(embedding, "cpu") and hasattr(embedding, "numpy"):
        embedding = embedding.detach().cpu().numpy()

    # scipy sparse matrices provide .copy()
    if issparse(embedding):
        return embedding.copy()

    try:
        return np.array(embedding, copy=True)
    except Exception:
        return embedding.copy() if hasattr(embedding, "copy") else embedding


def _ensure_neighbors(
    adata: AnnData,
    embedding_key: str,
    *,
    n_neighbors: int = 15,
    metric: str = "cosine",
    random_state: int = 0,
    force_recompute: bool = True,
    n_jobs: int = -1,
) -> None:
    """
    Ensure Scanpy kNN graph exists for the given embedding.

    Centralizes neighbor graph construction so it's computed once and reused by
    downstream steps (Leiden + scIB / scib-metrics). When ``force_recompute`` is
    True, any pre-existing neighbors graph is removed to avoid mismatches across
    different embeddings.
    """
    if force_recompute and "neighbors" in adata.uns:
        logger.warning(
            "neighbors in adata.uns found; overwriting neighbors graph for use_rep=%s "
            "(n_neighbors=%d, metric=%s).",
            embedding_key,
            n_neighbors,
            metric,
        )
        adata.uns.pop("neighbors", None)
        # Scanpy stores adjacency matrices in .obsp; remove them too to avoid
        # accidentally reusing stale graphs.
        adata.obsp.pop("connectivities", None)
        adata.obsp.pop("distances", None)

    if "neighbors" not in adata.uns:
        # Scanpy's signature differs across versions; `n_jobs` is optional.
        try:
            sc.pp.neighbors(
                adata,
                use_rep=embedding_key,
                n_neighbors=n_neighbors,
                metric=metric,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        except TypeError:
            sc.pp.neighbors(
                adata,
                use_rep=embedding_key,
                n_neighbors=n_neighbors,
                metric=metric,
                random_state=random_state,
            )


def _neighbors_results_for_scib_metrics(adata: AnnData, embedding_key: str):
    """
    Build ``scib_metrics.nearest_neighbors.NeighborsResults`` from Scanpy's kNN graph.

    Prefer :func:`scib_metrics.utils.convert_knn_graph_to_idx` on ``obsp['distances']``;
    if that fails (irregular sparsity), recompute kNN on the embedding with sklearn.
    """
    from scib_metrics.nearest_neighbors import NeighborsResults
    from scib_metrics.utils import convert_knn_graph_to_idx

    dist = adata.obsp["distances"]
    if not issparse(dist):
        dist = csr_matrix(dist)
    dist = dist.tocsr()
    try:
        knn_dists, knn_idx = convert_knn_graph_to_idx(dist)
        return NeighborsResults(indices=knn_idx, distances=knn_dists)
    except Exception as exc:
        logger.warning(
            "convert_knn_graph_to_idx failed (%s); recomputing kNN on embedding for scib-metrics",
            exc,
        )
    n_neighbors = int(adata.uns["neighbors"]["params"].get("n_neighbors", 15))
    metric = str(adata.uns["neighbors"]["params"].get("metric", "cosine"))
    x_emb = _as_2d_array(adata.obsm[embedding_key], dtype=np.float64)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    knn.fit(x_emb)
    knn_dists, knn_idx = knn.kneighbors(x_emb)
    return NeighborsResults(indices=knn_idx, distances=knn_dists)


def _knn_label_prediction_metrics(
    adata: AnnData,
    *,
    label_key: str,
    embedding_key: str,
    n_neighbors: int = 15,
    test_size: float = 0.2,
    random_state: int = 0,
    n_jobs: int = -1,
) -> Dict[str, float]:
    """
    Simple kNN classifier on the embedding to predict labels.

    Returns scalar metrics on a stratified train/test split.
    """
    x = _as_2d_array(adata.obsm[embedding_key], dtype=np.float64)
    y_raw = adata.obs[label_key].to_numpy()

    # Encode labels to 0..C-1 (handles strings / categoricals).
    _, y = np.unique(y_raw, return_inverse=True)

    # If too few samples per class, stratified split may fail; fall back gracefully.
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs, weights="distance")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return {
        "kNN_label_acc": float(metrics.accuracy_score(y_test, y_pred)),
        "kNN_label_f1_macro": float(
            metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
    }


def _knn_purity_from_distances(
    adata: AnnData,
    *,
    obs_key: str,
    k: int | None = None,
) -> float:
    """
    Mean fraction of kNN neighbors sharing the same category (obs_key).

    Uses Scanpy's sparse kNN graph stored in ``adata.obsp['distances']``.
    Higher means neighbors are more homogeneous for that key.
    """
    if "distances" not in adata.obsp:
        raise ValueError("Expected adata.obsp['distances'] to exist (run neighbors first).")

    dist = adata.obsp["distances"]
    if not issparse(dist):
        dist = csr_matrix(dist)
    dist = dist.tocsr()

    labels_raw = adata.obs[obs_key].to_numpy()
    _, labels = np.unique(labels_raw, return_inverse=True)

    n = adata.n_obs
    total = 0.0
    counted = 0

    for i in range(n):
        row = dist.getrow(i)
        neigh = row.indices
        if neigh.size == 0:
            continue
        if k is not None and neigh.size > k:
            # Keep the k smallest non-zero distances (closest neighbors)
            d = row.data
            sel = np.argpartition(d, kth=k - 1)[:k]
            neigh = neigh[sel]
        total += float(np.mean(labels[neigh] == labels[i]))
        counted += 1

    return float(total / counted) if counted > 0 else float("nan")


def _knn_batch_prediction_accuracy(
    adata: AnnData,
    *,
    batch_key: str,
    embedding_key: str,
    n_neighbors: int = 15,
    test_size: float = 0.2,
    random_state: int = 0,
    n_jobs: int = -1,
) -> float:
    """
    Predict batch from the embedding via kNN; lower is better (harder to predict batch).
    """
    x = _as_2d_array(adata.obsm[embedding_key], dtype=np.float64)
    y_raw = adata.obs[batch_key].to_numpy()
    _, y = np.unique(y_raw, return_inverse=True)

    # If only one batch, prediction is trivial / undefined; return NaN.
    if np.unique(y).size <= 1:
        return float("nan")

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs, weights="distance")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return float(metrics.accuracy_score(y_test, y_pred))


class EmbeddingEvaluator:
    def __init__(
        self,
        adata,
        embedding_key,
        save_dir,
        n_jobs: int = -1,
        save_id: str | int | None = None,
        *,
        random_state: int = 0,
        auto_subsample: bool = True,
        sample_size: int = 10000,
    ):
        self.auto_subsample = auto_subsample
        self.adata = adata
        self.batch_key = 'batch'
        self.label_key = 'label'
        self.embedding_key = embedding_key
        self.n_jobs = n_jobs
        self.save_id = save_id
        self.random_state = int(random_state)
        self.sample_size = sample_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self):
        logger.info(f"Evaluating embeddings {self.adata.X.shape}")

        adata = self.adata
        if self.auto_subsample and adata.shape[0] > self.sample_size:
            logger.info(f"Subsampling adata to {self.sample_size} cells for evaluation")
            adata = sample_adata(adata, sample_size=self.sample_size, stratify_by=self.batch_key)
            logger.info(f"Subsampled adata to {adata.X.shape} cells")

        # Build a lightweight AnnData for evaluation to avoid mutating the original adata
        # (Scanpy writes neighbors/leiden into .uns/.obsp/.obs) and to avoid mutating the
        # caller-owned embedding (some operations may touch views / shared arrays).
        x_emb = _embedding_copy(adata.obsm[self.embedding_key])
        adata_eval = AnnData(
            X=np.zeros((adata.n_obs, 1), dtype=np.float32),
            obs=adata.obs.copy(),
        )
        adata_eval.obsm[self.embedding_key] = x_emb
                
        results_dict = eval_scib_metrics(
            adata_eval,
            self.batch_key,
            self.label_key,
            self.embedding_key,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        unsupervised_metrics_df = pd.DataFrame.from_dict({self.embedding_key: results_dict})
        suffix = f"_{self.save_id}" if self.save_id is not None else ""
        csv_path = self.save_dir / f"embedding_metrics{suffix}.csv"
        unsupervised_metrics_df.to_csv(csv_path)
        logger.info(f"Saved results to '{csv_path}'")
        return results_dict


def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT",
    n_jobs: int = -1,
    *,
    random_state: int = 0,
) -> Dict:
    
    # in case just one batch scib.metrics.metrics doesn't work 
    # call them separately
    results_dict = dict()
    

    logger.info('leiden')
    _ensure_neighbors(
        adata,
        embedding_key,
        n_neighbors=15,
        metric="cosine",
        random_state=random_state,
        force_recompute=True,
        n_jobs=n_jobs,
    )
        
    # Avoid Scanpy FutureWarning about future default backend.
    try:
        sc.tl.leiden(
            adata,
            key_added="cluster",
            random_state=random_state,
            resolution=0.3,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except TypeError:
        # Older Scanpy versions don't support these kwargs.
        sc.tl.leiden(
            adata,
            key_added="cluster",
            random_state=random_state,
            resolution=0.3,
        )
    
#     sc.pp.neighbors(adata, use_rep=embedding_key)  # or your embedding
# sc.tl.louvain(adata, resolution=0.3, key_added="louvain_custom")

    
    # res_max, nmi_max, nmi_all = scib.metrics.clustering.opt_louvain(
    #         adata,
    #         label_key=label_key,
    #         cluster_key="cluster",
    #         use_rep=embedding_key,
    #         function=scib.metrics.nmi,
    #         plot=False,
    #         verbose=False,
    #         inplace=True,
    #         force=True,
    # )
    # print('nmi')
    
    results_dict["NMI_cluster/label"] = scib.metrics.nmi(
        adata, 
        "cluster",
        label_key,
        "arithmetic",
        nmi_dir=None
    )
    
    # print('ARI_cluster/label')
    results_dict["ARI_cluster/label"] = scib.metrics.ari(
        adata, 
        "cluster", 
        label_key
    )
    # print('ASW_label')
    
    results_dict["ASW_label"] = scib.metrics.silhouette(
        adata, 
        label_key, 
        embedding_key, 
        "euclidean"
    )   

    # print('graph_conn')
    results_dict["graph_conn"] = _graph_connectivity_no_deprecated_pandas(adata, label_key=label_key)
    
    # print('batchs')
    # Calculate this only if there are multiple batches
    if len(adata.obs[batch_key].unique()) > 1:
        results_dict["ASW_batch"] = scib.metrics.silhouette(
            adata,
            batch_key,
            embedding_key,
            "euclidean"
        )

        # scIB "Batch ASW" (per–cell-type mean |silhouette w.r.t. batch|); also saved as batch_ASW
        batch_asw = scib.metrics.silhouette_batch(
            adata,
            batch_key,
            label_key,
            embed=embedding_key,
            metric="euclidean",
            return_all=False,
            verbose=False,
        )
        results_dict["ASW_label/batch"] = batch_asw
        results_dict["batch_ASW"] = batch_asw

        results_dict["PCR_batch"] = _pcr_no_scanpy_warning(
            adata,
            covariate=batch_key,
            embed=embedding_key,
            n_comps=50,
            n_threads=1 if n_jobs in (-1, None) else int(n_jobs),
        )

        # iLISI, cLISI, kBET: scib-metrics (Python/JAX; no R). See https://scib-metrics.readthedocs.io/
        try:
            import scib_metrics
        except ImportError:
            logger.warning(
                "scib-metrics not installed; skipping iLISI, cLISI, and kBET. "
                "Add it to your environment (e.g. pip / pixi): scib-metrics"
            )
        else:
            try:
                nr = _neighbors_results_for_scib_metrics(adata, embedding_key)
                batches = adata.obs[batch_key].to_numpy()
                labels = adata.obs[label_key].to_numpy()
                results_dict["iLISI"] = float(
                    scib_metrics.ilisi_knn(nr, batches, scale=True)
                )
                results_dict["cLISI"] = float(
                    scib_metrics.clisi_knn(nr, labels, scale=True)
                )
                results_dict["kBET"] = float(
                    scib_metrics.kbet_per_label(nr, batches, labels, alpha=0.05)
                )
            except Exception as exc:
                logger.warning(
                    "scib-metrics (iLISI / cLISI / kBET) failed: %s: %r",
                    type(exc).__name__,
                    exc,
                )

    # print('avg_bio')
    results_dict["avg_bio"] = np.mean(
        [
            results_dict["NMI_cluster/label"],
            results_dict["ARI_cluster/label"],
            results_dict["ASW_label"],
        ]
    )

    # --- kNN neighborhood purity metrics (fast, graph-based) ---
    try:
        k_nn = (
            int(adata.uns["neighbors"]["params"].get("n_neighbors", 15))
            if "neighbors" in adata.uns
            else 15
        )
        results_dict["knn_label_purity"] = _knn_purity_from_distances(
            adata, obs_key=label_key, k=k_nn
        )
    except Exception as exc:
        logger.warning("knn_label_purity failed: %s: %r", type(exc).__name__, exc)

    if len(adata.obs[batch_key].unique()) > 1:
        try:
            results_dict["knn_batch_purity"] = _knn_purity_from_distances(
                adata, obs_key=batch_key, k=k_nn
            )
        except Exception as exc:
            logger.warning("knn_batch_purity failed: %s: %r", type(exc).__name__, exc)

        # --- batch predictability (lower is better) ---
        try:
            results_dict["batch_pred_acc"] = _knn_batch_prediction_accuracy(
                adata,
                batch_key=batch_key,
                embedding_key=embedding_key,
                n_neighbors=k_nn,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        except Exception as exc:
            logger.warning("batch_pred_acc failed: %s: %r", type(exc).__name__, exc)

    # kNN cell-type prediction on the embedding (fast supervised sanity check)
    try:
        results_dict.update(
            _knn_label_prediction_metrics(
                adata,
                label_key=label_key,
                embedding_key=embedding_key,
                n_neighbors=int(adata.uns["neighbors"]["params"].get("n_neighbors", 15))
                if "neighbors" in adata.uns
                else 15,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
    except Exception as exc:
        logger.warning("kNN label prediction metric failed: %s: %r", type(exc).__name__, exc)

    logger.debug(
        "\n".join([f"{k}: {v:.4f}" for k, v in results_dict.items()])
    )

    # remove nan value in result_dict
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}

    return results_dict


from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_classifier(y_test, y_pred, y_pred_score, estimator_name, label_names):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    n_classes = len(label_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    cm = confusion_matrix(y_test, y_pred)

    metrics_fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    # Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names).plot(ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.grid(False)

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=label_names).plot(ax=ax2)
    ax2.set_title('Normalized Confusion Matrix')
    ax2.grid(False)

    # ROC curve (One-vs-Rest)

    if n_classes == 2:
        if y_pred_score.ndim>1:
            y_pred_score = y_pred_score[:,1]
        # Binary case: use the second column only
        fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, 0], y_pred_score)
        auc = metrics.auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f'{label_names[1]} (AUC={auc:.2f})')
    else:
        # Multiclass case
        for i in range(n_classes):
            fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            auc = metrics.auc(fpr, tpr)
            ax3.plot(fpr, tpr, label=f'{label_names[i]} (AUC={auc:.2f})')
    ax3.plot([0, 1], [0, 1], 'k--', lw=1)
    ax3.set_title('ROC Curve')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc='lower right')

    # PR curve (One-vs-Rest)
    if n_classes == 2:
        # Binary case: use the second column only
        precision, recall, _ = metrics.precision_recall_curve(y_test_bin[:, 0], y_pred_score)
        ap = metrics.average_precision_score(y_test_bin[:, 0], y_pred_score)
        ax4.plot(recall, precision, label=f'{label_names[1]} (AP={ap:.2f})')
    else:
        for i in range(n_classes):
            precision, recall, _ = metrics.precision_recall_curve(y_test_bin[:, i], y_pred_score[:, i])
            ap = metrics.average_precision_score(y_test_bin[:, i], y_pred_score[:, i])
            ax4.plot(recall, precision, label=f'{label_names[i]} (AP={ap:.2f})')
    ax4.set_title('Precision-Recall Curve')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.legend(loc='lower left')

    plt.tight_layout()
    metrics_fig.suptitle(estimator_name, fontsize=BIGGER_SIZE, y=1.02)

    return metrics_fig


#     return  metrics_df, cls_report
from sklearn.preprocessing import label_binarize
# def eval_classifier(y_test, y_pred, y_pred_score, estimator_name, label_names):
    
#     logger.info(f'Evaluating classifier {y_test.shape } {y_pred_score.shape} {estimator_name}')
    
#     n_classes = len(label_names)
#     y_test_bin = label_binarize(y_test, classes=range(n_classes))

#     # AUC and AUPRC
#     if n_classes == 2:
#         if y_pred_score.ndim>1:
#             y_pred_score = y_pred_score[:,1]
            
#         fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_score)
#         roc_auc = metrics.auc(fpr, tpr)


#         precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_score)
#         average_precision = metrics.average_precision_score(y_test, y_pred_score)
#     else:
#         roc_auc = metrics.roc_auc_score(y_test_bin, y_pred_score, average='macro', multi_class='ovr')
#         average_precision = metrics.average_precision_score(y_test_bin, y_pred_score, average='macro')

#     # Overall metrics
#     f1 = metrics.f1_score(y_test, y_pred, average='macro')
#     precision_score = metrics.precision_score(y_test, y_pred, average='macro')
#     recall = metrics.recall_score(y_test, y_pred, average='macro')
#     accuracy = metrics.accuracy_score(y_test, y_pred)

#     # Summary dataframe
#     metrics_dict = dict(
#         AUC=roc_auc,
#         AUPRC=average_precision,
#         F1=f1,
#         Accuracy=accuracy,
#         Precision=precision_score,
#         Recall=recall
#     )
#     s = pd.Series(metrics_dict, name='Metrics')
#     metrics_df = s.to_frame()
#     metrics_df.columns = [estimator_name]
#     metrics_df.index.name = 'Metrics'
#     name_to_label = {name: i for i, name in enumerate(label_names)}
#     # labels = name_to_label
#     # Detailed classification report
#     cls_report = metrics.classification_report(
#         y_test, y_pred, output_dict=True, target_names=label_names,
#     )

#     return metrics_df, cls_report

def eval_classifier(y_test, y_pred, y_pred_score, estimator_name, label_names):
    """
    y_test: 1D array-like of true labels (ints 0..n_classes-1)
    y_pred: 1D array-like of predicted labels (same encoding)
    y_pred_score: 
        - binary: 1D scores for positive class OR 2D probas with shape (n, 2)
        - multiclass: 2D probas with shape (n, n_classes) matching label_names order
    label_names: list of class names in the canonical order; len defines n_classes
    """
    logger.info(f'Evaluating classifier {np.shape(y_test)} {np.shape(y_pred_score)} {estimator_name}')
    
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    n_classes = len(label_names)
    all_labels = list(range(n_classes))  # canonical label indices

    # --- AUC / AUPRC ---
    roc_auc = np.nan
    average_precision = np.nan

    if n_classes == 2:
        # Normalize scores to 1D for positive class=1
        if y_pred_score is None:
            pass  # leave NaNs
        else:
            yps = np.asarray(y_pred_score)
            if yps.ndim == 2 and yps.shape[1] >= 2:
                yps = yps[:, 1]
            yps = np.asarray(yps).reshape(-1)

            # Only compute AUC/PR when both classes present in y_test
            if np.unique(y_test).size > 1:
                try:
                    fpr, tpr, _ = metrics.roc_curve(y_test, yps)
                    roc_auc = metrics.auc(fpr, tpr)
                except ValueError:
                    roc_auc = np.nan
                try:
                    # precision_recall_curve itself can run, but AP is the scalar we want
                    average_precision = metrics.average_precision_score(y_test, yps)
                except ValueError:
                    average_precision = np.nan
    else:
        # Multiclass: expect y_pred_score to be (n_samples, n_classes)
        if y_pred_score is not None and np.asarray(y_pred_score).ndim == 2 and np.asarray(y_pred_score).shape[1] == n_classes:
            y_test_bin = label_binarize(y_test, classes=all_labels)
            # Only meaningful if at least 2 classes appear in this fold
            if np.unique(y_test).size > 1:
                try:
                    roc_auc = metrics.roc_auc_score(y_test_bin, y_pred_score, average='macro', multi_class='ovr')
                except ValueError:
                    roc_auc = np.nan
                try:
                    average_precision = metrics.average_precision_score(y_test_bin, y_pred_score, average='macro')
                except ValueError:
                    average_precision = np.nan

    # --- Overall metrics (robust to single-class with zero_division=0) ---
    f1 = metrics.f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision_score = metrics.precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, average='macro', zero_division=0)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    metrics_dict = dict(
        AUC=roc_auc,
        AUPRC=average_precision,
        F1=f1,
        Accuracy=accuracy,
        Precision=precision_score,
        Recall=recall,
    )
    metrics_df = pd.Series(metrics_dict, name='Metrics').to_frame()
    metrics_df.columns = [estimator_name]
    metrics_df.index.name = 'Metrics'

    # --- Detailed report: force full label space, avoid crashes ---
    cls_report = metrics.classification_report(
        y_test,
        y_pred,
        labels=all_labels,                # ensure rows for all classes in label_names
        target_names=list(label_names),   # must match len(labels)
        zero_division=0,
        output_dict=True,
    )

    return metrics_df, cls_report