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
from utils.logs_ import get_logger
import multiprocessing

from scib.metrics import nmi, ari, silhouette, graph_connectivity, silhouette_batch, pcr


logger = get_logger()

from sklearn.metrics import accuracy_score
from utils.sampling import sample_adata


class EmbeddingEvaluator:
    def __init__(self, adata, embedding_key, save_dir, auto_subsample=True):
        self.auto_subsample = auto_subsample
        self.adata = adata
        self.batch_key = 'batch'
        self.label_key = 'label'
        self.embedding_key = embedding_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self):
        logger.info(f'Evaluating embeddings {self.adata.X.shape}')

        if self.auto_subsample:
            if self.adata.shape[0]>10000:
                self.adata = sample_adata(self.adata, sample_size=5000, stratify_by=None)
                
        results_dict = eval_scib_metrics(self.adata,self.batch_key,  self.label_key,self.embedding_key )
        unsupervised_metrics_df = pd.DataFrame.from_dict({self.embedding_key: results_dict})
        unsupervised_metrics_df.to_csv(self.save_dir / 'embedding_metrics.csv')
        logger.info(f"Saved results to '{self.save_dir}/embedding_metrics.png'")
        return results_dict


def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT"
) -> Dict:
    
    # if adata.uns["neighbors"] exists, remove it to make sure the optimal 
    # clustering is calculated for the correct embedding
    # print a warning for the user
    if "neighbors" in adata.uns:        
        logger.warning(f"neighbors in adata.uns found \n {adata.uns['neighbors']} "
                    "\nto make sure the optimal clustering is calculated for the "
                    "correct embedding, removing neighbors from adata.uns."
                    "\nOverwriting calculation of neighbors with "
                    f"sc.pp.neighbors(adata, use_rep={embedding_key}).")
        adata.uns.pop("neighbors", None)
        sc.pp.neighbors(adata, use_rep=embedding_key)
        logger.info("neighbors in adata.uns removed, new neighbors calculated: "
                 f"{adata.uns['neighbors']}")


    # in case just one batch scib.metrics.metrics doesn't work 
    # call them separately
    results_dict = dict()
    

    logger.info('leiden')
    sc.pp.neighbors(
    adata,
    use_rep=embedding_key,
    n_neighbors=15,
    metric="euclidean", 
    random_state=0
    )
        
    sc.tl.leiden(
        adata,
        key_added="cluster",
        random_state=0,
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
    results_dict["graph_conn"] = scib.metrics.graph_connectivity(
        adata,
        label_key=label_key
    )
    
    # print('batchs')
    # Calculate this only if there are multiple batches
    if len(adata.obs[batch_key].unique()) > 1:
        results_dict["ASW_batch"] = scib.metrics.silhouette(
            adata,
            batch_key,
            embedding_key,
            "euclidean"
        )

        results_dict["ASW_label/batch"] = scib.metrics.silhouette_batch(
            adata, 
            batch_key,
            label_key, 
            embed=embedding_key, 
            metric="euclidean",
            return_all=False,
            verbose=False
        )

        results_dict["PCR_batch"] = scib.metrics.pcr(
            adata,
            covariate=batch_key,
            embed=embedding_key,
            recompute_pca=True,
            n_comps=50,
            verbose=False
        )

    # print('avg_bio')
    results_dict["avg_bio"] = np.mean(
        [
            results_dict["NMI_cluster/label"],
            results_dict["ARI_cluster/label"],
            results_dict["ASW_label"],
        ]
    )

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