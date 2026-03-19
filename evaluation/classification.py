"""
Classification evaluation module.

Evaluates classifier performance on embeddings for downstream tasks.
"""
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
from utils.logs_ import get_logger

logger = get_logger()


class ClassificationEvaluator:
    """
    Evaluates classifier performance on embeddings.
    
    Computes metrics including:
    - AUC (Area Under ROC Curve)
    - AUPRC (Average Precision)
    - F1, Accuracy, Precision, Recall
    - Detailed classification report
    """
    
    def __init__(
        self,
        y_test,
        y_pred,
        y_pred_score: Optional[np.ndarray] = None,
        label_names: list = None,
        save_dir: str = None,
        plots_dir: str = None,
        estimator_name: str = "classifier",
        plot: bool = True
    ):
        """
        Initialize classification evaluator.
        
        Args:
            y_test: True labels (1D array of ints 0..n_classes-1)
            y_pred: Predicted labels (1D array, same encoding)
            y_pred_score: Prediction scores/probabilities
                - Binary: 1D scores for positive class OR 2D probas with shape (n, 2)
                - Multiclass: 2D probas with shape (n, n_classes) matching label_names order
            label_names: List of class names in canonical order (defines n_classes)
            save_dir: Directory to save metrics CSV files
            plots_dir: Directory to save plots (if None, uses save_dir)
            estimator_name: Name of the classifier (for plots/saving)
            plot: Whether to generate and save plots
        """
        self.y_test = np.asarray(y_test)
        self.y_pred = np.asarray(y_pred)
        self.y_pred_score = y_pred_score
        self.label_names = label_names if label_names else [f"class_{i}" for i in range(len(np.unique(self.y_test)))]
        self.save_dir = Path(save_dir) if save_dir else None
        self.plots_dir = Path(plots_dir) if plots_dir else (self.save_dir if save_dir else None)
        self.estimator_name = estimator_name
        self.plot = plot
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_dir:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """
        Evaluate classifier performance.
        
        Returns:
            Tuple of (metrics_df, classification_report_dict, per_class_metrics_df)
        """
        logger.info(f'Evaluating classifier: {self.estimator_name}')
        logger.info(f'Test set shape: {self.y_test.shape}, Predicted shape: {self.y_pred.shape}')
        
        n_classes = len(self.label_names)
        all_labels = list(range(n_classes))
        
        # Compute AUC and AUPRC
        roc_auc = np.nan
        average_precision = np.nan
        
        if n_classes == 2:
            # Binary classification
            if self.y_pred_score is not None:
                yps = np.asarray(self.y_pred_score)
                if yps.ndim == 2 and yps.shape[1] >= 2:
                    yps = yps[:, 1]  # Use positive class probabilities
                yps = np.asarray(yps).reshape(-1)
                
                # Only compute if both classes present
                if np.unique(self.y_test).size > 1:
                    try:
                        fpr, tpr, _ = metrics.roc_curve(self.y_test, yps)
                        roc_auc = metrics.auc(fpr, tpr)
                    except ValueError:
                        roc_auc = np.nan
                    try:
                        average_precision = metrics.average_precision_score(self.y_test, yps)
                    except ValueError:
                        average_precision = np.nan
        else:
            # Multiclass classification
            if (self.y_pred_score is not None and 
                np.asarray(self.y_pred_score).ndim == 2 and 
                np.asarray(self.y_pred_score).shape[1] == n_classes):
                y_test_bin = label_binarize(self.y_test, classes=all_labels)
                if np.unique(self.y_test).size > 1:
                    try:
                        roc_auc = metrics.roc_auc_score(
                            y_test_bin, self.y_pred_score, 
                            average='macro', multi_class='ovr'
                        )
                    except ValueError:
                        roc_auc = np.nan
                    try:
                        average_precision = metrics.average_precision_score(
                            y_test_bin, self.y_pred_score, average='macro'
                        )
                    except ValueError:
                        average_precision = np.nan
        
        # Overall metrics
        f1 = metrics.f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        precision_score = metrics.precision_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        recall = metrics.recall_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        
        # Create metrics dataframe
        metrics_dict = {
            'AUC': roc_auc,
            'AUPRC': average_precision,
            'F1': f1,
            'Accuracy': accuracy,
            'Precision': precision_score,
            'Recall': recall,
        }
        metrics_df = pd.Series(metrics_dict, name='Metrics').to_frame()
        metrics_df.columns = [self.estimator_name]
        metrics_df.index.name = 'Metrics'
        
        # Detailed classification report
        cls_report = metrics.classification_report(
            self.y_test,
            self.y_pred,
            labels=all_labels,
            target_names=list(self.label_names),
            zero_division=0,
            output_dict=True,
        )
        
        # Extract per-class metrics from classification report
        per_class_metrics = {}
        for class_name in self.label_names:
            if class_name in cls_report:
                class_metrics = cls_report[class_name]
                per_class_metrics[f'{class_name}_precision'] = class_metrics.get('precision', np.nan)
                per_class_metrics[f'{class_name}_recall'] = class_metrics.get('recall', np.nan)
                per_class_metrics[f'{class_name}_f1'] = class_metrics.get('f1-score', np.nan)
                per_class_metrics[f'{class_name}_support'] = class_metrics.get('support', np.nan)
        
        # Create per-class metrics DataFrame
        per_class_df = pd.Series(per_class_metrics, name='Metrics').to_frame()
        per_class_df.columns = [self.estimator_name]
        per_class_df.index.name = 'Metrics'
        
        # Save results
        if self.save_dir:
            # Save overall metrics
            metrics_df.to_csv(self.save_dir / f'{self.estimator_name}_metrics.csv')
            logger.info(f"Saved classification metrics to '{self.save_dir}/{self.estimator_name}_metrics.csv'")
            
            # Save per-class metrics
            per_class_df.to_csv(self.save_dir / f'{self.estimator_name}_per_class_metrics.csv')
            logger.info(f"Saved per-class metrics to '{self.save_dir}/{self.estimator_name}_per_class_metrics.csv'")
        
        # Generate plots if requested
        if self.plot:
            self._plot_results()
        
        logger.info(f"Classification evaluation complete. Metrics: {metrics_dict}")
        return metrics_df, cls_report, per_class_df
    
    def _plot_results(self):
        """Generate and save classification plots."""
        if self.y_pred_score is None:
            logger.warning("Cannot generate plots without prediction scores.")
            return
        
        n_classes = len(self.label_names)
        y_test_bin = label_binarize(self.y_test, classes=range(n_classes))
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Create figure with subplots
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
        
        metrics_fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
        
        # Confusion Matrix
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_names).plot(ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.grid(False)
        
        # Normalized Confusion Matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=self.label_names).plot(ax=ax2)
        ax2.set_title('Normalized Confusion Matrix')
        ax2.grid(False)
        
        # ROC curve
        if n_classes == 2:
            yps = np.asarray(self.y_pred_score)
            if yps.ndim > 1:
                yps = yps[:, 1]
            fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, 0], yps)
            auc = metrics.auc(fpr, tpr)
            ax3.plot(fpr, tpr, label=f'{self.label_names[1]} (AUC={auc:.2f})')
        else:
            for i in range(n_classes):
                fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, i], self.y_pred_score[:, i])
                auc = metrics.auc(fpr, tpr)
                ax3.plot(fpr, tpr, label=f'{self.label_names[i]} (AUC={auc:.2f})')
        ax3.plot([0, 1], [0, 1], 'k--', lw=1)
        ax3.set_title('ROC Curve')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.legend(loc='lower right')
        
        # PR curve
        if n_classes == 2:
            yps = np.asarray(self.y_pred_score)
            if yps.ndim > 1:
                yps = yps[:, 1]
            precision, recall, _ = metrics.precision_recall_curve(y_test_bin[:, 0], yps)
            ap = metrics.average_precision_score(y_test_bin[:, 0], yps)
            ax4.plot(recall, precision, label=f'{self.label_names[1]} (AP={ap:.2f})')
        else:
            for i in range(n_classes):
                precision, recall, _ = metrics.precision_recall_curve(y_test_bin[:, i], self.y_pred_score[:, i])
                ap = metrics.average_precision_score(y_test_bin[:, i], self.y_pred_score[:, i])
                ax4.plot(recall, precision, label=f'{self.label_names[i]} (AP={ap:.2f})')
        ax4.set_title('Precision-Recall Curve')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.legend(loc='lower left')
        
        plt.tight_layout()
        metrics_fig.suptitle(self.estimator_name, fontsize=BIGGER_SIZE, y=1.02)
        
        # Save plot
        if self.plots_dir:
            plot_path = self.plots_dir / f'{self.estimator_name}_classification_plots.png'
            metrics_fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved classification plots to '{plot_path}'")
            plt.close(metrics_fig)
