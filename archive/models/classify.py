import pandas as pd
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import seaborn as sns
from matplotlib import pyplot as plt

# Suppress seaborn's internal deprecation warning about 'vert' parameter
warnings.filterwarnings(
    'ignore',
    category=PendingDeprecationWarning,
    message='vert: bool will be deprecated',
    module='seaborn.categorical'
)

from evaluation.eval import eval_classifier, plot_classifier
from run.run_utils import get_split_dict
from utils.saving import save_supervised
from utils.logs_ import get_logger
from models.mil_experiment import MILExperiment
# from models.gf_finetune import GFFineTuneModel
import numpy as np


logger = get_logger()

MODEL_REGISTRY = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    'svc': SVC
    # 'gf_finetune': GFFineTuneModel
}


def save_results(pred_df, metrics_df, cls_report, saving_dir, postfix, viz=False, model_name=None, label_names=None, pred_score_full=None, plots_dir=None):
    """Utility function to save results and visualizations
    
    Args:
        pred_df: DataFrame with label, pred, pred_score columns
        metrics_df: Metrics DataFrame
        cls_report: Classification report
        saving_dir: Directory to save CSV results (metrics, predictions, reports)
        postfix: Postfix for filenames
        viz: Whether to generate visualizations
        model_name: Name of the model
        label_names: List of label names
        pred_score_full: Full 2D probability matrix for multiclass (optional)
        plots_dir: Directory to save plots (if None, uses saving_dir)
    """
    # Use plots_dir if provided, otherwise use saving_dir
    plot_save_dir = plots_dir if plots_dir else saving_dir

    if pred_df is not None:
        fname = join(saving_dir, f'cls_predictions_{postfix}.csv')
        pred_df.to_csv(fname)

    if metrics_df is not None:
        fname = join(saving_dir, f'cls_metrics_{postfix}.csv')
        metrics_df.to_csv(fname)

    if cls_report is not None:
        fname = join(saving_dir, f'cls_report_{postfix}.csv')
        pd.DataFrame(cls_report).transpose().to_csv(fname)

    if viz and pred_df is not None:
        # For multiclass, use full probability matrix if provided
        if pred_score_full is not None and len(label_names) > 2:
            pred_score_for_plot = pred_score_full
        else:
            pred_score_for_plot = pred_df['pred_score'].values
        
        fig = plot_classifier(pred_df['label'].values, pred_df['pred'].values, pred_score_for_plot,
                              estimator_name=model_name, label_names=label_names)
        fname = join(plot_save_dir, f'cls_metrics_{postfix}.png')
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()


def get_binary_pred_score(y_pred_score, n_classes=2):
    """
    Safely extract prediction scores for binary classification.

    Handles cases where predict_proba returns (n_samples, 1) instead of (n_samples, 2)
    when only one class is present in training data.

    Args:
        y_pred_score: Array of prediction probabilities from predict_proba
        n_classes: Expected number of classes (default: 2 for binary)

    Returns:
        Array of scores for the positive class (or the only class if single-class)
    """
    y_pred_score = np.asarray(y_pred_score)

    if y_pred_score.ndim == 1:
        # Already 1D, return as is
        return y_pred_score
    elif y_pred_score.shape[1] == 1:
        # Only one class present, return the single probability
        return y_pred_score[:, 0]
    elif y_pred_score.shape[1] == 2:
        # Binary classification, return positive class probability
        return y_pred_score[:, 1]
    else:
        # Multiclass: return max probability (or could be customized)
        return y_pred_score.max(axis=1)


def train_classifier(X_train, y_train, X_test, y_test, model_name='random_forest'):
    """Train a classifier and return predictions"""
    model_cls = MODEL_REGISTRY.get(model_name, RandomForestClassifier)
    model = model_cls(probability=True) if model_name == 'svc' else model_cls()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_score_test = model.predict_proba(X_test)

    y_pred_train = model.predict(X_train)
    y_pred_score_train = model.predict_proba(X_train)

    # return model, y_test, y_pred, y_pred_score
    return model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test


class ClassifierPipeline:
    def __init__(self, params):
        self.params = params['params']
        logger.info(f'ClassifierPipeline {params}')

        self.saving_dir = self.params['save_dir']
        self.plots_dir = self.params.get('plots_dir', self.saving_dir)  # Use plots_dir if provided
        self.model_name = self.params['model']
        # self.model_dir = self.params['model_dir']
        self.viz = params['viz']
        self.evaluate = params['eval']
        self.embedding_col = self.params['embedding_col']
        # dictionary e.g. # {'Pre': 0, 'Post': 1}
        self.label_map = self.params['label_map']
        self.cv = self.params.get('cv', False)
        self.onesplit = self.params.get('onesplit', False)
        self.cls_level = self.params.get('cls_level', 'patient')
        # Default to ['avg', 'mil'] if train_funcs not specified or empty (common pattern in configs)
        self.train_funcs = self.params.get('train_funcs') or ['avg', 'mil']
        self.model = None
        # self.label_encoder = LabelEncoder()
        self.label_names = None

    def encode_labels(self):
        """Encode labels using LabelEncoder"""
        # Ensure we have a copy to avoid ImplicitModificationWarning
        if hasattr(self.adata, 'is_view') and self.adata.is_view:
            self.adata = self.adata.copy()
        
        # self.adata.obs['label'] = self.label_encoder.fit_transform(self.adata.obs[self.label_key])
        self.adata.obs['label'] = self.adata.obs[self.label_key].map(
            self.label_map)  # {'Pre': 0, 'Post': 1}
        # self.label_names = list(self.label_encoder.classes_)
        self.label_names = list(self.label_map.keys())
        logger.info(f"Label classes: {self.label_names}")

    def get_splits_cv(self, n_splits_default=5):
        """Get cross-validation splits
        
        Args:
            n_splits_default: Number of CV folds to generate if no predefined splits exist (default: 5)
        """
        # Use predefined CV splits if available
        if hasattr(self.data_loader, 'cv_split_dict') and self.data_loader.cv_split_dict is not None:
            cv = self.data_loader.cv_split_dict
            n_splits = cv['n_splits']
            id_column = cv['id_column']

            train_ids_list = []
            test_ids_list = []
            for i in range(n_splits):
                train_ids = cv[f'fold_{i+1}']['train_ids']
                test_ids = cv[f'fold_{i+1}']['test_ids']
                train_ids_list.append(train_ids)
                test_ids_list.append(test_ids)
            logger.info(f"Using predefined CV splits: {n_splits} folds, id_column='{id_column}'")
            return id_column, n_splits, train_ids_list, test_ids_list
        
        # Fallback: generate CV splits automatically using batch_key
        if hasattr(self.data_loader, 'batch_key') and self.data_loader.batch_key is not None:
            id_column = self.data_loader.batch_key
            unique_batches = self.adata.obs[id_column].unique().tolist()
            
            # Use KFold to generate splits
            kf = KFold(n_splits=n_splits_default, shuffle=True, random_state=42)
            train_ids_list = []
            test_ids_list = []
            
            for train_idx, test_idx in kf.split(unique_batches):
                train_batches = [unique_batches[i] for i in train_idx]
                test_batches = [unique_batches[i] for i in test_idx]
                train_ids_list.append(train_batches)
                test_ids_list.append(test_batches)
            
            logger.info(
                f"Generated {n_splits_default}-fold CV splits automatically using batch_key '{id_column}': "
                f"{len(unique_batches)} unique batches"
            )
            return id_column, n_splits_default, train_ids_list, test_ids_list
        
        # Last resort: raise an error
        raise ValueError(
            "No cv_splits or batch_key available for CV. "
            "Please provide cv_splits or batch_key in the dataset configuration."
        )

    def get_splits_loocv(self):
        """Get cross-validation splits"""
        id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_cv()
        flattened_list = []
        for tr, tst in zip(train_ids_list, test_ids_list):
            flattened_list.extend(tr)
            flattened_list.extend(tst)
        unique_elements = list(set(flattened_list))

        # LOOCV
        train_ids_list = []
        test_ids_list = []
        for i, test_id in enumerate(unique_elements):
            train_ids = unique_elements[:i] + unique_elements[i+1:]
            train_ids_list.append(train_ids)
            test_ids_list.append([test_id])

        n_splits = len(train_ids_list)

        return id_column, n_splits, train_ids_list, test_ids_list

    def split_data(self, id_column, train_ids, test_ids):
        """Split data into train and test sets"""
        # Copy to avoid ImplicitModificationWarning when modifying views
        adata_test = self.adata[self.adata.obs[id_column].isin(test_ids)].copy()
        adata_train = self.adata[self.adata.obs[id_column].isin(train_ids)].copy()
        return adata_train, adata_test

    def get_split_data(self):
        """Get train-test split data (single split, not CV)"""
        if hasattr(self.data_loader, 'train_test_split_dict') and self.data_loader.train_test_split_dict is not None:
            split_dict = self.data_loader.train_test_split_dict
            test_ids = split_dict['train_test_split']['test_ids']
            train_ids = split_dict['train_test_split']['train_ids']
            id_column = split_dict['id_column']
            # Ensure lists (JSON may have different types)
            train_ids = list(train_ids) if not isinstance(train_ids, list) else train_ids
            test_ids = list(test_ids) if not isinstance(test_ids, list) else test_ids
            logger.info(f"Using train_test_split_dict: test_ids={test_ids}")
            return id_column, train_ids, test_ids
        
        # Fallback: use CV splits if available (use first fold)
        if hasattr(self.data_loader, 'cv_split_dict') and self.data_loader.cv_split_dict is not None:
            cv = self.data_loader.cv_split_dict
            id_column = cv['id_column']
            train_ids = cv['fold_1']['train_ids']
            test_ids = cv['fold_1']['test_ids']
            # Ensure lists (JSON may have different types)
            train_ids = list(train_ids) if not isinstance(train_ids, list) else train_ids
            test_ids = list(test_ids) if not isinstance(test_ids, list) else test_ids
            logger.info(f"Using CV split (fold_1) as single split: test_ids={test_ids}")
            return id_column, train_ids, test_ids
        
        # Fallback: generate CV splits and use first fold, or create single split
        if hasattr(self.data_loader, 'batch_key') and self.data_loader.batch_key is not None:
            id_column = self.data_loader.batch_key
            unique_batches = self.adata.obs[id_column].unique().tolist()
            
            # If CV is enabled, generate CV splits and use first fold
            if self.cv:
                _, _, train_ids_list, test_ids_list = self.get_splits_cv()
                train_ids = train_ids_list[0]
                test_ids = test_ids_list[0]
                # Ensure lists (get_splits_cv should return lists, but be safe)
                train_ids = list(train_ids) if not isinstance(train_ids, list) else train_ids
                test_ids = list(test_ids) if not isinstance(test_ids, list) else test_ids
                logger.info(f"Generated CV splits and using fold_1 as single split: {len(train_ids)} train, {len(test_ids)} test batches")
            else:
                # Create single train-test split
                train_ids, test_ids = train_test_split(
                    unique_batches, 
                    test_size=0.2, 
                    random_state=42,
                    shuffle=True
                )
                # Convert to list if not already (train_test_split may return numpy arrays or lists)
                train_ids = train_ids.tolist() if hasattr(train_ids, 'tolist') else list(train_ids)
                test_ids = test_ids.tolist() if hasattr(test_ids, 'tolist') else list(test_ids)
                logger.info(f"Created default split using batch_key '{id_column}': {len(train_ids)} train, {len(test_ids)} test batches")
            
            return id_column, train_ids, test_ids
        
        # Last resort: raise an error with helpful message
        raise ValueError(
            "No train_test_split, cv_splits, or batch_key available. "
            "Please provide one of: train_test_split, cv_splits, or batch_key in the dataset configuration."
        )

    def prepare_data(self, adata_train, adata_test, id_column):
        """Prepare data for training"""
        # Ensure we have copies to avoid ImplicitModificationWarning
        # split_data() returns copies, but check here as well for safety
        if hasattr(adata_train, 'is_view') and adata_train.is_view:
            adata_train = adata_train.copy()
        if hasattr(adata_test, 'is_view') and adata_test.is_view:
            adata_test = adata_test.copy()
        
        adata_train.obs['sample_id'] = adata_train.obs[id_column]
        adata_test.obs['sample_id'] = adata_test.obs[id_column]
        return adata_train, adata_test

    def train(self, loader):
        self.data_loader = loader
        self.adata = loader.adata

        if self.cls_level == 'patient':
            self.train_sample(loader)
        elif self.cls_level == 'cell':
            # Encode labels first (convert string labels to integers)
            self.label_key = loader.label_key
            self.encode_labels()
            
            # Support CV for cell-level predictions (splits at patient/batch level)
            if self.cv:
                # Get CV splits (at patient/batch level to avoid data leakage)
                if self.cv == 'loocv':
                    id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_loocv()
                else:
                    id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_cv()
                
                # Run CV for cell-level predictions
                self.__train_cell_cv(id_column, n_splits, train_ids_list, test_ids_list)
            else:
                # Single split training
                id_column, train_ids, test_ids = self.get_split_data()
                adata_train, adata_test = self.split_data(
                    id_column, train_ids, test_ids)
                adata_train, adata_test = self.prepare_data(
                    adata_train, adata_test, id_column)
                self.__train_cell(adata_train, adata_test,
                                  evaluate=self.evaluate, viz=self.viz)

    def train_sample(self, loader):
        """Main training pipeline"""

        self.data_loader = loader  # Store reference for validation
        self.label_key = loader.label_key
        self.encode_labels()
        train_func_map = {'vote': self.__train_vote,
                          'avg': self.__train_avg_expression, 'mil': self.__train_mil}
        train_fcs = []
        prefix = []
        for fnc in self.train_funcs:
            train_fcs.append(train_func_map[fnc])
            prefix.append(fnc)

        # Single split training
        if self.onesplit:
            id_column, train_ids, test_ids = self.get_split_data()
            adata_train, adata_test = self.split_data(
                id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(
                adata_train, adata_test, id_column)
            for fnc in train_fcs:
                fnc(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_mil(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_vote(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_avg_expression(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)

        # Cross-validation training
        if self.cv:
            if self.cv == 'loocv':
                id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_loocv()
            else:
                id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_cv()

            for tr_f, pre in zip(train_fcs, prefix):
                self.__train_cv(tr_f, id_column, n_splits,
                                train_ids_list, test_ids_list, pre)

    def __train_cv(self, train_fnc, id_column, n_splits, train_ids_list, test_ids_list, prefix=''):
        """Run cross-validation training"""
        pred_list = []
        metrics_list = []
        per_class_metrics_list = []  # Collect per-class metrics from each fold
        pred_list_train = []
        metrics_list_train = []
        per_class_metrics_list_train = []  # Collect per-class metrics from train set
        logger.info(f'Running crossvalidation with {n_splits} folds')

        # Pre-validate all folds
        logger.info('Validating CV folds...')
        for i in range(n_splits):
            train_ids, test_ids = train_ids_list[i], test_ids_list[i]
            adata_train_check, adata_test_check = self.split_data(
                id_column, train_ids, test_ids)

            # Check label distribution (before encoding)
            label_col = self.data_loader.label_key
            unique_train = adata_train_check.obs[label_col].unique()
            unique_test = adata_test_check.obs[label_col].unique()

            if len(unique_train) < len(self.label_names):
                missing = set(self.label_names) - set(unique_train)
                logger.warning(
                    f"Fold {i+1}: Train set missing classes {missing}. "
                    f"Present: {unique_train}, Expected: {self.label_names}"
                )

            if len(unique_test) < len(self.label_names):
                missing = set(self.label_names) - set(unique_test)
                logger.warning(
                    f"Fold {i+1}: Test set missing classes {missing}. "
                    f"Present: {unique_test}, Expected: {self.label_names}"
                )

        logger.info('Starting CV training...')
        for i in range(n_splits):
            logger.info(f'---------- fold {i+1}----------')
            train_ids, test_ids = train_ids_list[i], test_ids_list[i]
            adata_train, adata_test = self.split_data(
                id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(
                adata_train, adata_test, id_column)

            # pred_df, metric_df = train_fnc(adata_train, adata_test, f'fold_{i+1}_')
            # pred_df_train, pred_df_test, metrics_df_train, metrics_df_test = train_fnc(adata_train, adata_test, f'fold_{i+1}_True
            result = train_fnc(adata_train, adata_test, evaluate=True, viz=False)
            if len(result) == 4:
                pred_df_train, pred_df_test, metrics_df_train, metrics_df_test = result
                per_class_test = None
                per_class_train = None
            else:
                # New format: includes per-class metrics
                pred_df_train, pred_df_test, metrics_df_train, metrics_df_test, per_class_train, per_class_test = result

            pred_df_test['fold'] = f'fold_{i+1}'
            metrics_df_test['fold'] = f'fold_{i+1}'

            pred_df_train['fold'] = f'fold_{i+1}'
            metrics_df_train['fold'] = f'fold_{i+1}'
            
            if per_class_test is not None:
                per_class_test['fold'] = f'fold_{i+1}'
                per_class_metrics_list.append(per_class_test)
            if per_class_train is not None:
                per_class_train['fold'] = f'fold_{i+1}'
                per_class_metrics_list_train.append(per_class_train)

            pred_list.append(pred_df_test)
            metrics_list.append(metrics_df_test)

            pred_list_train.append(pred_df_train)
            metrics_list_train.append(metrics_df_train)

        preds = pd.concat(pred_list)
        metrics = pd.concat(metrics_list)

        preds_train = pd.concat(pred_list_train)
        metrics_train = pd.concat(metrics_list_train)
        
        # Aggregate per-class metrics if available
        per_class_agg = None
        per_class_agg_train = None
        if per_class_metrics_list:
            per_class_all = pd.concat(per_class_metrics_list)
            # Compute mean and std across folds for each metric
            per_class_mean = per_class_all.groupby('Metrics')[self.model_name].mean()
            per_class_std = per_class_all.groupby('Metrics')[self.model_name].std()
            per_class_agg = pd.DataFrame({
                'mean': per_class_mean,
                'std': per_class_std
            })
        if per_class_metrics_list_train:
            per_class_all_train = pd.concat(per_class_metrics_list_train)
            per_class_mean_train = per_class_all_train.groupby('Metrics')[self.model_name].mean()
            per_class_std_train = per_class_all_train.groupby('Metrics')[self.model_name].std()
            per_class_agg_train = pd.DataFrame({
                'mean': per_class_mean_train,
                'std': per_class_std_train
            })

        save_dir = join(self.saving_dir, 'cv')
        plots_cv_dir = join(self.plots_dir, 'cv')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(plots_cv_dir, exist_ok=True)

        preds.to_csv(join(save_dir, f'{prefix}_cv_predictions.csv'))
        metrics.to_csv(join(save_dir, f'{prefix}_cv_metrics.csv'))
        
        # Compute and save aggregated overall metrics (mean ± std)
        metrics_agg = pd.DataFrame({
            'mean': metrics.groupby('Metrics')[self.model_name].mean(),
            'std': metrics.groupby('Metrics')[self.model_name].std()
        })
        metrics_agg.to_csv(join(save_dir, f'{prefix}_cv_metrics_aggregated.csv'))
        
        # Save per-class aggregated metrics
        if per_class_agg is not None:
            per_class_agg.to_csv(join(save_dir, f'{prefix}_cv_per_class_metrics_aggregated.csv'))

        preds_train.to_csv(
            join(save_dir, f'{prefix}_cv_predictions_train.csv'))
        metrics_train.to_csv(join(save_dir, f'{prefix}_cv_metrics_train.csv'))
        
        # Aggregated train metrics
        metrics_agg_train = pd.DataFrame({
            'mean': metrics_train.groupby('Metrics')[self.model_name].mean(),
            'std': metrics_train.groupby('Metrics')[self.model_name].std()
        })
        metrics_agg_train.to_csv(join(save_dir, f'{prefix}_cv_metrics_train_aggregated.csv'))
        
        if per_class_agg_train is not None:
            per_class_agg_train.to_csv(join(save_dir, f'{prefix}_cv_per_class_metrics_train_aggregated.csv'))

        # Plot metrics
        metrics.fillna(0, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Metrics', y=self.model_name, data=metrics, orientation='vertical')
        plt.title('Cross-Validation Metric Distribution')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(plots_cv_dir, f'{prefix}_cv_metrics_boxplot.png'))
        plt.close()

        metrics_train.fillna(0, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Metrics', y=self.model_name, data=metrics_train, orientation='vertical')
        plt.title('Cross-Validation Metric Distribution on Training set')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(plots_cv_dir, f'{prefix}_cv_metrics_boxplot_train.png'))
        plt.close()

        return preds, metrics

    def __train_avg_expression(self, adata_train, adata_test, evaluate=False, viz=False):
        """Train and evaluate using average expression per sample"""
        logger.info('Training model (Average Embedding Per Sample)')

        # Average embedding per sample
#         def aggregate_embeddings(adata):
#             emb = adata.obsm[self.embedding_col]

#             sample_ids = list(adata.obs['sample_id'].values)
#             print(sample_ids)
#             df_emb = pd.DataFrame(emb)
#             df_emb['sample_id'] = sample_ids
#             print(df_emb.head())
#             mean_emb = df_emb.groupby('sample_id').mean()
#             # mean_emb = df_emb.groupby(df_emb.index).mean()

#             labels = adata.obs[['sample_id', 'label']].drop_duplicates().set_index('sample_id')
#             return mean_emb.loc[labels.index], labels['label']
        def aggregate_embeddings(adata):

            # Validate embedding exists
            if self.embedding_col not in adata.obsm:
                raise KeyError(
                    f"Embedding '{self.embedding_col}' not found in adata.obsm")

            emb = adata.obsm[self.embedding_col]

            # Convert sparse to dense if necessary
            if not isinstance(emb, np.ndarray):
                emb = emb.toarray()

            # Validate sample-label consistency
            # Check for samples with multiple labels (can happen in real data)
            label_counts = adata.obs.groupby('sample_id', observed=True)[
                'label'].nunique()
            samples_with_multiple_labels = label_counts[label_counts > 1]
            
            if len(samples_with_multiple_labels) > 0:
                logger.warning(
                    f"Found {len(samples_with_multiple_labels)} samples with multiple labels. "
                    f"Using most frequent label for each sample. "
                    f"Affected samples: {samples_with_multiple_labels.index.tolist()[:10]}"
                    f"{'...' if len(samples_with_multiple_labels) > 10 else ''}"
                )

            sample_ids = adata.obs['sample_id'].values
            df_emb = pd.DataFrame(emb, index=sample_ids)

            mean_emb = df_emb.groupby(df_emb.index, observed=True).mean()

            # Get labels - use mode (most frequent) for samples with multiple labels
            labels = adata.obs.groupby('sample_id', observed=True)['label'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
            ).to_frame('label')
            
            labels = labels.loc[labels.index.intersection(mean_emb.index)]

            return mean_emb.loc[labels.index], labels['label']

        X_train, y_train = aggregate_embeddings(adata_train)
        X_test, y_test = aggregate_embeddings(adata_test)

        # Validate that both classes are present
        unique_train = np.unique(y_train)
        unique_test = np.unique(y_test)

        logger.info(
            f"Train set: {len(X_train)} samples, classes: {unique_train}, class counts: {np.bincount(y_train)}")
        logger.info(
            f"Test set: {len(X_test)} samples, classes: {unique_test}, class counts: {np.bincount(y_test)}")

        if len(unique_train) < len(self.label_names):
            missing_classes = set(self.label_names) - set(unique_train)
            logger.warning(
                f"WARNING: Train set missing classes: {missing_classes}. "
                f"Only {len(unique_train)}/{len(self.label_names)} classes present. "
                f"This may cause issues with prediction probabilities."
            )

        if len(unique_test) < len(self.label_names):
            missing_classes = set(self.label_names) - set(unique_test)
            logger.warning(
                f"WARNING: Test set missing classes: {missing_classes}. "
                f"Only {len(unique_test)}/{len(self.label_names)} classes present. "
                f"Metrics may be limited."
            )

        # Train classifier
        self.model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test = train_classifier(
            X_train, y_train, X_test, y_test, model_name=self.model_name)

        # Safely extract prediction scores (handles single-class cases)
        pred_score_test = get_binary_pred_score(
            y_pred_score_test, n_classes=len(self.label_names))
        pred_score_train = get_binary_pred_score(
            y_pred_score_train, n_classes=len(self.label_names))

        pred_df_test = pd.DataFrame({'label': y_test, 'pred': y_pred_test,
                                     'pred_score': pred_score_test
                                     }, index=X_test.index)

        pred_df_train = pd.DataFrame({'label': y_train, 'pred': y_pred_train,
                                      'pred_score': pred_score_train
                                      }, index=X_train.index)

        metrics_df_test, cls_report_test, per_class_df_test = eval_classifier(pred_df_test['label'], pred_df_test['pred'], pred_df_test['pred_score'],
                                                           estimator_name=self.model_name, label_names=self.label_names)

        metrics_df_train, cls_report_train, per_class_df_train = eval_classifier(pred_df_train['label'], pred_df_train['pred'], pred_df_train['pred_score'],
                                                             estimator_name=self.model_name, label_names=self.label_names)

        # Save per-class metrics
        if evaluate:
            postfix = "avg_expr"
            save_results(pred_df_test, metrics_df_test, cls_report_test,
                         self.saving_dir, postfix, viz, self.model_name, self.label_names, plots_dir=self.plots_dir)
            if per_class_df_test is not None:
                per_class_df_test.to_csv(join(self.saving_dir, f'{self.model_name}_per_class_metrics_{postfix}.csv'))
            
            postfix = "avg_expr_train"
            save_results(pred_df_train, metrics_df_train, cls_report_train,
                         self.saving_dir, postfix, viz, self.model_name, self.label_names, plots_dir=self.plots_dir)
            if per_class_df_train is not None:
                per_class_df_train.to_csv(join(self.saving_dir, f'{self.model_name}_per_class_metrics_{postfix}.csv'))

        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test, per_class_df_train, per_class_df_test

    def __train_mil(self, adata_train, adata_test, evaluate=False, viz=False):
        """Multi-instance learning training"""
        logger.info('Training model (Multi instance Learning (MIL))')

        exp = MILExperiment(embedding_col=self.embedding_col, label_key='label',
                            patient_key="sample_id", celltype_key="cell_type")
        exp.train(adata_train)

        # test
        pids, y_true, preds, pred_scores, metrics_test = exp.evaluate(
            adata_test)
        # pids, y_true, preds, pred_scores, metrics
        pred_df_test = pd.DataFrame(
            {'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})
        # train
        pids, y_true, preds, pred_scores, metrics_train = exp.evaluate(
            adata_train)
        pred_df_train = pd.DataFrame(
            {'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})
        metrics_df_test, cls_report_test, per_class_df_test = eval_classifier(
            pred_df_test['label'], pred_df_test['pred'], pred_df_test['pred_score'], estimator_name=self.model_name, label_names=self.label_names)

        metrics_df_train, cls_report_train, per_class_df_train = eval_classifier(
            pred_df_train['label'], pred_df_train['pred'], pred_df_train['pred_score'], estimator_name=self.model_name, label_names=self.label_names)

        if evaluate:

            save_results(pred_df_test, metrics_df_test, cls_report_test, self.saving_dir,
                         postfix='mil', viz=viz, model_name=self.model_name, label_names=self.label_names, plots_dir=self.plots_dir)
            if per_class_df_test is not None:
                per_class_df_test.to_csv(join(self.saving_dir, f'{self.model_name}_per_class_metrics_mil.csv'))

            save_results(pred_df_train, metrics_df_train, cls_report_train, self.saving_dir,
                         postfix='mil_train', viz=viz, model_name=self.model_name, label_names=self.label_names, plots_dir=self.plots_dir)
            if per_class_df_train is not None:
                per_class_df_train.to_csv(join(self.saving_dir, f'{self.model_name}_per_class_metrics_mil_train.csv'))

        if viz:
            from models.attention_viz import print_attention_summary, visualize_attention_comprehensive, export_top_cells_table
            results = visualize_attention_comprehensive(
                exp,
                adata_test,
                top_k=20,
                save_path=join(self.plots_dir, "attention_analysis.png")
            )

            # Print summary
            print_attention_summary(results)

            # Export top cells for further analysis
            top_cells_df = export_top_cells_table(
                results['attention_df'],
                top_k=20,
                save_path=join(self.saving_dir, "top_attention_cells.csv")
            )

        # return pred_df, metrics_df if evaluate else None
        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test, per_class_df_train, per_class_df_test

    def __train_cell(self, adata_train, adata_test, evaluate=False, viz=False):
        """Cell-level predictions training"""
        logger.info('Training model')
        X_train = adata_train.obsm[self.embedding_col]
        X_test = adata_test.obsm[self.embedding_col]
        
        # Extract labels and ensure they're integers
        y_train = adata_train.obs['label'].values
        y_test = adata_test.obs['label'].values
        
        # Convert to integers if they're not already (handles string labels)
        if y_train.dtype == object or y_train.dtype.name.startswith('string'):
            # Map string labels to integers using label_map
            y_train = np.array([self.label_map.get(str(label), label) for label in y_train])
            y_test = np.array([self.label_map.get(str(label), label) for label in y_test])
        else:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

        # Validate that both classes are present
        unique_train = np.unique(y_train)
        unique_test = np.unique(y_test)

        # Get class counts using value_counts for pandas or bincount for numpy
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()

        logger.info(
            f"Train set: {len(X_train)} cells, classes: {unique_train}, class counts: {train_counts.to_dict()}")
        logger.info(
            f"Test set: {len(X_test)} cells, classes: {unique_test}, class counts: {test_counts.to_dict()}")

        if len(unique_train) < len(self.label_names):
            missing_classes = set(self.label_names) - set(unique_train)
            logger.warning(
                f"WARNING: Train set missing classes: {missing_classes}. "
                f"Only {len(unique_train)}/{len(self.label_names)} classes present. "
                f"This may cause issues with prediction probabilities."
            )

        if len(unique_test) < len(self.label_names):
            missing_classes = set(self.label_names) - set(unique_test)
            logger.warning(
                f"WARNING: Test set missing classes: {missing_classes}. "
                f"Only {len(unique_test)}/{len(self.label_names)} classes present. "
                f"Metrics may be limited."
            )

        # self.model, y_test, y_pred, y_pred_score = train_classifier(X_train, y_train, X_test, y_test,
        #                                                           model_name=self.model_name)
        self.model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test = train_classifier(X_train, y_train, X_test, y_test,
                                                                                                                         model_name=self.model_name)

        # Safely extract prediction scores (handles single-class cases)
        pred_score_test = get_binary_pred_score(
            y_pred_score_test, n_classes=len(self.label_names))
        pred_score_train = get_binary_pred_score(
            y_pred_score_train, n_classes=len(self.label_names))

        adata_test.obs['pred'] = y_pred_test
        adata_test.obs['pred_score'] = pred_score_test

        adata_train.obs['pred'] = y_pred_train
        adata_train.obs['pred_score'] = pred_score_train

        if evaluate:
            save_dir = join(self.saving_dir, 'cell_level_pred')
            os.makedirs(save_dir, exist_ok=True)

            adata_test.obs.to_csv(join(save_dir, f'cell_pred_test.csv'))
            adata_train.obs.to_csv(join(save_dir, f'cell_pred_train.csv'))

            metrics_df_test, cls_report_test, per_class_df_test = eval_classifier(
                y_test, y_pred_test, y_pred_score_test, estimator_name=self.model_name, label_names=self.label_names)
            # Create proper DataFrame with label, pred, and pred_score columns
            pred_df_test = pd.DataFrame({
                'label': y_test,
                'pred': y_pred_test,
                'pred_score': pred_score_test  # 1D array for CSV
            }, index=adata_test.obs.index)
            # For multiclass, pass full 2D probability matrix for visualization
            pred_score_full_test = y_pred_score_test if len(self.label_names) > 2 else None
            save_results(pred_df_test, metrics_df_test, cls_report_test,
                         save_dir, 'cell', viz, self.model_name, self.label_names, pred_score_full=pred_score_full_test, plots_dir=self.plots_dir)
            if per_class_df_test is not None:
                per_class_df_test.to_csv(join(save_dir, f'{self.model_name}_per_class_metrics_cell.csv'))

            metrics_df_train, cls_report_train, per_class_df_train = eval_classifier(
                y_train, y_pred_train, y_pred_score_train, estimator_name=self.model_name, label_names=self.label_names)
            # Create proper DataFrame with label, pred, and pred_score columns
            pred_df_train = pd.DataFrame({
                'label': y_train,
                'pred': y_pred_train,
                'pred_score': pred_score_train  # 1D array for CSV
            }, index=adata_train.obs.index)
            # For multiclass, pass full 2D probability matrix for visualization
            pred_score_full_train = y_pred_score_train if len(self.label_names) > 2 else None
            save_results(pred_df_train, metrics_df_train, cls_report_train, save_dir,
                         'cell_train', viz, self.model_name, self.label_names, pred_score_full=pred_score_full_train, plots_dir=self.plots_dir)
            if per_class_df_train is not None:
                per_class_df_train.to_csv(join(save_dir, f'{self.model_name}_per_class_metrics_cell_train.csv'))

            # return adata_test.obs, metrics_df
            return adata_train.obs, adata_test.obs, metrics_df_train, metrics_df_test, per_class_df_train, per_class_df_test

        return adata_test.obs, None, None, None, None, None

    def __train_cell_cv(self, id_column, n_splits, train_ids_list, test_ids_list):
        """Run cross-validation for cell-level predictions
        
        Splits are done at patient/batch level to avoid data leakage (all cells from 
        the same patient stay in the same fold).
        
        Args:
            id_column: Column name to use for splitting (e.g., 'sample_id', 'batch_key')
            n_splits: Number of CV folds
            train_ids_list: List of train ID lists for each fold
            test_ids_list: List of test ID lists for each fold
        """
        pred_list = []
        metrics_list = []
        per_class_metrics_list = []
        pred_list_train = []
        metrics_list_train = []
        per_class_metrics_list_train = []
        
        logger.info(f'Running cell-level cross-validation with {n_splits} folds (splitting by {id_column})')
        
        # Pre-validate all folds
        logger.info('Validating CV folds...')
        for i in range(n_splits):
            train_ids, test_ids = train_ids_list[i], test_ids_list[i]
            adata_train_check, adata_test_check = self.split_data(
                id_column, train_ids, test_ids)
            
            # Check label distribution
            unique_train = adata_train_check.obs[self.label_key].unique()
            unique_test = adata_test_check.obs[self.label_key].unique()
            
            if len(unique_train) < len(self.label_names):
                missing = set(self.label_names) - set(unique_train)
                logger.warning(
                    f"Fold {i+1}: Train set missing classes {missing}. "
                    f"Present: {unique_train}, Expected: {self.label_names}"
                )
            
            if len(unique_test) < len(self.label_names):
                missing = set(self.label_names) - set(unique_test)
                logger.warning(
                    f"Fold {i+1}: Test set missing classes {missing}. "
                    f"Present: {unique_test}, Expected: {self.label_names}"
                )
        
        logger.info('Starting cell-level CV training...')
        for i in range(n_splits):
            logger.info(f'---------- Cell-level CV fold {i+1}/{n_splits} ----------')
            train_ids, test_ids = train_ids_list[i], test_ids_list[i]
            adata_train, adata_test = self.split_data(
                id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(
                adata_train, adata_test, id_column)
            
            # Train cell-level classifier on this fold
            cell_pred_train, cell_pred_test, metrics_df_train, metrics_df_test, per_class_df_train, per_class_df_test = self.__train_cell(
                adata_train, adata_test, evaluate=True, viz=False)
            
            # Add fold identifier
            if metrics_df_test is not None:
                metrics_df_test['fold'] = f'fold_{i+1}'
                metrics_list.append(metrics_df_test)
            if metrics_df_train is not None:
                metrics_df_train['fold'] = f'fold_{i+1}'
                metrics_list_train.append(metrics_df_train)
            
            if per_class_df_test is not None:
                per_class_df_test['fold'] = f'fold_{i+1}'
                per_class_metrics_list.append(per_class_df_test)
            if per_class_df_train is not None:
                per_class_df_train['fold'] = f'fold_{i+1}'
                per_class_metrics_list_train.append(per_class_df_train)
            
            # Add fold to predictions
            if cell_pred_test is not None:
                cell_pred_test['fold'] = f'fold_{i+1}'
                pred_list.append(cell_pred_test)
            if cell_pred_train is not None:
                cell_pred_train['fold'] = f'fold_{i+1}'
                pred_list_train.append(cell_pred_train)
        
        # Aggregate results
        if not metrics_list:
            logger.warning("No metrics collected from CV folds")
            return
        
        metrics = pd.concat(metrics_list)
        metrics_train = pd.concat(metrics_list_train) if metrics_list_train else None
        
        # Aggregate overall metrics (mean ± std)
        metrics_agg = pd.DataFrame({
            'mean': metrics.groupby('Metrics')[self.model_name].mean(),
            'std': metrics.groupby('Metrics')[self.model_name].std()
        })
        
        if metrics_train is not None:
            metrics_agg_train = pd.DataFrame({
                'mean': metrics_train.groupby('Metrics')[self.model_name].mean(),
                'std': metrics_train.groupby('Metrics')[self.model_name].std()
            })
        
        # Aggregate per-class metrics if available
        per_class_agg = None
        per_class_agg_train = None
        if per_class_metrics_list:
            per_class_all = pd.concat(per_class_metrics_list)
            per_class_mean = per_class_all.groupby('Metrics')[self.model_name].mean()
            per_class_std = per_class_all.groupby('Metrics')[self.model_name].std()
            per_class_agg = pd.DataFrame({
                'mean': per_class_mean,
                'std': per_class_std
            })
        if per_class_metrics_list_train:
            per_class_all_train = pd.concat(per_class_metrics_list_train)
            per_class_mean_train = per_class_all_train.groupby('Metrics')[self.model_name].mean()
            per_class_std_train = per_class_all_train.groupby('Metrics')[self.model_name].std()
            per_class_agg_train = pd.DataFrame({
                'mean': per_class_mean_train,
                'std': per_class_std_train
            })
        
        # Save results
        save_dir = join(self.saving_dir, 'cell_level_pred', 'cv')
        plots_cv_dir = join(self.plots_dir, 'cell_level_pred', 'cv')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(plots_cv_dir, exist_ok=True)
        
        if pred_list:
            preds = pd.concat(pred_list)
            preds.to_csv(join(save_dir, 'cell_cv_predictions.csv'))
        if pred_list_train:
            preds_train = pd.concat(pred_list_train)
            preds_train.to_csv(join(save_dir, 'cell_cv_predictions_train.csv'))
        
        metrics.to_csv(join(save_dir, 'cell_cv_metrics.csv'))
        metrics_agg.to_csv(join(save_dir, 'cell_cv_metrics_aggregated.csv'))
        
        if metrics_train is not None:
            metrics_train.to_csv(join(save_dir, 'cell_cv_metrics_train.csv'))
            metrics_agg_train.to_csv(join(save_dir, 'cell_cv_metrics_train_aggregated.csv'))
        
        if per_class_agg is not None:
            per_class_agg.to_csv(join(save_dir, 'cell_cv_per_class_metrics_aggregated.csv'))
        if per_class_agg_train is not None:
            per_class_agg_train.to_csv(join(save_dir, 'cell_cv_per_class_metrics_train_aggregated.csv'))
        
        # Plot metrics distribution
        metrics.fillna(0, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Metrics', y=self.model_name, data=metrics, orientation='vertical')
        plt.title('Cell-Level Cross-Validation Metric Distribution')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(plots_cv_dir, 'cell_cv_metrics_boxplot.png'))
        plt.close()
        
        logger.info(f"Cell-level CV completed. Aggregated metrics saved to {save_dir}")
        logger.info(f"Mean accuracy: {metrics_agg.loc['Accuracy', 'mean']:.4f} ± {metrics_agg.loc['Accuracy', 'std']:.4f}")

    def __train_vote(self, adata_train, adata_test, evaluate=False, viz=False):
        """Majority vote predictions training"""
        logger.info('Training model (Majority Vote)')

        cell_pred_train, cell_pred_test, metrics_df_train, metrics_df_test, per_class_df_train, per_class_df_test = self.__train_cell(adata_train, adata_test,
                                                                                               evaluate=evaluate, viz=viz)

        pred_df_test, metrics_df_test = self.save_patient_level(cell_pred_test,
                                                                evaluate, viz,
                                                                postfix='',
                                                                model='vote')

        pred_df_train, metrics_df_train = self.save_patient_level(cell_pred_train,
                                                                  evaluate, viz,
                                                                  postfix='_train',
                                                                  model='vote')

        # return sample_pred_test_df, sample_metrics_test_df
        # Note: per-class metrics are already saved in save_patient_level, but we return None here
        # since vote aggregates at patient level, not cell level
        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test, None, None

    def save_patient_level(self, adata_subset, evaluate=False, viz=False, postfix="", model=""):
        """Save patient-level predictions and metrics"""
        logger.info('Saving sample level performance')

        obs = adata_subset

        def majority_vote_score(x):
            majority_class = x.value_counts().idxmax()
            return (x == majority_class).sum() / len(x)

        y_pred_score_p = obs.groupby('sample_id', observed=True)['pred_score'].mean()

        # y_pred_score_p = obs.groupby('sample_id')['pred'].agg(majority_vote_score)

        y_pred_p = obs.groupby('sample_id', observed=True)['pred'].agg(
            lambda x: x.value_counts().idxmax())
        y_test_p = obs.groupby('sample_id', observed=True)[
            'label'].first().reindex(y_pred_score_p.index)

        pred_df = pd.concat([y_test_p, y_pred_p, y_pred_score_p], axis=1)
        pred_df.columns = ['label', 'pred', 'pred_score']

        metrics_df, cls_report, per_class_df = eval_classifier(pred_df['label'], pred_df['pred'], pred_df['pred_score'],
                                                 estimator_name=self.model_name, label_names=self.label_names)

        if evaluate:
            save_results(pred_df, metrics_df, cls_report, self.saving_dir,
                         f'vote{postfix}', viz, self.model_name, self.label_names, plots_dir=self.plots_dir)
            if per_class_df is not None:
                per_class_df.to_csv(join(self.saving_dir, f'{self.model_name}_per_class_metrics_vote{postfix}.csv'))

        return pred_df, metrics_df
