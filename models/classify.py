import pandas as pd
from os.path import join
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import seaborn as sns
from matplotlib import pyplot as plt

from evaluation.eval import eval_classifier, plot_classifier
from run.run_utils import get_split_dict
from utils.saving import save_supervised
from utils.logs_ import get_logger
from models.mil_experiment import MILExperiment
# from models.gf_finetune import GFFineTuneModel
import numpy as np


logger = get_logger()

_MIL_YAML_MODEL_KEYS = frozenset({
    "hidden_dim",
    "attention_dim",
    "epochs",
    "lr",
    "weight_decay",
    "dropout",
    "patience",
    "temperature",
    "min_delta",
    "monitor",
    "mode",
    "device",
    "celltype_key",
    "pos_weight",
})

MODEL_REGISTRY = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    'svc': SVC
    # 'gf_finetune': GFFineTuneModel
}

def save_results(pred_df, metrics_df, cls_report, saving_dir, postfix, viz=False, model_name=None, label_names=None):
    """Utility function to save results and visualizations"""
    
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
        fig = plot_classifier(pred_df['label'], pred_df['pred'], pred_df['pred_score'], estimator_name=model_name, label_names=label_names)
        fname = join(saving_dir, f'cls_metrics_{postfix}.png')
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()

def make_classifier_estimator(model_name: str = "random_forest", *, random_state: int = 42):
    """Unfitted sklearn classifier used by ``train_classifier`` and learning-curve plots."""
    model_cls = MODEL_REGISTRY.get(model_name, RandomForestClassifier)
    if model_name == "svc":
        return model_cls(probability=True, random_state=random_state)
    if model_name == "logistic_regression":
        return model_cls(random_state=random_state, max_iter=5000)
    if model_name == "random_forest":
        return model_cls(random_state=random_state)
    return model_cls(random_state=random_state)


def train_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name='random_forest',
    *,
    random_state: int = 42,
):
    """Train a classifier and return predictions (fixed ``random_state`` for reproducible fits)."""
    model = make_classifier_estimator(model_name, random_state=random_state)
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
        self.model_name = self.params['model']
        # self.model_dir = self.params['model_dir']
        self.viz = params['viz']
        self.evaluate = params['eval']
        self.embedding_col = self.params['embedding_col']
        self.label_map = self.params['label_map'] #dictionary e.g. # {'Pre': 0, 'Post': 1}
        self.cv = self.params.get('cv', False)
        self.onesplit = self.params.get('onesplit', False)
        self.cls_level = self.params.get('cls_level', 'patient')
        self.train_funcs = self.params['train_funcs']
        self.model = None
        # self.label_encoder = LabelEncoder()
        self.label_names = None
        self.random_state = int(self.params.get('random_state', 42))

        mil_raw = self.params.get("mil")
        if mil_raw is None:
            mil_raw = {}
        elif not isinstance(mil_raw, dict):
            logger.warning("classification.params['mil'] should be a dict; ignoring.")
            mil_raw = {}
        mil_raw = dict(mil_raw)
        self.mil_debug = bool(mil_raw.pop("debug", False))
        self.mil_top_k = int(mil_raw.pop("top_k", 20))
        self.mil_holdout_fraction = float(mil_raw.pop("holdout_fraction", 0.0))
        self.mil_model_kwargs = mil_raw
        self._cv_fold = None  # set during CV for per-fold artifact names

    @staticmethod
    def _mil_patient_pos_weight(adata, patient_key: str, label_key: str):
        y = adata.obs.groupby(patient_key, observed=False)[label_key].first()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos <= 0:
            return None
        return float(n_neg / max(n_pos, 1))

    def _split_mil_train_val(self, adata, patient_key: str, label_key: str, fraction: float, seed: int):
        """Hold out a fraction of patients for MIL early stopping (monitor val_loss)."""
        if fraction <= 0:
            return adata, None
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            logger.warning("sklearn unavailable; mil holdout_fraction ignored.")
            return adata, None
        obs = adata.obs
        pids = obs[patient_key].unique()
        y_p = obs.groupby(patient_key, observed=False)[label_key].first().reindex(pids)
        try:
            tr, va = train_test_split(
                np.asarray(pids), test_size=fraction, random_state=seed, stratify=np.asarray(y_p)
            )
        except ValueError:
            tr, va = train_test_split(np.asarray(pids), test_size=fraction, random_state=seed)
        mtr = obs[patient_key].isin(set(tr)).to_numpy()
        mva = obs[patient_key].isin(set(va)).to_numpy()
        return adata[mtr].copy(), adata[mva].copy()

    def _training_diagnostics_path(self, method: str, filename: str) -> str:
        """Stable path for loss / learning-curve plots (single split vs CV fold)."""
        fold = getattr(self, "_cv_fold", None)
        if fold is not None:
            d = join(self.saving_dir, "cv")
            os.makedirs(d, exist_ok=True)
            stem, ext = os.path.splitext(filename)
            return join(d, f"{method}_fold_{fold}_{stem}{ext}")
        os.makedirs(self.saving_dir, exist_ok=True)
        return join(self.saving_dir, f"{method}_{filename}")

    def _plot_sklearn_learning_curve_log_loss(
        self,
        X_train,
        y_train,
        out_png: str,
        *,
        title: str,
        max_train_samples: Optional[int] = None,
    ) -> None:
        """
        Sklearn classifiers have no training epochs; plot **log loss** vs. training-set size
        with internal stratified CV (train score vs. CV validation score). Complements MIL
        per-epoch BCE curves for side-by-side diagnostics.
        """
        if isinstance(X_train, pd.DataFrame):
            X_use = X_train
        else:
            X_use = pd.DataFrame(np.asarray(X_train))
        if isinstance(y_train, pd.Series):
            y_use = y_train
        else:
            y_use = pd.Series(np.asarray(y_train).ravel())

        n = len(y_use)
        if n < 4:
            logger.warning("Skipping sklearn learning curve: need >= 4 training points (got %d).", n)
            return

        if max_train_samples is not None and n > max_train_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n, size=max_train_samples, replace=False)
            X_use = X_use.iloc[idx]
            y_use = y_use.iloc[idx].reset_index(drop=True)
            n = len(y_use)

        y_np = y_use.to_numpy()
        if len(np.unique(y_np)) < 2:
            logger.warning("Skipping sklearn learning curve: single class in training slice.")
            return

        counts = y_use.value_counts()
        min_class = int(counts.min())
        n_splits = min(5, max(2, min_class, n // 3))
        n_splits = min(n_splits, n // 2)
        if n_splits < 2:
            logger.warning("Skipping sklearn learning curve: insufficient samples per class.")
            return

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        n_ticks = min(12, max(3, n - 1))
        train_sizes = np.unique(np.clip((np.linspace(0.12, 1.0, n_ticks) * n).astype(int), 2, n))

        est = make_classifier_estimator(self.model_name, random_state=self.random_state)
        try:
            sizes, train_neg_ll, test_neg_ll = learning_curve(
                est,
                X_use,
                y_use,
                train_sizes=train_sizes,
                cv=cv,
                scoring="neg_log_loss",
                n_jobs=1,
            )
        except Exception as exc:
            logger.warning("sklearn learning_curve failed (%s); skipping plot.", exc)
            return

        train_loss = -np.mean(train_neg_ll, axis=1)
        val_loss = -np.mean(test_neg_ll, axis=1)
        train_std = np.std(-train_neg_ll, axis=1, ddof=0)
        val_std = np.std(-test_neg_ll, axis=1, ddof=0)

        hist = pd.DataFrame(
            {
                "n_train_samples": sizes,
                "train_log_loss_mean": train_loss,
                "train_log_loss_std": train_std,
                "cv_val_log_loss_mean": val_loss,
                "cv_val_log_loss_std": val_std,
            }
        )
        stem = out_png[:-4] if out_png.lower().endswith(".png") else out_png
        hist.to_csv(f"{stem}_history.csv", index=False)

        plt.figure(figsize=(7, 4.5))
        plt.plot(sizes, train_loss, "o-", label="train log loss (CV fit)", linewidth=1.5)
        plt.fill_between(sizes, train_loss - train_std, train_loss + train_std, alpha=0.2)
        plt.plot(sizes, val_loss, "o-", label="val log loss (CV held-out)", linewidth=1.5)
        plt.fill_between(sizes, val_loss - val_std, val_loss + val_std, alpha=0.2)
        plt.xlabel("Training samples")
        plt.ylabel("Log loss (lower is better)")
        plt.title(title)
        plt.suptitle(
            "Sklearn: no per-epoch loss — curves show learning vs. train size (internal CV).",
            fontsize=9,
            y=0.02,
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        d = os.path.dirname(out_png) or "."
        os.makedirs(d, exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()

    def encode_labels(self):
        """Encode labels using LabelEncoder"""
        # self.adata.obs['label'] = self.label_encoder.fit_transform(self.adata.obs[self.label_key])
        self.adata.obs['label'] = self.adata.obs[self.label_key].map(self.label_map) # {'Pre': 0, 'Post': 1}
        # self.label_names = list(self.label_encoder.classes_)
        self.label_names = list(self.label_map.keys())
        logger.info(f"Label classes: {self.label_names}")

    def get_splits_cv(self):
        """Get cross-validation splits"""
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
        return id_column, n_splits, train_ids_list, test_ids_list
    
    def get_splits_loocv(self):
        """Get cross-validation splits"""
        id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_cv()
        flattened_list = []
        for tr, tst in zip(train_ids_list, test_ids_list):
            flattened_list.extend(tr)
            flattened_list.extend(tst)
        unique_elements = list(set(flattened_list))
        
        #LOOCV
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
        adata_test = self.adata[self.adata.obs[id_column].isin(test_ids)]
        adata_train = self.adata[self.adata.obs[id_column].isin(train_ids)]
        return adata_train, adata_test

    def get_split_data(self):
        """Get train-test split data"""
        if hasattr(self.data_loader, 'train_test_split_dict'):
            split_dict = self.data_loader.train_test_split_dict
            test_ids = split_dict['train_test_split']['test_ids']
            train_ids = split_dict['train_test_split']['train_ids']
            id_column = split_dict['id_column']
            logger.info(test_ids)
        return id_column, train_ids, test_ids

    def prepare_data(self, adata_train, adata_test, id_column):
        """Prepare data for training"""
        adata_train = adata_train.copy()
        adata_test = adata_test.copy()
        adata_train.obs['sample_id'] = adata_train.obs[id_column]
        adata_test.obs['sample_id'] = adata_test.obs[id_column]
        return adata_train, adata_test

    def train(self, loader):
        self.data_loader = loader
        self.adata = loader.adata
        
        if self.cls_level == 'patient':
            self.train_sample(loader)
        elif self.cls_level =='cell':
            # Single split training
            id_column, train_ids, test_ids = self.get_split_data()
            adata_train, adata_test = self.split_data(id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(adata_train, adata_test, id_column)
            self.__train_cell(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
        
    def train_sample(self, loader):
        """Main training pipeline"""
        self._cv_fold = None
        self.label_key = loader.label_key
        self.encode_labels()
        train_func_map = {'vote': self.__train_vote, 'avg': self.__train_avg_expression, 'mil': self.__train_mil}
        train_fcs = [] 
        prefix =[]
        for fnc in self.train_funcs:
            train_fcs.append(train_func_map[fnc])
            prefix.append(fnc)
            
        # Single split training
        if self.onesplit:
            id_column, train_ids, test_ids = self.get_split_data()
            adata_train, adata_test = self.split_data(id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(adata_train, adata_test, id_column)
            for fnc in train_fcs:
                fnc(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_mil(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_vote(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
                # self.__train_avg_expression(adata_train, adata_test, evaluate=self.evaluate, viz=self.viz)
        
        # Cross-validation training
        if self.cv:
            if self.cv =='loocv':
                id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_loocv()
            else:
                id_column, n_splits, train_ids_list, test_ids_list = self.get_splits_cv()
                
            for tr_f, pre in zip(train_fcs, prefix):
                self.__train_cv(tr_f, id_column, n_splits, train_ids_list, test_ids_list, pre)

    def __train_cv(self, train_fnc, id_column, n_splits, train_ids_list, test_ids_list, prefix=''):
        """Run cross-validation training"""
        pred_list = []
        metrics_list = []
        pred_list_train = []
        metrics_list_train = []
        logger.info(f'Running crossvalidation with {n_splits} folds')
        
        for i in range(n_splits):
            logger.info(f'---------- fold {i+1}----------')
            train_ids, test_ids = train_ids_list[i], test_ids_list[i]
            adata_train, adata_test = self.split_data(id_column, train_ids, test_ids)
            adata_train, adata_test = self.prepare_data(adata_train, adata_test, id_column)
            self._cv_fold = i + 1
            # pred_df, metric_df = train_fnc(adata_train, adata_test, f'fold_{i+1}_')
            # pred_df_train, pred_df_test, metrics_df_train, metrics_df_test = train_fnc(adata_train, adata_test, f'fold_{i+1}_True
            pred_df_train, pred_df_test, metrics_df_train, metrics_df_test = train_fnc(adata_train, adata_test, evaluate=True, viz=False)

            pred_df_test['fold'] = f'fold_{i+1}'
            metrics_df_test['fold'] = f'fold_{i+1}'
            
            pred_df_train['fold'] = f'fold_{i+1}'
            metrics_df_train['fold'] = f'fold_{i+1}'
            
            pred_list.append(pred_df_test)
            metrics_list.append(metrics_df_test)
            
            pred_list_train.append(pred_df_train)
            metrics_list_train.append(metrics_df_train)
            self._cv_fold = None

        preds = pd.concat(pred_list)
        metrics = pd.concat(metrics_list)
        
        preds_train = pd.concat(pred_list_train)
        metrics_train = pd.concat(metrics_list_train)
        
        save_dir = join(self.saving_dir, 'cv')
        os.makedirs(save_dir, exist_ok=True)
        
        preds.to_csv(join(save_dir, f'{prefix}_cv_predictions.csv'))
        metrics.to_csv(join(save_dir, f'{prefix}_cv_metrics.csv'))
        
        preds_train.to_csv(join(save_dir, f'{prefix}_cv_predictions_train.csv'))
        metrics_train.to_csv(join(save_dir, f'{prefix}_cv_metrics_train.csv'))
        
        # Plot metrics (suppress seaborn/matplotlib bxp `vert` PendingDeprecationWarning)
        metrics.fillna(0, inplace=True)
        plt.figure(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PendingDeprecationWarning)
            sns.boxplot(
                data=metrics,
                x='Metrics',
                y=self.model_name,
                orient='v',
            )
        plt.title('Cross-Validation Metric Distribution')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(save_dir, f'{prefix}_cv_metrics_boxplot.png'))
        plt.close()
        
        metrics_train.fillna(0, inplace=True)
        plt.figure(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PendingDeprecationWarning)
            sns.boxplot(
                data=metrics_train,
                x='Metrics',
                y=self.model_name,
                orient='v',
            )
        plt.title('Cross-Validation Metric Distribution on Training set')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(save_dir, f'{prefix}_cv_metrics_boxplot_train.png'))
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
                raise KeyError(f"Embedding '{self.embedding_col}' not found in adata.obsm")
    
            emb = adata.obsm[self.embedding_col]

            # Convert sparse to dense if necessary
            if not isinstance(emb, np.ndarray):
                emb = emb.toarray()
                
            # Validate sample-label consistency
            label_counts = adata.obs.groupby('sample_id', observed=False)['label'].nunique()
            if (label_counts > 1).any():
                raise ValueError(f"Samples with multiple labels found: {label_counts[label_counts > 1].index.tolist()}")


            sample_ids = adata.obs['sample_id'].values
            df_emb = pd.DataFrame(emb, index=sample_ids)

            mean_emb = df_emb.groupby(df_emb.index).mean()

            labels = adata.obs[['sample_id', 'label']].drop_duplicates().set_index('sample_id')
            labels = labels.loc[labels.index.intersection(mean_emb.index)]

            return mean_emb.loc[labels.index], labels['label']


        X_train, y_train = aggregate_embeddings(adata_train)
        X_test, y_test = aggregate_embeddings(adata_test)
        

        # Train classifier
        self.model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test = train_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            model_name=self.model_name,
            random_state=self.random_state,
        )

        pred_df_test = pd.DataFrame({ 'label': y_test,'pred': y_pred_test,
                                'pred_score': y_pred_score_test[:, 1]  # Assuming binary classification
                               }, index=X_test.index)

        pred_df_train = pd.DataFrame({ 'label': y_train,'pred': y_pred_train,
                                'pred_score': y_pred_score_train[:, 1]  # Assuming binary classification
                               }, index=X_train.index)
            
        metrics_df_test, cls_report_test = eval_classifier(pred_df_test['label'], pred_df_test['pred'], pred_df_test['pred_score'],
                                                 estimator_name=self.model_name, label_names=self.label_names)
        
        metrics_df_train, cls_report_train = eval_classifier(pred_df_train['label'], pred_df_train['pred'], pred_df_train['pred_score'],
                                                 estimator_name=self.model_name, label_names=self.label_names)

        # metrics_df_train= None
        # metrics_df_test = None
        if evaluate:
            postfix = "avg_expr"
            save_results(pred_df_test, metrics_df_test, cls_report_test, self.saving_dir, postfix, viz, self.model_name, self.label_names)
            postfix = "avg_expr_train"
            save_results(pred_df_train, metrics_df_train, cls_report_train, self.saving_dir, postfix, viz, self.model_name, self.label_names)

        avg_png = self._training_diagnostics_path("avg", "learning_logloss_curve.png")
        self._plot_sklearn_learning_curve_log_loss(
            X_train,
            y_train,
            avg_png,
            title="Mean embedding + classifier (log loss vs. train size)",
            max_train_samples=None,
        )

        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test

    def __train_mil(self, adata_train, adata_test, evaluate=False, viz=False):
        """Multi-instance learning training"""
        logger.info('Training model (Multi instance Learning (MIL))')

        patient_key = "sample_id"
        label_key = "label"

        adata_mil_train = adata_train
        adata_mil_val = None
        if self.mil_holdout_fraction > 0:
            adata_mil_train, adata_mil_val = self._split_mil_train_val(
                adata_train,
                patient_key,
                label_key,
                self.mil_holdout_fraction,
                self.random_state,
            )
            if adata_mil_val is None:
                adata_mil_train = adata_train
            elif adata_mil_val.obs[patient_key].nunique() < 1:
                adata_mil_train, adata_mil_val = adata_train, None

        mk = dict(self.mil_model_kwargs)
        pw = mk.get("pos_weight", None)
        if isinstance(pw, str) and pw.strip().lower() == "auto":
            mk["pos_weight"] = self._mil_patient_pos_weight(adata_mil_train, patient_key, label_key)
        for key in list(mk.keys()):
            if key not in _MIL_YAML_MODEL_KEYS:
                logger.warning("Ignoring unknown MIL config key (not passed to MILExperiment): %s", key)
                mk.pop(key)

        exp = MILExperiment(
            embedding_col=self.embedding_col,
            label_key=label_key,
            patient_key=patient_key,
            seed=self.random_state,
            **mk,
        )
        exp.train(adata_mil_train, adata_val=adata_mil_val)

        mil_png = self._training_diagnostics_path("mil", "train_val_loss.png")
        exp.plot_training_loss_curves(mil_png)

        #test
        pids, y_true, preds, pred_scores, metrics_test = exp.evaluate(adata_test)
        # pids, y_true, preds, pred_scores, metrics
        pred_df_test = pd.DataFrame({'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})
        #train
        pids, y_true, preds, pred_scores, metrics_train = exp.evaluate(adata_train)
        pred_df_train = pd.DataFrame({'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})
        metrics_df_test, cls_report_test = eval_classifier(pred_df_test['label'], pred_df_test['pred'], pred_df_test['pred_score'], estimator_name=self.model_name, label_names=self.label_names)
        
        metrics_df_train, cls_report_train = eval_classifier(pred_df_train['label'], pred_df_train['pred'], pred_df_train['pred_score'], estimator_name=self.model_name, label_names=self.label_names)
        
       

            
        if evaluate:

            
            save_results(pred_df_test, metrics_df_test, cls_report_test, self.saving_dir, postfix='mil', viz=viz, model_name=self.model_name, label_names=self.label_names)
            
            
            save_results(pred_df_train, metrics_df_train, cls_report_train, self.saving_dir, postfix='mil_train', viz=viz, model_name=self.model_name, label_names=self.label_names)
            
        if self.mil_debug:
            dbg_root = join(self.saving_dir, "mil_debug")
            if self._cv_fold is not None:
                dbg_root = join(dbg_root, f"fold_{self._cv_fold}")
            os.makedirs(dbg_root, exist_ok=True)
            logger.info("MIL debug: writing attention diagnostics to %s", dbg_root)
            exp.export_attention_debug(adata_test, dbg_root, top_k=self.mil_top_k)

        if viz:
            try:
                from models.attention_viz import (
                    print_attention_summary,
                    visualize_attention_comprehensive,
                    export_top_cells_table,
                )
            except ImportError:
                logger.warning(
                    "viz=True but models.attention_viz is not available; "
                    "skip legacy attention viz (use classification.params.mil.debug for built-in exports)."
                )
            else:
                results = visualize_attention_comprehensive(
                    exp,
                    adata_test,
                    top_k=20,
                    save_path=join(self.saving_dir, "attention_analysis.png"),
                )
                print_attention_summary(results)
                export_top_cells_table(
                    results["attention_df"],
                    top_k=20,
                    save_path=join(self.saving_dir, "top_attention_cells.csv"),
                )

        # return pred_df, metrics_df if evaluate else None
        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test

    def __train_cell(self, adata_train, adata_test, evaluate=False, viz=False, plot_vote_learning_curve=False):
        """Cell-level predictions training"""
        logger.info('Training model')
        X_train, y_train = adata_train.obsm[self.embedding_col], adata_train.obs['label']
        X_test, y_test = adata_test.obsm[self.embedding_col], adata_test.obs['label']

        # self.model, y_test, y_pred, y_pred_score = train_classifier(X_train, y_train, X_test, y_test, 
        #                                                           model_name=self.model_name)
        self.model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test = train_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            model_name=self.model_name,
            random_state=self.random_state,
        )

        if plot_vote_learning_curve:
            vote_png = self._training_diagnostics_path("vote", "learning_logloss_curve.png")
            self._plot_sklearn_learning_curve_log_loss(
                X_train,
                y_train,
                vote_png,
                title="Vote: cell-level classifier (log loss vs. train size)",
                max_train_samples=8000,
            )

        adata_test.obs['pred'] = y_pred_test
        adata_test.obs['pred_score'] = y_pred_score_test[:, 1] #assume binary classification
        
        adata_train.obs['pred'] = y_pred_train
        adata_train.obs['pred_score'] = y_pred_score_train[:, 1] #assume binary classification
        
        if evaluate:
            save_dir = join(self.saving_dir, 'cell_level_pred')
            os.makedirs(save_dir, exist_ok=True)
            
            adata_test.obs.to_csv(join(save_dir, f'cell_pred_test.csv'))
            adata_train.obs.to_csv(join(save_dir, f'cell_pred_train.csv'))
            
            metrics_df_test, cls_report = eval_classifier(y_test, y_pred_test, y_pred_score_test, estimator_name=self.model_name, label_names=self.label_names)
            pred_df = adata_test.obs['pred'].copy()
            save_results(pred_df, metrics_df_test,cls_report,save_dir, 'cell', viz, self.model_name, self.label_names)
            
            metrics_df_train, cls_report = eval_classifier(y_test, y_pred_test, y_pred_score_test, estimator_name=self.model_name, label_names=self.label_names)
            pred_df = adata_train.obs['pred'].copy()
            save_results(pred_df, metrics_df_train,cls_report,save_dir, 'cell_train', viz, self.model_name, self.label_names)
            
            # return adata_test.obs, metrics_df
            return adata_train.obs, adata_test.obs, metrics_df_train, metrics_df_test
        
        return adata_test.obs, None

    def __train_vote(self, adata_train, adata_test, evaluate=False, viz=False):
        """Majority vote predictions training"""
        logger.info('Training model (Majority Vote)')
        
        cell_pred_train, cell_pred_test, metrics_df_train, metrics_df_test = self.__train_cell(
            adata_train,
            adata_test,
            evaluate=evaluate,
            viz=viz,
            plot_vote_learning_curve=True,
        )
        
        pred_df_test, metrics_df_test = self.save_patient_level(cell_pred_test, 
                                                                            evaluate, viz, 
                                                                            postfix='', 
                                                                            model='vote')
        
        pred_df_train, metrics_df_train = self.save_patient_level(cell_pred_train, 
                                                                            evaluate, viz, 
                                                                            postfix='_train', 
                                                                            model='vote')
        
        # return sample_pred_test_df, sample_metrics_test_df
        return pred_df_train, pred_df_test, metrics_df_train, metrics_df_test

    def save_patient_level(self, adata_subset, evaluate=False, viz=False, postfix="", model=""):
        """Save patient-level predictions and metrics"""
        logger.info('Saving sample level performance')
        
        obs = adata_subset
        
        def majority_vote_score(x):
            majority_class = x.value_counts().idxmax()
            return (x == majority_class).sum() / len(x)

        y_pred_score_p = obs.groupby('sample_id', observed=False)['pred_score'].mean()
        
        # y_pred_score_p = obs.groupby('sample_id')['pred'].agg(majority_vote_score)
        
        y_pred_p = obs.groupby('sample_id', observed=False)['pred'].agg(
            lambda x: x.value_counts().idxmax()
        )
        y_test_p = (
            obs.groupby('sample_id', observed=False)['label']
            .first()
            .reindex(y_pred_score_p.index)
        )
        
        pred_df = pd.concat([y_test_p, y_pred_p, y_pred_score_p], axis=1)
        pred_df.columns = ['label', 'pred', 'pred_score']
        
        metrics_df, cls_report = eval_classifier(pred_df['label'], pred_df['pred'], pred_df['pred_score'],
                                              estimator_name=self.model_name, label_names=self.label_names)
        
        if evaluate:
            save_results(pred_df, metrics_df,cls_report, self.saving_dir, f'vote{postfix}', viz, self.model_name, self.label_names)
        
        return pred_df, metrics_df
