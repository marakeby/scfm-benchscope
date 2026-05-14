from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import anndata as ad
from os.path import join
import pandas as pd

import sys

sys.path.append('/home/jupyter/scFM_eval')

from scfm_cancer_eval.models.mil_experiment import MILExperiment



MODEL_REGISTRY = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    'svc': SVC
    # 'gf_finetune': GFFineTuneModel
}

def train_classifier(X_train, y_train, X_test, y_test, model_name='random_forest'):
    """Train a classifier and return predictions"""
    model_cls = MODEL_REGISTRY.get(model_name, RandomForestClassifier)
    model = model_cls(probability=True) if model_name == 'svc' else model_cls(max_depth=5, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_score_test = model.predict_proba(X_test)
    
    y_pred_train = model.predict(X_train)
    y_pred_score_train = model.predict_proba(X_train)
    
    print(y_pred_score_test.shape, y_pred_train.shape)
    
    return model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test

def subsample_majority_class(X, y, random_state=42):
    """
    Subsample the majority class to match the number of minority samples.
    
    Parameters:
    - X: np.ndarray or similar feature matrix
    - y: np.ndarray of binary labels (0 = majority, 1 = minority)
    
    Returns:
    - X_sub: Subsampled feature matrix
    - y_sub: Corresponding subsampled labels
    """
    import numpy as np
    from sklearn.utils import resample

    y = np.asarray(y)
    majority_mask = y == 0
    minority_mask = y == 1

    n_minority = minority_mask.sum()
    n_majority = majority_mask.sum()

    if n_minority == 0 or n_majority == 0:
        raise ValueError("One of the classes has zero samples — cannot perform subsampling.")

    idx_majority = np.where(majority_mask)[0]
    idx_minority = np.where(minority_mask)[0]

    idx_majority_subsampled = resample(
        idx_majority,
        replace=False,
        n_samples=n_minority,
        random_state=random_state
    )

    selected_indices = np.concatenate([idx_majority_subsampled, idx_minority])
    np.random.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]


def __train_cell( adata_train, adata_test, embedding_col, model_name, subsample=False ):
        """Cell-level predictions training"""
        print('Training model')
        # X_train, y_train = adata_train.obsm[embedding_col], adata_train.obs['label'].values
        X_test, y_test = adata_test.obsm[embedding_col], adata_test.obs['label'].values
        


        X = adata_train.obsm[embedding_col]
        y = adata_train.obs['label'].values

        # Subsample
        if subsample:
            X_train, y_train = subsample_majority_class(X, y)
        else:
            X_train, y_train = adata_train.obsm[embedding_col], adata_train.obs['label'].values
    
        model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test = train_classifier(X_train, y_train, X_test, y_test, 
                                                                  model_name=model_name)
        

        # adata_test.obs['pred'] = y_pred_test
        # adata_test.obs['pred_score'] =  y_pred_score_test[:, 1]
     
        adata_test.obs['pred'] = model.predict(X_test)
        adata_test.obs['pred_score'] =  model.predict_proba(X_test)[:, 1]
        # adata_train.obs['pred'] = y_pred_train
        # adata_train.obs['pred_score'] =  y_pred_score_train[:, 1]
        
        adata_train.obs['pred'] = model.predict(X)
        adata_train.obs['pred_score'] = model.predict_proba(X)[:, 1]
        
        pred_df_train = adata_train.obs.copy()
        pred_df_test = adata_test.obs.copy()

        
        
        return pred_df_train, pred_df_test

def __train_vote(adata_train, adata_test, embedding_col, model_name,subsample=False):
        """Majority vote predictions training"""
        print('Training model (Majority Vote)')
        
        X_train, y_train = adata_train.obsm[embedding_col], adata_train.obs['label'].values
        X_test, y_test = adata_test.obsm[embedding_col], adata_test.obs['label'].values

        print("Unique values in train label:", np.unique(y_train))
        print("Unique values in test label:", np.unique(y_test))
        
        cell_pred_train, cell_pred_test = __train_cell(adata_train, adata_test, embedding_col, model_name, subsample)
        
        print("Cell-level prediction distribution (train):")
        print(cell_pred_train['pred'].value_counts())

        print("Cell-level prediction distribution (test):")
        print(cell_pred_test['pred'].value_counts())
        
        pred_df_test = get_patient_level(cell_pred_test)
        
        pred_df_train  = get_patient_level(cell_pred_train)
        
        return pred_df_test, pred_df_train


def __train_mil( adata_train, adata_test, embedding_col, model_name,subsample=False):
        
    """Multi-instance learning training"""

    print('Training model (Multi instance Learning (MIL))')

    exp = MILExperiment(embedding_col= embedding_col, label_key='label', 
                      patient_key="sample_id", celltype_key="cell_type")
    exp.train(adata_train)

    #test
    pids, y_true, preds, pred_scores,m = exp.evaluate(adata_test)
    pred_df_test = pd.DataFrame({'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})

    #train
    pids, y_true, preds, pred_scores,m = exp.evaluate(adata_train)
    pred_df_train = pd.DataFrame({'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})

    return pred_df_test, pred_df_train


def __train_avg_expression( adata_train, adata_test, embedding_col, model_name,subsample=False):
        """Train and evaluate using average expression per sample"""
        print('Training model (Average Embedding Per Sample)')

        def aggregate_embeddings(adata, embedding_col):
            emb = adata.obsm[embedding_col]

            # Convert sparse to dense if necessary
            if not isinstance(emb, np.ndarray):
                emb = emb.toarray()

            sample_ids = adata.obs['sample_id'].values
            df_emb = pd.DataFrame(emb, index=sample_ids)

            mean_emb = df_emb.groupby(df_emb.index).mean()

            labels = adata.obs[['sample_id', 'label']].drop_duplicates().set_index('sample_id')
            labels = labels.loc[labels.index.intersection(mean_emb.index)]

            return mean_emb.loc[labels.index], labels['label']


        X_train, y_train = aggregate_embeddings(adata_train, embedding_col)
        X_test, y_test = aggregate_embeddings(adata_test, embedding_col)
        

        print("Unique values in train label:", np.unique(y_train))
        print("Unique values in test label:", np.unique(y_test))
        
        # Train classifier
        model, y_train, y_test, y_pred_train, y_pred_test, y_pred_score_train, y_pred_score_test = train_classifier(X_train, y_train, X_test, y_test, model_name=model_name)

        pred_df_test = pd.DataFrame({ 'label': y_test,'pred': y_pred_test,
                                'pred_score': y_pred_score_test[:, 1]  # Assuming binary classification
                               }, index=X_test.index)

        pred_df_train = pd.DataFrame({ 'label': y_train,'pred': y_pred_train,
                                'pred_score': y_pred_score_train[:, 1]  # Assuming binary classification
                               }, index=X_train.index)
        


        
        return pred_df_test, pred_df_train

def get_patient_level( adata_subset):
    """Save patient-level predictions and metrics"""
    # logger.info('Saving sample level performance')

    obs = adata_subset

    def majority_vote_score(x):
        majority_class = x.value_counts().idxmax()
        return (x == majority_class).sum() / len(x)

    y_pred_score_p = obs.groupby('sample_id', observed=False)['pred_score'].mean()
    # y_pred_score_p = obs.groupby('sample_id')['pred'].agg(majority_vote_score)


    y_pred_p = obs.groupby('sample_id',  observed=False)['pred'].agg(lambda x: x.value_counts().idxmax())
    y_test_p = obs.groupby('sample_id',  observed=False)['label'].first().reindex(y_pred_score_p.index)

    pred_df = pd.concat([y_test_p, y_pred_p, y_pred_score_p], axis=1)
    pred_df.columns = ['label', 'pred', 'pred_score']
    return pred_df


def get_splits_cv(cv_split_dict):
        """Get cross-validation splits"""
        cv = cv_split_dict
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

def split_data( adata,  id_column, train_ids, test_ids):
    """Split data into train and test sets"""
    adata_test = adata[adata.obs[id_column].isin(test_ids)]
    adata_train = adata[adata.obs[id_column].isin(train_ids)]
    return adata_train, adata_test


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import numpy as np

def get_classification_metrics(pred_df, threshold=0.5, average='binary'):
    """
    Compute classification metrics from a DataFrame with true labels, predicted labels, and predicted scores.
    
    Args:
        pred_df (pd.DataFrame): DataFrame with columns ['label', 'pred', 'pred_score']
        threshold (float): Threshold to convert pred_score to class if 'pred' not already provided.
        average (str): Averaging method for multiclass/multilabel: 'binary', 'macro', 'micro', 'weighted'
        
    Returns:
        dict: Dictionary of computed metrics
    """
    y_true = pred_df['label'].values
    y_pred = pred_df['pred'].values
    y_score = pred_df['pred_score'].values

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # AUC and AUPRC (only valid for binary classification)
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = 'N/A'

    try:
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        auprc = 'N/A'

    # Confusion matrix and support
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    # Extract support for each class
    support = {str(label): int(info["support"]) for label, info in report.items() if label.isdigit()}

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'auprc': auprc,
        'confusion_matrix': conf_matrix.tolist(),
        'support': support,
        'classification_report': report
    }

    return metrics
