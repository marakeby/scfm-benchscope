import os
import json
import logging
from collections import Counter
from typing import Tuple, Any
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from imblearn.over_sampling import RandomOverSampler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def read_data(data_file: str) -> ad.AnnData:
    """Read AnnData from h5ad file."""
    return ad.read_h5ad(data_file)

def get_patient_info(adata: ad.AnnData, patient_key: str, label_key: str) -> Tuple[np.ndarray, pd.Series, np.ndarray]:
    """Get patient IDs, labels, and sample counts."""
    patient_labels = adata.obs.groupby(patient_key)[label_key].first()
    sample_counts = adata.obs[patient_key].value_counts()
    patient_info = patient_labels.to_frame().join(sample_counts)
    ids = patient_info.index.values
    labels = patient_info[label_key]
    counts = patient_info['count'].values
    return ids, labels, counts

def oversample_data(train_ids: np.ndarray, y_train: Any, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly oversample to balance classes."""
    train_ids = np.array(train_ids).reshape(-1, 1)
    ros = RandomOverSampler(random_state=random_state)
    train_ids_oversampled, y_train_oversampled = ros.fit_resample(train_ids, y_train)
    train_ids_oversampled = train_ids_oversampled.ravel()
    return train_ids_oversampled, y_train_oversampled

def has_class_imbalance(y: Any, threshold: float) -> bool:
    counts = list(Counter(y).values())
    return min(counts) / max(counts) < threshold

def safe_stratified_split(patient_ids, counts, labels, test_size=0.3, random_state=42, max_tries=100):
    for attempt in range(max_tries):
        train_ids, test_ids, train_counts, test_counts, y_train, y_test = train_test_split(
            patient_ids,
            counts,
            labels,
            test_size=test_size,
            random_state=random_state + attempt,
            stratify=labels
        )
        if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
            return train_ids, test_ids, train_counts, test_counts, y_train, y_test
    raise ValueError("Could not generate a valid stratified split with at least 2 classes per split after multiple attempts.")

def save_cv_splits(patient_ids, labels, save_dir, n_splits=5, random_state=42, imbalance_threshold=0.8, patient_key='donor_id', label_key='label'):
    os.makedirs(save_dir, exist_ok=True)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = {}
    cv_splits_oversampled = {}
    for i, (train_idx, test_idx) in enumerate(splitter.split(patient_ids, labels)):
        logging.info(f"=== Fold {i + 1}/{n_splits} ===")
        train_ids = patient_ids[train_idx]
        test_ids = patient_ids[test_idx]
        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]
        train_ids_oversampled, y_train_oversampled = oversample_data(train_ids, y_train, random_state)
        cv_splits[f'fold_{i+1}'] = {
            'train_ids': train_ids.tolist(),
            'test_ids': test_ids.tolist(),
            'train_labels': y_train.tolist(),
            'test_labels': y_test.tolist()
        }
        cv_splits_oversampled[f'fold_{i+1}'] = {
            'train_ids': train_ids_oversampled.tolist(),
            'test_ids': test_ids.tolist(),
            'train_labels': y_train_oversampled.tolist(),
            'test_labels': y_test.tolist()
        }
        fold_data = {
            'train_ids': train_ids.tolist(),
            'test_ids': test_ids.tolist(),
            'train_labels': y_train.tolist(),
            'test_labels': y_test.tolist()
        }
        with open(os.path.join(save_dir, f'fold_{i+1}.json'), 'w') as f:
            json.dump(fold_data, f)
    cv_splits['id_column'] = patient_key
    cv_splits['label_column'] = label_key
    cv_splits['random_state'] = random_state
    cv_splits['n_splits'] = n_splits
    cv_splits_oversampled['id_column'] = patient_key
    cv_splits_oversampled['label_column'] = label_key
    cv_splits_oversampled['random_state'] = random_state
    cv_splits_oversampled['n_splits'] = n_splits
    with open(os.path.join(save_dir, 'cv_splits.json'), 'w') as f:
        json.dump(cv_splits, f)
    with open(os.path.join(save_dir, 'cv_splits_oversampled.json'), 'w') as f:
        json.dump(cv_splits_oversampled, f)
    logging.info(f"Cross-validation splits saved to {save_dir}")
    return save_dir

def save_train_test_split(patient_ids, labels, counts, save_dir, test_size=0.33, random_state=42, max_tries=10, imbalance_threshold=0.8, patient_key='donor_id', label_key='label'):
    os.makedirs(save_dir, exist_ok=True)
    train_ids, test_ids, train_counts, test_counts, y_train, y_test = safe_stratified_split(
        patient_ids, counts, labels, test_size=test_size, random_state=random_state, max_tries=max_tries
    )
    def save_df(data, filename):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_dir, filename), index=False)
    save_df({'patient_id': train_ids.ravel(), 'label': y_train, 'cell_count': train_counts}, 'train.csv')
    save_df({'patient_id': test_ids.ravel(), 'label': y_test, 'cell_count': test_counts}, 'test.csv')
    def build_metadata(train_ids, y_train, oversampled):
        return {
            'train_test_split': {
                'train_ids': train_ids.tolist(),
                'test_ids': test_ids.tolist(),
                'train_labels': y_train.tolist(),
                'test_labels': y_test.tolist()
            },
            'id_column': patient_key,
            'label_column': label_key,
            'random_state': random_state,
            'test_size': test_size,
            'oversampled': oversampled
        }
    with open(os.path.join(save_dir, 'train_test_split.json'), 'w') as f:
        json.dump(build_metadata(train_ids, y_train, oversampled=False), f)
    perform_oversampling = has_class_imbalance(y_train, imbalance_threshold)
    if perform_oversampling:
        train_ids_oversampled, y_train_oversampled = oversample_data(train_ids, y_train, random_state)
        save_df({'patient_id': train_ids_oversampled.ravel(), 'label': y_train_oversampled}, 'train_oversampled.csv')
        with open(os.path.join(save_dir, 'train_test_split_oversampled.json'), 'w') as f:
            json.dump(build_metadata(train_ids_oversampled, y_train_oversampled, oversampled=True), f)
    logging.info(f"Train-test split saved to {save_dir}")
    return save_dir

def main():
    # Example usage for chemo split (edit as needed)
    DATA_FILE = '/path/to/data.h5ad'
    SAVE_DIR = '/path/to/save_dir'
    PATIENT_KEY = 'donor_id'
    LABEL_KEY = 'cohort'
    TEST_SIZE = 0.33
    RANDOM_STATE = 42
    N_SPLITS = 5
    # Read data
    adata = read_data(DATA_FILE)
    # Filter as needed, e.g.:
    # adata = adata[adata.obs['timepoint'] == 'Pre']
    # adata = adata[adata.obs[LABEL_KEY].isin(['class1', 'class2'])]
    patient_ids, labels, counts = get_patient_info(adata, PATIENT_KEY, LABEL_KEY)
    save_train_test_split(patient_ids, labels, counts, SAVE_DIR, test_size=TEST_SIZE, random_state=RANDOM_STATE, patient_key=PATIENT_KEY, label_key=LABEL_KEY)
    save_cv_splits(patient_ids, labels, SAVE_DIR, n_splits=N_SPLITS, random_state=RANDOM_STATE, patient_key=PATIENT_KEY, label_key=LABEL_KEY)

if __name__ == "__main__":
    main() 