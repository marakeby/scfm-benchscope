import anndata as ad
import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
import random




def read_data(DATA_FILE):
    fname = DATA_FILE
    adata = ad.read_h5ad(fname)
    return adata

# def get_patient_info(adata, patient_key, label_key):
#     patient_labels = adata.obs.groupby(patient_key)[label_key].first()
#     counts = adata.obs[patient_key].value_counts()

#     return patient_labels.index.values, patient_labels.values, counts

def get_patient_info(adata, patient_key, label_key):
    patient_labels = adata.obs.groupby(patient_key)[label_key].first()
    sample_counts = adata.obs[patient_key].value_counts()
    
    # labels = adata.obs.groupby(patient_key)[label_key].first()
    # counts = adata.obs.sample_id.value_counts()
    patient_info = patient_labels.to_frame().join(sample_counts)
    patient_ids = patient_info.index.values
    ids = patient_info.index.values
    labels = patient_info[label_key]
    counts= patient_info['count'].values
    return ids, labels, counts


def oversample_data(train_ids, y_train, random_state):
    # Convert train_ids to array if it's not
    train_ids = np.array(train_ids).reshape(-1, 1)  # Reshape for oversampling

    # Apply random oversampling to balance the classes
    ros = RandomOverSampler(random_state=random_state)
    train_ids_oversampled, y_train_oversampled = ros.fit_resample(train_ids, y_train)

    # Flatten the train_ids back to 1D if needed
    train_ids_oversampled = train_ids_oversampled.ravel()
    return train_ids_oversampled, y_train_oversampled

def save_cv_splits(patient_ids,labels, save_dir, n_splits=5, random_state=42):

    
    # Create directory for CV splits
    # cv_save_dir = os.path.join(save_dir, 'cv_splits')
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize splitter
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Dictionary to store all splits
    cv_splits = {}    
    cv_splits_oversampled = {}

    # Generate and save splits
    for i, (train_idx, test_idx) in enumerate(splitter.split(patient_ids, labels)):
        print(f"=== Fold {i + 1}/{n_splits} ===")
        train_ids = patient_ids[train_idx]
        test_ids = patient_ids[test_idx]
        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]

        train_ids_oversampled, y_train_oversampled = oversample_data(train_ids, y_train, random_state)

        # y_train_oversampled_df = pd.DataFrame({
        #     'patient_id': train_ids_oversampled,
        #     'label': y_train_oversampled
        # })

        # Store the splits for this fold
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
        
        # Save individual fold data
        fold_data = {
            'train_ids': train_ids.tolist(),
            'test_ids': test_ids.tolist(),
            'train_labels': labels.iloc[train_idx].tolist(),
            'test_labels': labels.iloc[test_idx].tolist()
        }
        
        # Save this fold's data
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

    # Save all splits in one file
    with open(os.path.join(save_dir, 'cv_splits.json'), 'w') as f:
        json.dump(cv_splits, f)

    with open(os.path.join(save_dir, 'cv_splits_oversampled.json'), 'w') as f:
        json.dump(cv_splits_oversampled, f)
    

    print(f"Cross-validation splits saved to {save_dir}")
    return save_dir


def safe_stratified_split(patient_ids, counts, labels, test_size=0.3, random_state=42, max_tries=100):
    for attempt in range(max_tries):
        train_ids, test_ids, train_counts, test_counts, y_train, y_test = train_test_split(
            patient_ids,
            counts,
            labels,
            test_size=test_size,
            random_state=random_state + attempt,  # to vary the seed
            stratify=labels
        )
        if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
            return train_ids, test_ids, train_counts, test_counts, y_train, y_test
    raise ValueError("Could not generate a valid stratified split with at least 2 classes per split after multiple attempts.")


def save_train_test_split(patient_ids,labels, counts, save_dir, test_size=0.33, random_state=42, max_tries=10):
    # save_dir = os.path.join(save_dir, 'train_test_split')
    os.makedirs(save_dir, exist_ok=True)
    
    
    train_ids, test_ids, train_counts, test_counts, y_train, y_test = safe_stratified_split(patient_ids, counts, labels, test_size=test_size, random_state=42, max_tries=max_tries)
    
    train_ids_oversampled, y_train_oversampled = oversample_data(train_ids, y_train, random_state)

    
    df_train = pd.DataFrame({
        'patient_id': train_ids.ravel(),
        'label': y_train, 
        'cell_count': train_counts
    })

    df_train_oversampled = pd.DataFrame({
        'patient_id': train_ids_oversampled.ravel(),
        'label': y_train_oversampled
    })

    # df_train_oversampled.join()
    df_test = pd.DataFrame({
        'patient_id': test_ids.ravel(),
        'label': y_test, 
        'cell_count': test_counts
    })

    df_train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)

    df_train_oversampled.to_csv(os.path.join(save_dir, 'train_oversampled.csv'), index=False)

    train_test_dict = {}
    train_test_split_oversampled = {}

    train_test_dict[f'train_test_split'] = {
            'train_ids': train_ids.tolist(),
            'test_ids': test_ids.tolist(),
            'train_labels': y_train.tolist(),
            'test_labels': y_test.tolist()
        }
    
    train_test_split_oversampled[f'train_test_split'] = {
            'train_ids': train_ids_oversampled.tolist(),
            'test_ids': test_ids.tolist(),
            'train_labels': y_train_oversampled.tolist(),
            'test_labels': y_test.tolist()
        }   

    train_test_dict['id_column'] = patient_key
    train_test_dict['label_column'] = label_key
    train_test_dict['random_state'] = random_state
    train_test_dict['test_size'] = test_size
    train_test_dict['oversampled'] = False

    train_test_split_oversampled['id_column'] = patient_key
    train_test_split_oversampled['label_column'] = label_key
    train_test_split_oversampled['random_state'] = random_state
    train_test_split_oversampled['test_size'] = test_size
    train_test_split_oversampled['oversampled'] = True


    with open(os.path.join(save_dir, 'train_test_split.json'), 'w') as f:
        json.dump(train_test_dict, f)

    with open(os.path.join(save_dir, 'train_test_split_oversampled.json'), 'w') as f:
        json.dump(train_test_split_oversampled, f)

    print(f"Train-test split saved to {save_dir}")
    return save_dir

if __name__ == "__main__":
    # Example usage
    # save_dir = '/home/jupyter/scFM_eval/data_splits/luad/luad11'
    # DATA_FILE = '/home/jupyter/mnt/DATA/luad_brca/luad/luad_analysis1.h5ad'

    save_dir = '/home/jupyter/scFM_eval/data_splits/luad/luad22'
    DATA_FILE = '/home/jupyter/mnt/DATA/luad_brca/luad/luad_analysis2.h5ad'
    
    # save_dir = '/home/jupyter/sceval/data_splits/brca/brca4'
    # DATA_FILE = '/home/jupyter/6_BRCA/input_data/brca_analysis4.h5ad'
    
    patient_key = 'sample_id'
    label_key = 'group_id'
    # test_size = 0.33
    random_state = 42
    test_size = 0.4
    max_tries = 10
    
    random.seed(random_state)
    np.random.seed(random_state)

    n_splits = 5
    # Read data
    adata = read_data(DATA_FILE)
    
    # Get patient IDs and labels
    patient_ids, labels, counts = get_patient_info(adata, patient_key, label_key)
    
    # filter
    dd = pd.DataFrame(index= patient_ids)
    dd['label'] = labels
    dd['counts'] =counts
    dd= dd[dd.counts>=100]
    
    labels = dd['label']
    patient_ids = dd.index
    counts = list(dd['counts'])

    print(labels.value_counts())
    minority_samples = labels.value_counts().min()
    n_splits = int(min(n_splits, minority_samples))
    # train_ids, test_ids = train_test_split( patient_ids, test_size=test_size, random_state=42, stratify=labels)

    # print(patient_ids)
    
    save_train_test_split(patient_ids,labels, counts, save_dir, test_size, random_state)

    # Save CV splits    
    save_cv_splits(patient_ids,labels, save_dir, n_splits=n_splits, random_state=random_state) 
