from os.path import dirname, abspath, join
import pandas as pd
import yaml

def get_split_dict( datasplits_dir):
    ## load trian and test sample ids from the corresponding files (train_ids.csv, test_ids.csv). Put the results in a dictionary
    # datasplits_dir = join(base_dir, datasplits_dir)
    test_ids = pd.read_csv(join(datasplits_dir, 'test_ids.csv'))
    train_ids = pd.read_csv(join(datasplits_dir, 'train_ids.csv'))

    split_dict= {'split_on': 'sample_id', 'train_ids': train_ids['ID'].astype(str).values,
                 'test_ids': test_ids['ID'].astype(str).values}
    return split_dict

# read param YAML file and parse first level
def get_params(params_file):
    with open(params_file) as f:
        run_params = yaml.safe_load(f)
    run_id = run_params['run_id']
    data_params = run_params['dataset']
    qc = run_params['qc']
    features = run_params['features']
    evals = run_params['eval']
    return run_id, data_params, qc, features, evals

