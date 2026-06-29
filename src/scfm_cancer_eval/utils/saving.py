import os
from os.path import join
import pandas as pd


def check_dir(saving_dir):
    dir_= saving_dir
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    return dir_
        
def save_supervised(adata, saving_dir, supervised_metrics_fig, supervised_metrics_df, cls_report, postfix= '_test'):
    dir_ = check_dir(saving_dir)    
     
    fname = join(dir_, 'classification' + postfix + '.png')
    supervised_metrics_fig.savefig(fname, dpi=100, bbox_inches="tight")

    fname = join(dir_, 'supervised_metrics'+ postfix + '.csv')
    supervised_metrics_df.to_csv(fname)
    
    df = pd.DataFrame(cls_report).transpose()
    fname = join(dir_, 'cls_report'+ postfix + '.csv')
    df.to_csv(fname)

def save_unsupervised(adata, saving_dir, unsupervised_metrics_df):
    dir_ = check_dir(saving_dir) 
    fname = join(dir_, 'unsupervised_metrics.csv')
    unsupervised_metrics_df.to_csv(fname)

def save_embeddings(adata, saving_dir, embeddings_fig):
    dir_ = check_dir(saving_dir) 
    fname=join(dir_, "embeddings.png")
    # embeddings_fig.title = model_name
    embeddings_fig.savefig(fname, dpi=100, bbox_inches='tight')

def save_h5ad(adata, saving_dir,model_name):
    dir_ = check_dir(saving_dir) 
    fname = join(dir_, f'data_{model_name}.h5ad') 
    adata.write_h5ad(fname,compression='gzip')
    print(f'Done saving results for {model_name}')
    
def save_all_resuts(adata, saving_dir, embeddings_fig, supervised_metrics_fig, supervised_metrics_df, unsupervised_metrics_df, model_name):
    dir_ = check_dir(saving_dir, model_name) 
        
    fname=join(dir_, "embeddings.png")
    embeddings_fig.title = model_name
    embeddings_fig.savefig(fname, dpi=100, bbox_inches='tight')
    
    fname = join(dir_, 'classification.png')
    supervised_metrics_fig.savefig(fname, dpi=100, bbox_inches="tight")

    fname = join(dir_, 'supervised_metrics.csv')
    supervised_metrics_df.to_csv(fname)


    fname = join(dir_, 'unsupervised_metrics.csv')
    unsupervised_metrics_df.to_csv(fname)
    
    fname = join(dir_, f'data_{model_name}.h5ad') 
    adata.write_h5ad(fname,compression='gzip')
    print(f'Done saving results for {model_name}')