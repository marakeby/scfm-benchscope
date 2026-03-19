from utils.logs_ import get_logger
import torch
from os.path import join
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from geneformer import DataCollatorForCellClassification
from features.gf_extractor import GeneformerExtractor
import logging
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
import pickle
from geneformer import TranscriptomeTokenizer
logger = get_logger()
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Suppress seaborn's internal deprecation warning about 'vert' parameter
warnings.filterwarnings(
    'ignore',
    category=PendingDeprecationWarning,
    message='vert: bool will be deprecated',
    module='seaborn.categorical'
)
from collections import defaultdict
import torch.nn as nn
from evaluation.eval import eval_classifier, plot_classifier
from datasets import Dataset, load_from_disk
from geneformer.collator_for_classification import DataCollatorForCellClassification
# Import refactored utilities
from models.train_utils import (
    # DataCollatorForPatientClassification,
    # BertMILClassifier,
    train_classifier_cell,
    train_patient_classifier,
    save_results,
    get_splits_cv
)

class GFFineTuneModel:
    """Geneformer Fine-tuning Model for cell type classification.
    
    This class implements a fine-tuned Geneformer model for cell type classification tasks.
    It inherits the base Geneformer model and adds classification-specific functionality.
    """
    
    def __init__(self, params): #model_dir, dict_dir, model_input_size=4096, output_dir):
        # super().__init__(params)
        
        
        """Initialize the fine-tuning model.
        
        Args:
            model_dir (str): Directory containing the pretrained model
            dict_dir (str): Directory containing the model dictionaries
            model_input_size (int): Size of model input sequences
        """
        print(params)
        self.params = params['params']
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'GFFineTuneModel ({self.params})')
        
        self.save_dir = self.params['save_dir']
        
        self.model_name = self.params['model']
        self.model_dir = self.params['model_dir']
        self.model_dir = join(self.model_dir,self.model_name)
        self.dict_dir = self.params['dict_dir']
        self.model_input_size =  self.params['model_input_size']
        self.batch_size = self.params['batch_size']
        self.model_version = self.params.get('version', "V1")
        self.freeze_layers = self.params.get('freeze_layers', 0)
        self.label_map = self.params['label_map'] #dictionary e.g. # {'Pre': 0, 'Post': 1}
        self.cv = self.params.get('cv', False)
        self.onesplit = self.params.get('onesplit', False)
        self.cls_level = self.params.get('cls_level', 'patient')
        self.train_funcs = self.params['train_funcs']
        self.epoch = self.params['epoch']
            
        self.evaluate = params['eval']
        self.viz = params['viz']
        
        if self.model_version == "V1":
            token_dictionary_file = join(self.dict_dir,"token_dictionary_gc30M.pkl")
            gene_median_file = join(self.dict_dir,"gene_median_dictionary_gc30M.pkl")
            gene_mapping_file = join(self.dict_dir,"ensembl_mapping_dict_gc30M.pkl")
            gene_name_id_path =  join(self.dict_dir,"gene_name_id_dict_gc30M.pkl")

        else:
            token_dictionary_file = join(self.dict_dir,"token_dictionary_gc104M.pkl")
            gene_median_file = join(self.dict_dir,"gene_median_dictionary_gc104M.pkl")
            gene_mapping_file = join(self.dict_dir,"ensembl_mapping_dict_gc104M.pkl")
            gene_name_id_path =  join(self.dict_dir,"gene_name_id_dict_gc104M.pkl")

        self.model_files = {
            "model_args": "config.json",
            "model_training": "training_args.bin",
            "model_weights": "pytorch_model.bin",
            "model_vocab": token_dictionary_file,
            "gene_name_id_path":gene_name_id_path,
            "gene_median_file": gene_median_file,
            "gene_mapping_file":gene_mapping_file
        }            
        self.label_encoder = LabelEncoder()
    def load_vocab(self):
        with open(self.model_files['model_vocab'], "rb") as f:
            self.vocab = pickle.load(f)

        self.pad_token_id = self.vocab.get("<pad>")
        self.vocab_size = len(self.vocab)

        with open(self.model_files['gene_name_id_path'], "rb") as f:
            self.gene_name_id = pickle.load(f)
            
    def tokenize_data(self, geneformer_reader, file_format):
        '''

        :param geneformer_reader: a GFLoader instance, required fields : processed_dir, dataset_name,
        :param cell_type_col:
        :param batch_key:
        :param file_format:
        :return: None, tokenized_dataset will be added to self
        '''

        logger.info('Tokenizing dataset')
        processed_dir = geneformer_reader.processed_dir
        output_directory = geneformer_reader.processed_dir
        dataset_name = geneformer_reader.dataset_name
        cell_type_col = geneformer_reader.label_key
        batch_key = geneformer_reader.batch_key

        columns_to_keep = ["adata_order", batch_key, "label"]
        
        if geneformer_reader.train_test_split_dict:
            split_col = geneformer_reader.train_test_split_dict['id_column']
            columns_to_keep.append(split_col)
            
        cols_to_keep = dict(zip([cell_type_col] + columns_to_keep, ['cell_type'] + columns_to_keep))
        
        logger.info(f'cols to keep {cols_to_keep}')

        nproc = os.cpu_count()
        # self.tokenizer = TranscriptomeTokenizer(cols_to_keep, nproc=nproc, model_input_size=self.model_input_size)
        self.tokenizer = TranscriptomeTokenizer(cols_to_keep, nproc=nproc, 
                                                model_input_size=self.model_input_size,
                                                model_version = self.model_version, 
                                                token_dictionary_file=self.model_files['model_vocab'], 
                                                gene_mapping_file=self.model_files['gene_mapping_file'],
                                                gene_median_file=self.model_files['gene_median_file'])
        self.tokenizer.tokenize_data(processed_dir,
                                     output_directory,
                                     dataset_name,
                                     file_format=file_format)
        
        datase_fname = os.path.join(output_directory, f"{dataset_name}.dataset")
        tokenized_dataset = load_from_disk(datase_fname)
        logger.info(tokenized_dataset)

        self.tokenized_dataset = tokenized_dataset         
            
    def encode_labels(self):
        """Encode labels using LabelEncoder"""
        # self.adata.obs['label'] = self.label_encoder.fit_transform(self.adata.obs[self.label_key])
        for key, value in self.label_map.items():
               self.label_map[key] = int(value)
        
        self.gf_data_loader.adata.obs['label'] = self.gf_data_loader.adata.obs[self.label_key].map(self.label_map) # {'Pre': 0, 'Post': 1}
        # self.label_names = list(self.label_encoder.classes_)
        self.label_names = list(self.label_map.keys())
        logger.info(f"Label classes: {self.label_names}")
        logger.info(f"Label classes: {self.gf_data_loader.adata.obs['label'].value_counts()}")
        
    def train(self, gf_data_loader):
         # load model and vocab
        # if not hasattr(self, 'model'):
        #     self.load_model()
        self.label_key = gf_data_loader.label_key
        self.gf_data_loader = gf_data_loader
        self.sample_id = self.gf_data_loader.train_test_split_dict['id_column']
        self.load_vocab()
        
        # train_func_map = {'vote': self.__train_vote, 'avg': self.__train_avg_expression, 'mil': self.__train_mil}
        train_func_map = {'vote': self.__train_vote,   'mil': self.__train_mil}

        train_fcs = [] 
        prefix =[]
        for fnc in self.train_funcs:
            train_fcs.append(train_func_map[fnc])
            prefix.append(fnc)
            
        
        if not hasattr(self, 'data_prepared'):
            # add ensembl_id to the var variable (assume gene names are in the var.index). Also remove genes without ensembl_id
            gf_data_loader.map_ensembl(self.gene_name_id)
            self.encode_labels()
            gf_data_loader.prepare_data(self.save_dir, save_ext = 'loom')# prepare data, save loom file,
            
            self.data_prepared = True
            
        self.adata = gf_data_loader.adata
        # tokenize data
        if not hasattr(self, 'tokenized_dataset'):
            self.tokenize_data(gf_data_loader, file_format='loom')# tokenize data, saved to a dataset file
            
        
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
            id_column, n_splits, train_ids_list, test_ids_list = get_splits_cv(self.gf_data_loader)
            for tr_f, pre in zip(train_fcs, prefix):
                self.__train_cv(tr_f, id_column, n_splits, train_ids_list, test_ids_list, pre)
                

    def split_data(self, id_column, train_ids, test_ids):
        """Split data into train and test sets"""
        adata_test = self.adata[self.adata.obs[id_column].isin(test_ids)]
        adata_train = self.adata[self.adata.obs[id_column].isin(train_ids)]
        return adata_train, adata_test
    
    
    def __train_cell(self, train_ids, test_ids, split_col, evaluate=False, viz=False):
        """Cell-level predictions training"""
        logger.info('Training model')

        
        train_ds = self.tokenized_dataset.filter(lambda x: x[split_col] in train_ids)
        test_ds = self.tokenized_dataset.filter(lambda x: x[split_col] in test_ids)
        adata_train, adata_test = self.split_data(split_col, train_ids, test_ids)
        
        
        model_dir = self.model_dir
        dict_dir = self.dict_dir
        # output_dir = self.output_dir
        output_dir = self.save_dir
        num_labels = 2  #TODO: MAKE THE NUMBER OF LABELS VARIABLE 
        device = self.device
        vocab_dir = self.model_files['model_vocab']
        
        trainer, y_test, y_pred, y_pred_score = train_classifier_cell(train_ds, test_ds, model_dir, vocab_dir, output_dir, num_labels, device, freeze_layers=self.freeze_layers, batch_size=self.batch_size, num_train_epochs= self.epoch)
        
        
        adata_test.obs['pred'] = y_pred
        adata_test.obs['pred_score'] = y_pred_score[:, 1]
        
        if evaluate:
            save_dir = join(self.save_dir, 'cell_level_pred')
            os.makedirs(save_dir, exist_ok=True)
            
            adata_test.obs.to_csv(join(save_dir, f'cell_pred_test.csv'))
            adata_train.obs.to_csv(join(save_dir, f'cell_pred_train.csv'))
            
            metrics_df, cls_report = eval_classifier(y_test, y_pred, y_pred_score,
                                                  estimator_name=self.model_name, label_names=self.label_names)
            
            pred_df = adata_test.obs['pred'].copy()
            save_results(pred_df, metrics_df,cls_report,save_dir, 'cell', viz, self.model_name, self.label_names)
            
            return adata_test.obs, metrics_df
        
        return adata_test.obs, None
    
    def __train_mil(self, train_ids, test_ids, split_col, evaluate=False, viz=False):
        """Multi-instance learning training"""
        logger.info('Training model (Multi instance Learning (MIL))')
        
        train_ds = self.tokenized_dataset.filter(lambda x: x[split_col] in train_ids)
        test_ds = self.tokenized_dataset.filter(lambda x: x[split_col] in test_ids)
        adata_train, adata_test = self.split_data(split_col, train_ids, test_ids)
        
        train_ds = train_ds.rename_column(split_col, "sample_id")
        test_ds = test_ds.rename_column(split_col, "sample_id")
        
        # train_ds.add
        
        
        model_dir = self.model_dir
        dict_dir = self.dict_dir
        # output_dir = self.output_dir
        output_dir = self.save_dir
        num_labels = 1  #TODO: MAKE THE NUMBER OF LABELS VARIABLE 
        device = self.device
        vocab_dir = self.model_files['model_vocab']
        
        trainer,sample_ids,  y_test, y_pred, y_pred_score = train_patient_classifier(train_ds, test_ds,split_col, model_dir, vocab_dir, output_dir, num_labels, device, epochs= self.epoch)

        logger.info(y_test)
        logger.info(y_pred)
        logger.info(y_pred_score)
        
        pred_df = pd.DataFrame({ 'label': y_test,'pred': y_pred,
                                'pred_score': y_pred_score,  # Assuming binary classification
                               'sample_id':  sample_ids})
            
        y_pred_score_p = pred_df.groupby('sample_id')['pred_score'].mean()
        y_pred_p = pred_df.groupby('sample_id')['pred'].agg(lambda x: x.value_counts().idxmax())
        y_test_p = pred_df.groupby('sample_id')['label'].first().reindex(y_pred_score_p.index)

        pred_df = pd.concat([y_test_p, y_pred_p, y_pred_score_p], axis=1)
        
        # adata_test.obs['pred'] = y_pred
        # adata_test.obs['pred_score'] = y_pred_score[:, 1]
        
        # pred_df = pd.DataFrame({'id': pids, 'label': y_true, 'pred': preds, 'pred_score': pred_scores})
        metrics_df, cls_report = eval_classifier(pred_df['label'], pred_df['pred'], pred_df['pred_score'],
                                                  estimator_name=self.model_name, label_names=self.label_names)
        if evaluate:
            
            save_results(pred_df, metrics_df, cls_report, self.saving_dir, postfix='mil', viz=viz, model_name=self.model_name, label_names=self.label_names)
  
        
        return pred_df, metrics_df
    
    
    def __train_vote(self, train_ids, test_ids, split_col, evaluate=False, viz=False):
        """Majority vote predictions training"""
        logger.info('Training model (Majority Vote)')
        
        cell_pred_test_df, cell_metrics_df = self.__train_cell(train_ids, test_ids, split_col,
                                                             evaluate=True, viz=False)
        
        sample_pred_test_df, sample_metrics_test_df = self.save_patient_level(cell_pred_test_df, 
                                                                            evaluate, viz, 
                                                                            postfix='test', 
                                                                            model='vote')
        
        return sample_pred_test_df, sample_metrics_test_df
    
    
    def __train_avg(self, train_ids, test_ids, split_col, evaluate=True, viz=False, postfix=""):
        """
        Probability averaging at patient level.
        """
        cell_pred_test_df, _ = self.__train_cell(train_ids, test_ids, split_col, evaluate=True, viz=False, postfix=postfix)

        obs = cell_pred_test_df
        y_score_p = obs.groupby(self.sample_id)["pred_score"].mean()
        y_pred_p = (y_score_p > 0.5).astype(int)
        y_true_p = obs.groupby(self.sample_id)["label"].first().reindex(y_score_p.index)

        pred_df = pd.DataFrame({"label": y_true_p, "pred": y_pred_p, "pred_score": y_score_p})

        from evaluation.eval import eval_classifier
        metrics_df, cls_report = eval_classifier(pred_df["label"], pred_df["pred"], pred_df["pred_score"],
                                                 estimator_name=self.model_name, label_names=self.label_names)
        if evaluate:
            save_results(pred_df, metrics_df, cls_report, self.save_dir, postfix=f"avg_{postfix}" if postfix else "avg",
                         viz=viz, model_name=self.model_name, label_names=self.label_names)
            
        return pred_df, metrics_df

    def save_patient_level(self, adata_subset, evaluate=False, viz=False, postfix="", model=""):
        """Save patient-level predictions and metrics"""
        logger.info('Saving sample level performance')
        
        obs = adata_subset
        y_pred_score_p = obs.groupby(self.sample_id)['pred_score'].mean()
        y_pred_p = obs.groupby(self.sample_id)['pred'].agg(lambda x: x.value_counts().idxmax())
        y_test_p = obs.groupby(self.sample_id)['label'].first().reindex(y_pred_score_p.index)
        
        pred_df = pd.concat([y_test_p, y_pred_p, y_pred_score_p], axis=1)
        pred_df.columns = ['label', 'pred', 'pred_score']
        
        metrics_df, cls_report = eval_classifier(pred_df['label'], pred_df['pred'], pred_df['pred_score'],
                                              estimator_name=self.model_name, label_names=self.label_names)
        
        if evaluate:
            save_results(pred_df, metrics_df,cls_report, self.save_dir, 'vote', viz, self.model_name, self.label_names)
        
        return pred_df, metrics_df
    
    def __train_cv(self, train_fnc, id_column, n_splits, train_ids_list, test_ids_list, prefix=''):
        """Run cross-validation training"""
        pred_list = []
        metrics_list = []
        logger.info(f'Running crossvalidation with {n_splits} folds')
        
        for i in range(n_splits):
            logger.info(f'---------- fold {i+1}----------')
            train_ids, test_ids = train_ids_list[i], test_ids_list[i]
            
            pred_df, metric_df = train_fnc(train_ids, test_ids, id_column)
            pred_df['fold'] = f'fold_{i+1}'
            metric_df['fold'] = f'fold_{i+1}'
            pred_list.append(pred_df)
            metrics_list.append(metric_df)
        
        preds = pd.concat(pred_list)
        metrics = pd.concat(metrics_list)
        
        save_dir = join(self.save_dir, 'cv')
        os.makedirs(save_dir, exist_ok=True)
        
        preds.to_csv(join(save_dir, f'{prefix}cv_predictions.csv'))
        metrics.to_csv(join(save_dir, f'{prefix}cv_metrics.csv'))
        mteric_mean = metrics.groupby(['Metrics']).mean(numeric_only=True)
        mteric_std = metrics.groupby(['Metrics']).std(numeric_only=True)

        mteric_mean.to_csv(join(save_dir, f'{prefix}cv_metrics_mean.csv'))
        mteric_std.to_csv(join(save_dir, f'{prefix}cv_metrics_std.csv'))

        
        # Plot metrics
        metrics.fillna(0, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Metrics', y=self.model_name, data=metrics, orientation='vertical')
        plt.title('Cross-Validation Metric Distribution')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(save_dir, f'{prefix}cv_metrics_boxplot.png'))
        plt.close()
        
        return preds, metrics
    
    def predict(self, test_ds=None):
        """Make predictions using the trained model.
        
        Args:
            test_ds: Optional test dataset. If None, uses the stored test dataset.
            
        Returns:
            Predictions from the model
        """
        if test_ds is None:
            test_ds = self.test_ds
        return self.trainer.predict(test_ds)
    
    def save(self, output_dir):
        """Save the trained model.
        
        Args:
            output_dir (str): Directory to save the model
        """
        if self.trainer is not None:
            self.trainer.save_model(output_dir)
            
    def load(self, model_dir):
        """Load a trained model.
        
        Args:
            model_dir (str): Directory containing the saved model
        """
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model = self.model.to(self.device) 
