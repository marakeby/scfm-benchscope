import scanpy as sc
import logging
import torch
import numpy as np
from os.path import join
import pickle

from geneformer import TranscriptomeTokenizer

from features.extractor import EmbeddingExtractor
from transformers import BertForMaskedLM, BertForSequenceClassification
from datasets import Dataset, load_from_disk
from utils.logs_ import get_logger
import os
from tqdm import trange
import torch.nn.functional as F
from transformers.training_args import TrainingArguments
from transformers import Trainer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def pad_batch(batch_dataset, device, pad_token_id, return_attention_mask=True):
    max_size = max(batch_dataset['length'])
    def pad_tensor(t, max_size, pad_token_id):
        return F.pad(t, pad=(0, max_size - t.numel()), mode='constant', value=pad_token_id)
    batch_ = [pad_tensor(x, max_size, pad_token_id)
              for x in batch_dataset['input_ids']]

    batch_ = torch.stack(batch_).to(device)

    if return_attention_mask:
        mask_ = [[1] * l + [0] * (max_size - l)
                 for l in batch_dataset['length']]
        mask_ = torch.tensor(mask_).to(device)
        return batch_, mask_

    return batch_

# get cell embeddings excluding padding
def mean_nonpadding_embs(embs, original_lens, device='cuda'):
    # mask based on padding lengths
    # mask = torch.arange(embs.size(1)).unsqueeze(0).to("cuda") < original_lens.unsqueeze(1)
    mask = torch.arange(embs.size(1)).unsqueeze(0).to(device) < original_lens.unsqueeze(1)
    # extend mask dimensions to match the embeddings tensor
    mask = mask.unsqueeze(2).expand_as(embs)
    # use the mask to zero out the embeddings in padded areas
    masked_embs = embs * mask.float()
    # sum and divide by the lengths to get the mean of non-padding embs
    mean_embs = masked_embs.sum(1) / original_lens.view(-1, 1).float()
    return mean_embs

class GeneformerExtractor(EmbeddingExtractor):
    def __init__(self, params):
        super().__init__(params)
        self.log = get_logger()
        self.log.info(f'GeneformerExtractor ({self.params})')
 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.model_name = self.params.get('model', '.')
        base_model_dir = self.params.get('model_dir', '.')
        
        # Check if model directory structure is flat or nested
        # Try direct path first (Geneformer-V2-104M/)
        direct_path = join(base_model_dir, self.model_name)
        nested_path = join(base_model_dir, self.model_name, self.model_name)
        
        if os.path.exists(join(direct_path, "config.json")):
            self.model_dir = direct_path
            self.log.info(f"Using direct model path: {self.model_dir}")
        elif os.path.exists(join(nested_path, "config.json")):
            self.model_dir = nested_path
            self.log.info(f"Using nested model path: {self.model_dir}")
        else:
            # Fallback to old behavior
            self.model_dir = join(base_model_dir, self.model_name)
            self.log.warning(f"Could not find config.json, using: {self.model_dir}")
        
        self.dict_dir = self.params.get('dict_dir', '.')
        self.model_input_size = self.params.get('model_input_size', 4096)
        self.batch_size = self.params.get('batch_size', 16)
        self.save_dir = self.params.get('save_dir', '.')
        self.layer = self.params.get('layer', -2)
        self.model_version = self.params.get('version', "V1")


        
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

        # Check for safetensors format first
        model_safetensors = join(self.model_dir, self.model_name, "model.safetensors")
        model_pytorch = join(self.model_dir, self.model_name, "pytorch_model.bin")
        
        if os.path.exists(model_safetensors):
            model_weights_file = "model.safetensors"
            self.use_safetensors = True
        elif os.path.exists(model_pytorch):
            model_weights_file = "pytorch_model.bin"
            self.use_safetensors = False
        else:
            # Fallback to default
            model_weights_file = "pytorch_model.bin"
            self.use_safetensors = False
            self.log.warning(f"Neither safetensors nor pytorch_model.bin found, using default")
        
        self.model_files = {
            "model_args": "config.json",
            "model_training": "training_args.bin",
            "model_weights": model_weights_file,
            "model_vocab": token_dictionary_file,
            "gene_name_id_path":gene_name_id_path,
            "gene_median_file": gene_median_file,
            "gene_mapping_file":gene_mapping_file
        }


    def load_model(self):
        self.log.info(f'Loading Model {self.model_dir} on {self.device}')
        print(f'Loading Model {self.model_dir} on {self.device}')
        self.model = BertForMaskedLM.from_pretrained(self.model_dir,output_attentions=False, output_hidden_states=True)
        self.model = self.model.to(self.device)
        self.log.info(f"Model successfully loaded from {self.model_dir}")
        self.log.info(f"Total parameters: { count_parameters(self.model)}")
        self.log.info(f"Trainable parameters: {count_trainable_parameters(self.model)}")

    def load_vocab(self):
        with open(self.model_files['model_vocab'], "rb") as f:
            self.vocab = pickle.load(f)
            #TODO remove this 
            # self.vocab = {value: key for key, value in self.vocab.items()}
   

        self.pad_token_id = self.vocab.get("<pad>")
        self.vocab_size = len(self.vocab)

        with open(self.model_files['gene_name_id_path'], "rb") as f:
            self.gene_name_id = pickle.load(f)

        # with open(self.token_dictionary_file, "rb") as f:
        #     self.token_dictionary = pickle.load(f)

    def load_tokenized_dataset(self, dataset_path):
        self.tokenized_dataset = load_from_disk(dataset_path)

    def tokenize_data(self, geneformer_reader, file_format):
        '''

        :param geneformer_reader: a GFLoader instance, required fields : processed_dir, dataset_name,
        :param cell_type_col:
        :param batch_key:
        :param file_format:
        :return: None, tokenized_dataset will be added to self
        '''

        processed_dir = geneformer_reader.processed_dir
        output_directory = geneformer_reader.processed_dir
        dataset_name = geneformer_reader.dataset_name
        cell_type_col = geneformer_reader.label_key
        batch_key = geneformer_reader.batch_key

        
        columns_to_keep = ["adata_order", batch_key, "label"]
        if self.gf_data_loader.train_test_split_dict:
            split_col = self.gf_data_loader.train_test_split_dict['id_column']
            columns_to_keep.append(split_col)
            
        cols_to_keep = dict(zip([cell_type_col] + columns_to_keep, ['cell_type'] + columns_to_keep))

        nproc = os.cpu_count()
        
            
        ## TODO: pass these variable in the yaml file
        # custom_attr_name_dict=None,
        # nproc=1,
        # chunk_size=512,
        # model_input_size=4096,
        # special_token=True,
        # collapse_gene_ids=True,
        # model_version="V1",
        # gene_median_file=GENE_MEDIAN_FILE,
        # token_dictionary_file=TOKEN_DICTIONARY_FILE,
        # gene_mapping_file=ENSEMBL_MAPPING_FILE,
        ####
        
        
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

        self.tokenized_dataset = tokenized_dataset

    @staticmethod
    def validate_config(params):
        assert 'params' in params and params is not None, "Missing 'params' in parameters"


    def fit_transform(self, gf_data_loader):

        self.gf_data_loader = gf_data_loader
        # load model and vocab
        if not hasattr(self, 'model'):
            self.load_model()
            self.load_vocab()

        if not hasattr(self, 'data_prepared'):
            # add ensembl_id to the var variable (assume gene names are in the var.index). Also remove genes without ensembl_id
            gf_data_loader.map_ensembl(self.gene_name_id)
            # prepare data, save loom file,
            gf_data_loader.prepare_data(self.save_dir, save_ext = 'loom')
            self.data_prepared = True

        # tokenize data
        if not hasattr(self, 'tokenized_dataset'):
            # tokenize data, saved to a dataset file
            self.tokenize_data(gf_data_loader, file_format='loom')
            
        if 'continue_training' in self.params:
            self.log.info(f'Continue Training')
            training_args = self.params['continue_training']
            training_output_dir = join(self.save_dir, 'training_output_dir')
            
            self.log.info(f'continue_training {gf_data_loader.params}')

            data_split_dict = gf_data_loader.train_test_split_dict

            # data_split_dict = gf_data_loader.params['datasplits_dir']
            self.continue_training(data_split_dict, training_args, training_output_dir)
        
        gf_data_loader.adata.obsm['X_geneformer'] = self.extract_embedding()
        return gf_data_loader.adata.obsm['X_geneformer'].copy()

    def extract_embedding(self):
        cell_embs_list = []
        rankings_list = []

        size = len(self.tokenized_dataset)
        self.model.eval()
        device = self.device
        pad_token_id = self.pad_token_id

        for i in trange(0, size, self.batch_size, desc="Geneformer (extracting embeddings)"):
            max_range = min(i + self.batch_size, size)
            batch_dataset = self.tokenized_dataset.select(list(range(i, max_range)))
            batch_dataset.set_format(type='torch')

            org_lengths = torch.tensor(batch_dataset['length']).to(device)
            batch, attn_mask = pad_batch(batch_dataset, device, pad_token_id)
            batch = batch.to(device)
            attn_mask = attn_mask.to(device)

            with torch.no_grad():
                model_output = self.model(input_ids=batch, attention_mask=attn_mask)

                # model_output = self._pass_batch(batch,
                #                                 attention_mask = attn_mask)

                embs = model_output.hidden_states[self.layer]

                # cell_embs = average_embeddings(embs, org_lengths)
                cell_embs = mean_nonpadding_embs(embs, org_lengths, device=self.device)

                # add cell embeddings to the list
                cell_embs_list.extend(cell_embs.detach().cpu().numpy())

                # now, get the ranking reconstruction
                out_rankings = (model_output.logits
                                .argmax(axis=-1)
                                .detach().cpu().numpy())

                # save the rankings with the original order
                rankings_list.extend(out_rankings)

                torch.cuda.empty_cache()
                del model_output
                del batch
                del attn_mask
                del embs
                del cell_embs

        cell_embeddings = np.array(cell_embs_list)

        output_rankings = rankings_list
        input_rankings = [np.array(item)
                          for item
                          in self.tokenized_dataset['input_ids']]

        # self.adata.obsm['X_geneformer'] = cell_embeddings
        # data.obsm["X_gf"] = model.get_latent_representation()
        # embedding_col = 'X_scVI'

        return cell_embeddings

    def continue_training(self, datasplits_dir, training_args, output_dir):       
            from geneformer import GeneformerPretrainer
            import pickle
            import pandas as pd

            from os.path import join
            from datasets import Dataset, load_from_disk

            # self.load_model()
            # self.load_vocab()

            training_args['output_dir'] = output_dir
            training_args['logging_dir'] = output_dir

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            device = self.device
            pad_token_id = self.pad_token_id
            self.model = self.model.to(device)
            

            #if we have training ids, filter data to include only these trining ids
            if datasplits_dir:
                
                split_dict = datasplits_dir
                test_ids = split_dict['train_test_split']['test_ids']
                train_ids = split_dict['train_test_split']['train_ids']
                split_col = split_dict['id_column']
                
                self.log.info(f'tokenized_dataset full {self.tokenized_dataset.shape}')
                train_ds = self.tokenized_dataset.filter(lambda x: x[split_col] in train_ids)
                self.log.info(f'train_ds filtered {train_ds.shape}')
            else: 
                train_ds = self.tokenized_dataset
                self.log.info(f'train_ds full {train_ds.shape}')

            # train_ds = self.tokenized_dataset
            self.log.info(f'train_ds full {train_ds.shape}')
            train_ds = train_ds.remove_columns("label")

            training_args = TrainingArguments(**training_args)

            #length file is needed for the trainer ?
            # lengths = [len(row['input_ids']) for row in train_ds]
            # lengths = [feature['length'] for feature in train_ds]
            lengths = [train_ds[i]["length"] for i in range(len(train_ds))]
            length_fname= join(output_dir, 'example_lengths_file.pkl')
            with open(length_fname, 'wb') as f:
                pickle.dump(lengths, f)

            # self.log.info(self.vocab )
            # define the trainer
            first_parameter = next(self.model.parameters())
            model_device = first_parameter.device


            self.log.info(f'model device {model_device}')
            self.log.info(training_args)
            self.log.info(f'train_ds {train_ds.shape}')
            self.log.info(length_fname)
            
            for f in train_ds.features:
                self.log.info(train_ds[f][0])
                
            # trying to solve error: return_dict_in_generate` is NOT set to `True`, but `output_hidden_states` is. When `return_dict_in_generate` is not `True`, `output_hidden_states` is ignored.'
            self.model.generation_config.return_dict_in_generate = True

    
            trainer = GeneformerPretrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_ds,
                    example_lengths_file=length_fname, #"genecorpus_30M_2048_lengths.pkl",
                    token_dictionary=self.vocab, 
                    # device = self.device
                )
            
            from transformers import GenerationConfig
            
            trainer.model.generation_config = trainer.model.generation_config or GenerationConfig()


            trainer.train()
            trainer.save_model(output_dir)
            # self.model = model
            self.trainer = trainer
            return trainer