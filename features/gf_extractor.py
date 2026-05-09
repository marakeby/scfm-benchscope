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
from data.data_loader import _stable_ensembl_from_var_label
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
    MODELS_PATH_KEYS = frozenset({"model_dir", "dict_dir"})

    def __init__(self, params):
        super().__init__(params)
        self.log = get_logger()
        self.log.info(f'GeneformerExtractor ({self.params})')
 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.model_name = self.params.get('model', '.')
        self.model_dir = self.params.get('model_dir', '.')
        self.model_dir = join(self.model_dir,self.model_name )
        self.dict_dir = self.params.get('dict_dir', '.')
        self.model_input_size = self.params.get(
            'model_input_size', self.params.get('input_size', 4096)
        )
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

        self.model_files = {
            "model_args": "config.json",
            "model_training": "training_args.bin",
            "model_weights": "pytorch_model.bin",
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
        
        
        # Upstream Geneformer: if model_version=="V1", TranscriptomeTokenizer ignores
        # custom gene_* paths and loads bundled files (often Git LFS pointers), causing
        # pickle errors (e.g. invalid load key 'v' from "version https://git-lfs...").
        # Pass model_version="V2" so the tokenizer keeps our dict_dir paths; set
        # special_token / model_input_size to match the real model series.
        tokenizer_mv = "V2" if self.model_version == "V1" else self.model_version
        special_token = self.model_version == "V2"
        self.tokenizer = TranscriptomeTokenizer(
            cols_to_keep,
            nproc=nproc,
            model_input_size=self.model_input_size,
            model_version=tokenizer_mv,
            special_token=special_token,
            token_dictionary_file=self.model_files['model_vocab'],
            gene_mapping_file=self.model_files['gene_mapping_file'],
            gene_median_file=self.model_files['gene_median_file'],
        )
        
        # Restrict to this dataset's loom only; default glob "*.loom" would merge
        # every .loom in processed_dir (e.g. stale full runs) and break obs alignment.
        self.tokenizer.tokenize_data(
            processed_dir,
            output_directory,
            dataset_name,
            file_format=file_format,
            input_identifier=dataset_name,
        )
        datase_fname = os.path.join(output_directory, f"{dataset_name}.dataset")
        tokenized_dataset = load_from_disk(datase_fname)

        self.tokenized_dataset = tokenized_dataset

    @staticmethod
    def validate_config(params):
        assert 'params' in params and params is not None, "Missing 'params' in parameters"

    @staticmethod
    def _fraction_ensembl_like(values) -> float:
        """Share of entries that contain a parseable Ensembl gene stable ID (ENS…G…)."""
        vals = list(values)
        if not vals:
            return 0.0
        ok = sum(_stable_ensembl_from_var_label(v) is not None for v in vals)
        return ok / len(vals)

    def _map_ensembl_ids(self, geneformer_reader):
        """
        Set ``adata.var['ensembl_id']`` and drop genes without a usable ID.

        If ``var['ensembl_id']`` already holds Ensembl-like IDs for a majority of genes,
        those values are used (after stable-ID extraction). Else if ``var_names`` are
        mostly Ensembl-like, IDs are taken from the index. Otherwise symbols in
        ``var_names`` are mapped via Geneformer's ``gene_name_id`` pickle.
        """
        adata = geneformer_reader.adata
        var = adata.var
        gene_to_ensembl = self.gene_name_id

        use_column = None
        col_frac = 0.0
        if "ensembl_id" in var.columns:
            col_vals = var["ensembl_id"].tolist()
            col_frac = self._fraction_ensembl_like(col_vals)
            if col_frac > 0.5:
                use_column = "ensembl_id"

        index_vals = list(adata.var_names.astype(str))
        index_frac = self._fraction_ensembl_like(index_vals)

        if use_column is not None:
            parsed = [_stable_ensembl_from_var_label(v) for v in var[use_column].tolist()]
            geneformer_reader.adata.var["ensembl_id"] = parsed
            self.log.info(
                "Geneformer: using var[%r] for Ensembl IDs (%.1f%% of genes match ENS*G* stable-id pattern).",
                use_column,
                100.0 * col_frac,
            )
        elif index_frac > 0.5:
            parsed = [_stable_ensembl_from_var_label(v) for v in index_vals]
            geneformer_reader.adata.var["ensembl_id"] = parsed
            self.log.info(
                "Geneformer: var_names look like Ensembl (%.1f%% parseable); skipping gene_name_id map.",
                100.0 * index_frac,
            )
        else:
            geneformer_reader.adata.var["ensembl_id"] = var.index.map(gene_to_ensembl)
            self.log.info("Geneformer: mapping gene symbols → Ensembl via gene_name_id dict.")

        nan_idx = geneformer_reader.adata.var.ensembl_id.isna()
        if nan_idx.any():
            n_removed = int(nan_idx.sum())
            geneformer_reader.adata = geneformer_reader.adata[:, ~nan_idx]
            self.log.warning(f"warning: genes dont have ensembl IDs {n_removed}. Genes without ensembl ID are REMOVED")

    def _prepare_geneformer_data(self, geneformer_reader, save_ext: str = "loom"):
        """
        Persist the AnnData into Geneformer-readable files inside a processed directory.
        Previously this lived in `GFLoader.prepare_data`; we inline it so the dataset loader
        can be a plain `H5ADLoader`.
        """
        adata = geneformer_reader.adata

        # `total_counts` is normally produced by `qc_data()`; compute if missing for robustness.
        if "total_counts" not in adata.obs.columns:
            import scanpy as sc
            sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

        adata.obs["n_counts"] = adata.obs["total_counts"]
        adata.obs["adata_order"] = adata.obs.index.tolist()

        geneformer_reader.processed_dir = join(self.save_dir, "processed_data")
        if not os.path.exists(geneformer_reader.processed_dir):
            os.makedirs(geneformer_reader.processed_dir)

        dataset_name = geneformer_reader.dataset_name
        if save_ext == "loom":
            loom_path = os.path.join(geneformer_reader.processed_dir, f"{dataset_name}.loom")
            adata.write_loom(loom_path)
            geneformer_reader.procssed_file = loom_path  # kept for backward compatibility with older code
            self.log.info(f"saving loom file to {loom_path}")
        elif save_ext == "h5ad":
            h5ad_path = os.path.join(geneformer_reader.processed_dir, f"{dataset_name}.h5ad")
            adata.write_h5ad(h5ad_path)
            geneformer_reader.procssed_file = h5ad_path
            self.log.info(f"saving h5ad file to {h5ad_path}")
        else:
            raise ValueError(f"Unsupported save_ext: {save_ext}")


    def fit_transform(self, gf_data_loader):

        self.gf_data_loader = gf_data_loader
        # load model and vocab
        if not hasattr(self, 'model'):
            self.load_model()
            self.load_vocab()

        if not hasattr(self, 'data_prepared'):
            # geneformer preparation step (inlined from `GFLoader`)
            self._map_ensembl_ids(gf_data_loader)
            self._prepare_geneformer_data(gf_data_loader, save_ext="loom")
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

            # Patch for 🤗 Transformers Trainer API changes:
            # Newer versions call `_get_train_sampler(dataset)` while older Geneformer
            # pretrainer implementations may define `_get_train_sampler(self)` only.
            class _PatchedGeneformerPretrainer(GeneformerPretrainer):
                def _get_train_sampler(self, *args, **kwargs):  # dataset may be passed
                    return super()._get_train_sampler()

                def _get_eval_sampler(self, *args, **kwargs):  # dataset may be passed
                    return super()._get_eval_sampler()

                def _get_test_sampler(self, *args, **kwargs):  # dataset may be passed
                    return super()._get_test_sampler()

                def _save(self, output_dir: str | None = None, state_dict=None):
                    """
                    Transformers>=4.4x may try to call `self.data_collator.tokenizer.save_pretrained(...)`.
                    Geneformer uses a custom collator where `.tokenizer` is not a HF tokenizer, which breaks
                    checkpoint saving. We temporarily provide a no-op `save_pretrained` to keep training going.
                    """
                    collator = getattr(self, "data_collator", None)
                    had_tokenizer = False
                    orig_tokenizer = None

                    class _NoOpTokenizer:
                        def save_pretrained(self, *args, **kwargs):
                            return None

                    try:
                        if collator is not None and hasattr(collator, "tokenizer"):
                            had_tokenizer = True
                            orig_tokenizer = collator.tokenizer
                            if not hasattr(orig_tokenizer, "save_pretrained"):
                                collator.tokenizer = _NoOpTokenizer()
                        return super()._save(output_dir=output_dir, state_dict=state_dict)
                    finally:
                        if had_tokenizer and collator is not None:
                            collator.tokenizer = orig_tokenizer

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
                
            # Ensure a GenerationConfig exists before tweaking generation flags.
            # Some HF models set `generation_config=None` until first initialized.
            from transformers import GenerationConfig
            if getattr(self.model, "generation_config", None) is None:
                self.model.generation_config = GenerationConfig()

            # trying to solve error: return_dict_in_generate` is NOT set to `True`, but `output_hidden_states` is.
            # When `return_dict_in_generate` is not `True`, `output_hidden_states` is ignored.
            self.model.generation_config.return_dict_in_generate = True

    
            trainer = _PatchedGeneformerPretrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_ds,
                    example_lengths_file=length_fname, #"genecorpus_30M_2048_lengths.pkl",
                    token_dictionary=self.vocab, 
                    # device = self.device
                )
            
            # Keep trainer + model generation configs consistent.
            trainer.model.generation_config = trainer.model.generation_config or GenerationConfig()


            trainer.train()
            trainer.save_model(output_dir)
            # self.model = model
            self.trainer = trainer
            return trainer