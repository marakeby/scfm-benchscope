from utils.logs_ import get_logger
import torch
import os
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import json

import torch
import numpy as np

from features.features_scgpt import scGPT_Processor

class scGPT_Model():
    
    def __init__(self, model_dir):
        self.logger = get_logger()
        
        self.model_dir = model_dir
        self.model_files = {
                    "model_args": "args.json", 
                    "model_vocab": "vocab.json",
                    "model_weights": "best_model.pt"
                 }
        
        vocab_file = os.path.join(self.model_dir, self.model_files['model_vocab'])
        config_file = os.path.join(self.model_dir, self.model_files['model_args'])

        self.vocab = self._load_vocab(vocab_file)
        self.config = self._load_config(config_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def _load_config(self, config_file):
        # model
        with open(config_file, "r") as f:
            model_configs = json.load(f)
        self.logger.info(
        f"Loading config from {config_file}."
    )
        
        
    def get_default_params(self):
        default_params = dict(seed= 0,
                       # ---> scGPT_human defaults
                       embsize= 512,
                       nheads= 8,
                       d_hid= 512,
                       nlayers=12,
                       nlayers_cls= 3,
                       dropout= 0.2,
                       pad_token= "<pad>",
                       pad_value = -2,
                       mask_value= -1,
                       mask_ratio=[0.25, 0.5, 0.75],
                       do_mvc = True, # MVC in args.json
                       input_emb_style = "continuous",
                       n_bins = 51,
                       use_fast_transformer= True,
                       # <--- scGPT_human defaults
                       # ---> scgpt.TransformerModel class default
                       n_cls = 1, 
                       do_dab = False, 
                       use_batch_labels = False, 
                       domain_spec_batchnorm = False, 
                       cell_emb_style = "cls", 
                       mvc_decoder_style = "inner product", 
                       ecs_threshold = 0.3, 
                       explicit_zero_prob = False, 
                       fast_transformer_backend = "flash", 
                       pre_norm = False, 
                       # <--- scgpt.TransformerModel class default
                       do_cce = False,
                       do_ecs = False,
                       do_cls = False,
                       max_seq_len = 1200,
                       per_seq_batch_sample = False, 
                       shuffle = False,
                       append_cls = True,
                       permute_gene_order = True)
        return default_params
    
    def _build_model(self):
        self.default_params = self.get_default_params()
        ntokens = len(self.vocab)
        self.logger.info(ntokens)
        model = TransformerModel(ntoken=ntokens,
            d_model = self.default_params['embsize'],
            nhead = self.default_params['nheads'],
            d_hid = self.default_params['d_hid'],
            nlayers = self.default_params['nlayers'],
            nlayers_cls = self.default_params['nlayers_cls'], 
            n_cls = self.default_params['n_cls'],
            vocab = self.vocab, 
            dropout = self.default_params['dropout'], 
            pad_token = self.default_params['pad_token'], 
            pad_value = self.default_params['pad_value'], 
            do_mvc = self.default_params['do_mvc'], 
            do_dab = self.default_params['do_dab'], 
            use_batch_labels = self.default_params['use_batch_labels'],            
            # num_batch_labels = self.default_params['num_batch_labels'],
            domain_spec_batchnorm = self.default_params['domain_spec_batchnorm'], 
            input_emb_style = self.default_params['input_emb_style'], 
            n_input_bins = self.default_params['n_bins'], 
            cell_emb_style = self.default_params['cell_emb_style'], 
            mvc_decoder_style = self.default_params['mvc_decoder_style'], 
            ecs_threshold = self.default_params["ecs_threshold"],
            explicit_zero_prob = self.default_params["explicit_zero_prob"], 
            use_fast_transformer = self.default_params["use_fast_transformer"], 
            fast_transformer_backend = self.default_params["fast_transformer_backend"],
            pre_norm = self.default_params['pre_norm']
        )
        
        
        return model
    
    def _load_vocab(self, vocab_file):
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        self.vocab = vocab
        return vocab
            
    
    def load_pretrained_model(self):
        model_file = os.path.join(self.model_dir, 
                                  self.model_files['model_weights'])
        
        self.logger.info(f"Loading model from {model_file}")
        
        ntokens = len(self.vocab)  # size of vocabulary
        
        model = self._build_model()

        try:
            model.load_state_dict(torch.load(model_file))
            self.logger.info(f"Loading all model params from {model_file}")
        except:
            self.logger.info('Failed to load params')
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                self.logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        total_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

        pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
        self.logger.info(f'# of params {total_param_count},# of frozen params {pre_freeze_param_count}')
        model.to(self.device)
        self.model = model
        return model


    
    def get_embeddings(self):
        pass
    
    def fine_tune(self):
        pass

        self.model = BertForMaskedLM.from_pretrained(self.model_dir,output_attentions=False, output_hidden_states=True)
        self.model = self.model.to(self.device)
        self.logger.info(f"Model successfully loaded from {self.model_dir}")
        
    def load_tokenized_dataset(self, dataset_path):
        
        self.tokenized_dataset = load_from_disk(dataset_path)

        
    def load_vocab(self):

        parent_dir = self
        with open(self.model_files['model_vocab'], "rb") as f:
            self.vocab = pickle.load(f)
        
        self.pad_token_id = self.vocab.get("<pad>")
        self.vocab_size = len(self.vocab)  

        with open(self.model_files['gene_name_id_path'], "rb") as f:
            self.gene_name_id = pickle.load(f)

   

    def extract_embeddings(self, scgpt_reader, gene_col = 'gene_name', output_embedding_key= "X_scGPT", MVC=True, ECS=False, CLS=False, CCE=False, amp=True, use_batch_labels=False, pad_token="<pad>"):

        vocab = self.vocab
        model = self.model
        device = self.device
        # data = scgpt_reader.data

        model.eval()

        processor = scGPT_Processor()
        tokenized_data = processor.tokenize_data(scgpt_reader,vocab, gene_col=gene_col )
        data_loader = processor.get_dataloader(tokenized_data, 8)

        cell_embeddings = []
        mlm_output = []
        batch_idxs = []
        mvc_output = []
        masked_values = []
        for batch, batch_data in enumerate(data_loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

            with torch.no_grad() and torch.cuda.amp.autocast(enabled=amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask = src_key_padding_mask,
                        batch_labels = batch_labels if use_batch_labels else None,
                        # gene expression prediction from cell embedding? GEPC
                        MVC =MVC,
                        # elastic cell similarity
                        ECS =ECS, 
                        # cell type classification objective
                        CLS =CLS, 
                        # contrastive cell embedding objective
                        CCE =CCE
                    )
                    cell_embeddings.append(output_dict["cell_emb"].detach().cpu().numpy())
                    mlm_output.append(output_dict["mlm_output"].detach().cpu().numpy())
                    if MVC:
                        mvc_output.append(output_dict["mvc_output"].detach().cpu().numpy())

        cell_embeddings = np.concatenate(cell_embeddings, axis=0)
        # normalize cell embeddings
        cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis = 1, keepdims = True)
        # flatten the list of mlm_output
        smlm_output = np.concatenate(mlm_output, axis=0)

        if MVC:
            mvc_output = np.concatenate(mvc_output, axis=0)
        else:
            mvc_output = None

        scgpt_reader.adata.obsm[output_embedding_key] = cell_embeddings

        return 
