from torch.utils.data import Dataset, DataLoader
from typing import Dict
import torch

from scipy.sparse import issparse
import numpy as np


class SeqDataset(Dataset):
    def __init__(self, 
                 data: Dict[str, torch.Tensor],
                 gene_key: str = "gene_ids"):
        self.data = data
        self.gene_key = gene_key

    def __len__(self):
        return self.data[self.gene_key].shape[0]

    def __getitem__(self, idx):
        d_ = {k: v[idx] for k, v in self.data.items()}
        d_['idx'] = idx
        return d_


class scGPT_Processor():
    
     def __init__(self):
        pass
    
     def tokenize_data(self, data_,vocab, input_layer_key = "X_binned",include_zero_genes= False, max_seq_len=1200, pad_token="<pad>", pad_value=-2,append_cls= False, gene_col = "gene_name", batch_key=None):
        
        from scgpt.tokenizer import tokenize_and_pad_batch
        input_data = (
                data_.adata.layers[input_layer_key].A
                if issparse(data_.adata.layers[input_layer_key])
                else data_.adata.layers[input_layer_key]
            )
        
        genes = data_.adata.var[gene_col].values.tolist()
        gene_ids = np.array(vocab(genes), dtype=int)

        tokenized_data = tokenize_and_pad_batch( input_data,
            gene_ids,
            max_len = max_seq_len,
            vocab = vocab,
            pad_token = pad_token,
            pad_value = pad_value,
            # append <cls> token at the beginning
            append_cls = append_cls,  
            include_zero_gene = include_zero_genes       )

        if batch_key is not None:
            batch_labels = (
                data
                .adata
                .obs[data.batch_id_col]
                .values
                )
            batch_labels = torch.from_numpy(np.array(batch_labels)).long()
            tokenized_data["batch_labels"] = batch_labels
        
        tokenized_data["values"] = tokenized_data["values"].float()
        return tokenized_data
    
     def get_dataloader(self, 
        tokenized_data, batch_size,
        per_seq_batch_sample = False,
        shuffle = False,
        intra_domain_shuffle = False,
        drop_last = False,
        num_workers= 0):
    
        data_pt = {
            "gene_ids": tokenized_data["genes"],
            "values": tokenized_data["values"]
        }
    
        dataset = SeqDataset(data_pt)

        data_loader = DataLoader(
                dataset = dataset,
                batch_size = batch_size,
                shuffle = shuffle,
                drop_last = drop_last,
                num_workers = num_workers,
                pin_memory = True,
            )
        return data_loader


