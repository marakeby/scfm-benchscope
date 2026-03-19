
import random
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
from features.load import *
from utils.logs_ import get_logger
from features.extractor import EmbeddingExtractor


class scfoundationExtractor(EmbeddingExtractor):
    """
    A class for extracting embeddings from single-cell or bulk RNA-seq data using pre-trained models.
    """
    
    # def __init__(self, 
    #              task_name='deepcdr',
    #              input_type='singlecell',
    #              output_type='cell',
    #              pool_type='all',
    #              tgthighres='t4',
    #              pre_normalized='F',
    #              demo=False,
    #              version='ce',
    #              model_path='None',
    #              ckpt_name='01B-resolution',
    #              gene_index_path='./OS_scRNA_gene_index.19264.tsv'):
    
    def __init__(self, params):
        super().__init__(params)
        self.log = get_logger()
        self.log.info(f'scFoundation ({self.params})')
        """
        Initialize the scfoundationExtractor.
        
        Parameters:
        -----------
        task_name : str
            Task name for the embedding extraction
        input_type : str
            Input type: 'singlecell' or 'bulk'
        output_type : str
            Output type: 'cell', 'gene', 'gene_batch', or 'gene_expression'
        pool_type : str
            Pooling type for cell embedding: 'all' or 'max'
        tgthighres : str
            Targeted high resolution (starts with 't', 'f', or 'a')
        pre_normalized : str
            Normalization status: 'F', 'T', or 'A'
        demo : bool
            If True, only process 10 samples for demo
        version : str
            Model version: 'ce', 'rde', or 'noversion'
        model_path : str
            Path to pre-trained model
        ckpt_name : str
            Checkpoint name
        gene_index_path : str
            Path to gene index file
        """
        
        self.task_name=  self.params.get('task_name', 'deepcdr') 
        self.input_type= self.params.get('input_type', 'singlecell') 
        self.output_type= self.params.get('output_type', 'cell') 
        self.pool_type= self.params.get('pool_type', 'all') 
        self.tgthighres=self.params.get('tgthighres', 't4') 
        self.pre_normalized=self.params.get('pre_normalized', 'F') 
        self.demo=self.params.get('demo', False) 
        self.version=self.params.get('version', 'ce')
        self.model_path=self.params.get('model_path', 'None') 
        self.ckpt_name=self.params.get('ckpt_name', '01B-resolution') 
        self.gene_index_path=self.params.get('gene_index_path', './OS_scRNA_gene_index.19264.tsv') 
        
        
        print(f'scFoundation ({self.params})')
#         self.task_name = task_name
#         self.input_type = input_type
#         self.output_type = output_type
#         self.pool_type = pool_type
#         self.tgthighres = tgthighres
#         self.pre_normalized = pre_normalized
#         self.demo = demo
#         self.version = version
#         self.model_path = model_path
#         self.ckpt_name = ckpt_name
        
        # Load gene list
        self.gene_list = self._load_gene_list(self.gene_index_path)
        
        # Set random seeds
        # self._set_random_seeds()
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(self.version)
        # Load model
        self.pretrainmodel, self.pretrainconfig = self._load_model()
        
    def _load_gene_list(self, gene_index_path):
        """Load the gene list from the index file."""
        gene_list_df = pd.read_csv(gene_index_path, header=0, delimiter='\t')
        return list(gene_list_df['gene_name'])
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _load_model(self):
        """Load the pre-trained model."""
        if self.version == 'noversion':
            ckpt_path = self.model_path
            key = None
        else:
            # ckpt_path = './models/models.ckpt'
            ckpt_path = self.model_path
            if self.output_type == 'cell':
                if self.version == 'ce':
                    key = 'cell'
                elif self.version == 'rde':
                    key = 'rde'
                else:
                    raise ValueError('No version found')
            elif self.output_type in ['gene', 'gene_batch', 'gene_expression']:
                key = 'gene'
            else:
                raise ValueError('output_type must be one of cell, gene, gene_batch, gene_expression')
        
        pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, key)
        pretrainmodel.eval()
        return pretrainmodel, pretrainconfig
    

    
    def _load_data(self, data_path):
        """
        Load data from various file formats.
        
        Parameters:
        -----------
        data_path : str
            Path to the data file
            
        Returns:
        --------
        pd.DataFrame
            Loaded gene expression data
        """
        if data_path.endswith('.npz'):
            gexpr_feature = scipy.sparse.load_npz(data_path)
            gexpr_feature = pd.DataFrame(gexpr_feature.toarray())
        elif data_path.endswith('.h5ad'):
            gexpr_feature = sc.read_h5ad(data_path)
            idx = gexpr_feature.obs_names.tolist()
            try:
                col = gexpr_feature.var.gene_name.tolist()
            except:
                col = gexpr_feature.var_names.tolist()
            if issparse(gexpr_feature.X):
                gexpr_feature = gexpr_feature.X.toarray()
            else:
                gexpr_feature = gexpr_feature.X
            gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
        elif data_path.endswith('.npy'):
            gexpr_feature = np.load(data_path)
            gexpr_feature = pd.DataFrame(gexpr_feature)
        else:
            gexpr_feature = pd.read_csv(data_path, index_col=0)
        
        return gexpr_feature
    
    def _main_gene_selection(self, X_df, gene_list):
        """
        Rebuild the input data to select target genes.
        
        Parameters:
        -----------
        X_df : pd.DataFrame
            Input gene expression data
        gene_list : list
            List of target genes
            
        Returns:
        --------
        tuple
            (processed_dataframe, to_fill_columns, var_info)
        """
        to_fill_columns = list(set(gene_list) - set(X_df.columns))
        padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                                  columns=to_fill_columns, 
                                  index=X_df.index)
        X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                            index=X_df.index, 
                            columns=list(X_df.columns) + list(padding_df.columns))
        X_df = X_df[gene_list]
        
        var = pd.DataFrame(index=X_df.columns)
        var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
        return X_df, to_fill_columns, var
    
    def _preprocess_data(self, gexpr_feature):
        """
        Preprocess the gene expression data.
        
        Parameters:
        -----------
        gexpr_feature : pd.DataFrame
            Raw gene expression data
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed gene expression data
        """
        # Ensure we have at least 19264 genes
        if gexpr_feature.shape[1] < 19264:
            print('Converting gene feature to 19264 dimensions')
            gexpr_feature, to_fill_columns, var = self._main_gene_selection(gexpr_feature, self.gene_list)
            assert gexpr_feature.shape[1] >= 19264
        
        # Normalize bulk data if needed
        if (self.pre_normalized == 'F') and (self.input_type == 'bulk'):
            adata = sc.AnnData(gexpr_feature)
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            gexpr_feature = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        
        # Demo mode: only process 10 samples
        if self.demo:
            gexpr_feature = gexpr_feature.iloc[:10, :]
        
        print(f"Data shape: {gexpr_feature.shape}")
        return gexpr_feature
    
    def _process_bulk_data(self, gexpr_feature, i):
        """Process bulk RNA-seq data for a single sample."""
        if self.pre_normalized == 'T':
            totalcount = gexpr_feature.iloc[i, :].sum()
        elif self.pre_normalized == 'F':
            totalcount = np.log10(gexpr_feature.iloc[i, :].sum())
        else:
            raise ValueError('pre_normalized must be T or F')
        
        tmpdata = (gexpr_feature.iloc[i, :]).tolist()
        pretrain_gene_x = torch.tensor(tmpdata + [totalcount, totalcount]).unsqueeze(0).cuda()
        data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
        
        return pretrain_gene_x, data_gene_ids
    
    def _process_singlecell_data(self, gexpr_feature, i):
        """Process single-cell RNA-seq data for a single sample."""
        # Pre-normalization
        if self.pre_normalized == 'F':
            tmpdata = (np.log1p(gexpr_feature.iloc[i, :] / (gexpr_feature.iloc[i, :].sum()) * 1e4)).tolist()
        elif self.pre_normalized == 'T':
            tmpdata = (gexpr_feature.iloc[i, :]).tolist()
        elif self.pre_normalized == 'A':
            tmpdata = (gexpr_feature.iloc[i, :-1]).tolist()
        else:
            raise ValueError('pre_normalized must be T, F or A')
        
        if self.pre_normalized == 'A':
            totalcount = gexpr_feature.iloc[i, -1]
        else:
            totalcount = gexpr_feature.iloc[i, :].sum()
        
        # Select resolution
        if self.tgthighres[0] == 'f':
            pretrain_gene_x = torch.tensor(tmpdata + [np.log10(totalcount * float(self.tgthighres[1:])), np.log10(totalcount)]).unsqueeze(0).cuda()
        elif self.tgthighres[0] == 'a':
            pretrain_gene_x = torch.tensor(tmpdata + [np.log10(totalcount) + float(self.tgthighres[1:]), np.log10(totalcount)]).unsqueeze(0).cuda()
        elif self.tgthighres[0] == 't':
            pretrain_gene_x = torch.tensor(tmpdata + [float(self.tgthighres[1:]), np.log10(totalcount)]).unsqueeze(0).cuda()
        else:
            raise ValueError('tgthighres must start with f, a or t')
        
        data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
        
        return pretrain_gene_x, data_gene_ids
    
    def _extract_cell_embedding(self, pretrain_gene_x, data_gene_ids, value_labels):
        """Extract cell embeddings."""
        # Check if we have any valid values
        if not value_labels.any():
            print("Warning: No valid values found, returning zero embedding")
            # Return a zero embedding with appropriate size
            # You may need to adjust this size based on your model's output dimension
            return np.zeros((1, 1024))  # Adjust size as needed
        
        # First gather the data
        x, x_padding = gatherData(pretrain_gene_x, value_labels, self.pretrainconfig['pad_token_id'])
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.pretrainconfig['pad_token_id'])
        
        # Token embedding
        x = self.pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pretrainmodel.pos_emb(position_gene_ids)
        x += position_emb
        geneemb = self.pretrainmodel.encoder(x, x_padding)
        
        geneemb1 = geneemb[:, -1, :]
        geneemb2 = geneemb[:, -2, :]
        geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
        geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
        
        if self.pool_type == 'all':
            geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
        elif self.pool_type == 'max':
            geneembmerge, _ = torch.max(geneemb, dim=1)
        else:
            raise ValueError('pool_type must be all or max')
        
        return geneembmerge.detach().cpu().numpy()
    
    def _extract_gene_embedding(self, pretrain_gene_x):
        """Extract gene embeddings."""
        self.pretrainmodel.to_final = None
        encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(), pretrain_gene_x.float(), self.pretrainconfig)
        
        out = self.pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                                        encoder_position_gene_ids=encoder_position_gene_ids,
                                        encoder_labels=encoder_labels,
                                        decoder_data=decoder_data,
                                        mask_gene_name=False,
                                        mask_labels=None,
                                        decoder_position_gene_ids=decoder_position_gene_ids,
                                        decoder_data_padding_labels=decoder_data_padding)
        
        out = out[:, :19264, :].contiguous()
        return out.detach().cpu().numpy()
    
    def _extract_gene_batch_embedding(self, batchcontainer):
        """Extract gene embeddings in batch mode."""
        self.pretrainmodel.to_final = None
        encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(batchcontainer, batchcontainer, self.pretrainconfig)
        
        out = self.pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                                        encoder_position_gene_ids=encoder_position_gene_ids,
                                        encoder_labels=encoder_labels,
                                        decoder_data=decoder_data,
                                        mask_gene_name=False,
                                        mask_labels=None,
                                        decoder_position_gene_ids=decoder_position_gene_ids,
                                        decoder_data_padding_labels=decoder_data_padding)
        
        return out[:, :19264, :].contiguous().detach().cpu().numpy()
    
    def _extract_gene_expression_embedding(self, pretrain_gene_x):
        """Extract gene expression embeddings."""
        encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(), pretrain_gene_x.float(), self.pretrainconfig)
        
        out = self.pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                                        encoder_position_gene_ids=encoder_position_gene_ids,
                                        encoder_labels=encoder_labels,
                                        decoder_data=decoder_data,
                                        mask_gene_name=False,
                                        mask_labels=None,
                                        decoder_position_gene_ids=decoder_position_gene_ids,
                                        decoder_data_padding_labels=decoder_data_padding)
        
        out = out[:, :19264].contiguous()
        return out.detach().cpu().numpy()
    
    # def extract_embedding(self, data_path, save_path=None):
    def extract_embedding(self, gexpr_feature, save_path=None):
        
        """
        Extract embeddings from the input data.
        
        Parameters:
        -----------
        data_path : str
            Path to the input data file
        save_path : str, optional
            Path to save the embeddings. If None, embeddings are not saved.
            
        Returns:
        --------
        np.ndarray
            Extracted embeddings
        """
        # # Load and preprocess data
        # gexpr_feature = self._load_data(data_path)
        # gexpr_feature = self._preprocess_data(gexpr_feature)
        
        geneexpemb = []
        batchcontainer = []
        
        # Inference
        for i in tqdm(range(gexpr_feature.shape[0])):
            with torch.no_grad():
                # Process data based on input type
                if self.input_type == 'bulk':
                    pretrain_gene_x, data_gene_ids = self._process_bulk_data(gexpr_feature, i)
                elif self.input_type == 'singlecell':
                    pretrain_gene_x, data_gene_ids = self._process_singlecell_data(gexpr_feature, i)
                else:
                    raise ValueError('input_type must be bulk or singlecell')
                
                value_labels = pretrain_gene_x > 0
                x, x_padding = gatherData(pretrain_gene_x, value_labels, self.pretrainconfig['pad_token_id'])
                
                # Extract embeddings based on output type
                if self.output_type == 'cell':
                    embedding = self._extract_cell_embedding(pretrain_gene_x, data_gene_ids, value_labels)
                    geneexpemb.append(embedding)
                
                elif self.output_type == 'gene':
                    embedding = self._extract_gene_embedding(pretrain_gene_x)
                    geneexpemb.append(embedding)
                
                elif self.output_type == 'gene_batch':
                    batchcontainer.append(pretrain_gene_x.float())
                    if len(batchcontainer) == gexpr_feature.shape[0]:
                        batchcontainer = torch.concat(batchcontainer, axis=0)
                        geneexpemb = self._extract_gene_batch_embedding(batchcontainer)
                    else:
                        continue
                
                elif self.output_type == 'gene_expression':
                    embedding = self._extract_gene_expression_embedding(pretrain_gene_x)
                    geneexpemb.append(embedding)
                
                else:
                    raise ValueError('output_type must be cell, gene, gene_batch, or gene_expression')
        
        # Process final embeddings
        if self.output_type != 'gene_batch':
            geneexpemb = np.squeeze(np.array(geneexpemb))
        
        print(f"Embedding shape: {geneexpemb.shape}")
        
        # Save embeddings if save_path is provided
        if save_path is not None:
            strname = os.path.join(save_path, 
                                 f"{self.task_name}_{self.ckpt_name}_{self.input_type}_{self.output_type}_embedding_{self.tgthighres}_resolution.npy")
            print(f'Saving at {strname}')
            np.save(strname, geneexpemb)
        
        return geneexpemb

    def fit_transform(self, scf_data_loader):
        
        self.data_loader = scf_data_loader
        gexpr_feature = self.data_loader.adata
        
        idx = gexpr_feature.obs_names.tolist()
        try:
            col = gexpr_feature.var.gene_name.tolist()
        except:
            col = gexpr_feature.var_names.tolist()
        if issparse(gexpr_feature.X):
            gexpr_feature = gexpr_feature.X.toarray()
        else:
            gexpr_feature = gexpr_feature.X
        gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
            
        
        
        if not hasattr(self, 'pretrainmodel'):
            self.pretrainmodel, self.pretrainconfig = self._load_model()
            
        if not hasattr(self, 'data_prepared'):
            gexpr_feature = self._preprocess_data(gexpr_feature)
            self.data_prepared = True
            
        cell_embeddings = self.extract_embedding(gexpr_feature)
        self.data_loader.adata.obsm['X_scfoundation'] = cell_embeddings
            
        return cell_embeddings
    
    def load_cashed(self, scf_data_loader, data_path):
        self.data_loader = scf_data_loader
        gexpr_feature = sc.read_h5ad(data_path)
        self.data_loader.adata = gexpr_feature
        return 

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Drug_response_pre')
    parser.add_argument('--task_name', type=str, default='deepcdr', help='task name')
    parser.add_argument('--input_type', type=str, default='singlecell', choices=['singlecell', 'bulk'], help='input type; default: singlecell')
    parser.add_argument('--output_type', type=str, default='cell', choices=['cell', 'gene', 'gene_batch', 'gene_expression'], help='cell or gene embedding; default: cell')
    parser.add_argument('--pool_type', type=str, default='all', choices=['all', 'max'], help='pooling type of cell embedding; default: all only valid for output_type=cell')
    parser.add_argument('--tgthighres', type=str, default='t4', help='the targeted high resolution')
    parser.add_argument('--data_path', type=str, default='./', help='input data path')
    parser.add_argument('--save_path', type=str, default='./', help='save path')
    parser.add_argument('--pre_normalized', type=str, default='F', choices=['F', 'T', 'A'], help='if normalized before input')
    parser.add_argument('--demo', action='store_true', default=False, help='if demo, only infer 10 samples')
    parser.add_argument('--version', type=str, default='ce', help='model version')
    parser.add_argument('--model_path', type=str, default='None', help='pre-trained model path')
    parser.add_argument('--ckpt_name', type=str, default='01B-resolution', help='checkpoint name')
    
    args = parser.parse_args()
    
    # Create extractor instance
    extractor = scfoundationExtractor(
        task_name=args.task_name,
        input_type=args.input_type,
        output_type=args.output_type,
        pool_type=args.pool_type,
        tgthighres=args.tgthighres,
        pre_normalized=args.pre_normalized,
        demo=args.demo,
        version=args.version,
        model_path=args.model_path,
        ckpt_name=args.ckpt_name
    )
    
    # Extract embeddings
    embeddings = extractor.extract_embedding(args.data_path, args.save_path)
    
    return embeddings


if __name__ == '__main__':
    # main() 
    pass
