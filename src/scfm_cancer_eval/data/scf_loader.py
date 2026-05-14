# from scgpt.preprocess import Preprocessor
from scfm_cancer_eval.data.data_loader import H5ADLoader
from scipy.sparse import issparse


class scfLoader(H5ADLoader):
    
    def __init__(self, params):
        super().__init__(params)
        self.prepared = False
        
#     def _main_gene_selection(self, X_df, gene_list):
#         """
#         Rebuild the input data to select target genes.
        
#         Parameters:
#         -----------
#         X_df : pd.DataFrame
#             Input gene expression data
#         gene_list : list
#             List of target genes
            
#         Returns:
#         --------
#         tuple
#             (processed_dataframe, to_fill_columns, var_info)
#         """
#         to_fill_columns = list(set(gene_list) - set(X_df.columns))
#         padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
#                                   columns=to_fill_columns, 
#                                   index=X_df.index)
#         X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
#                             index=X_df.index, 
#                             columns=list(X_df.columns) + list(padding_df.columns))
#         X_df = X_df[gene_list]
        
#         var = pd.DataFrame(index=X_df.columns)
#         var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
#         return X_df, to_fill_columns, var
    
    def prepare_data(self):
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
        gexpr_feature = self.adata
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
        self.prepared = True
        return gexpr_feature