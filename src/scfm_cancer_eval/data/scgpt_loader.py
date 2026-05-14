from scgpt.preprocess import Preprocessor
from scfm_cancer_eval.data.data_loader import H5ADLoader
from scipy.sparse import issparse
import numpy as np

class scgptLoader(H5ADLoader):
    
    def __init__(self, params):
        super().__init__(params)
        self.prepared = False
    

    def prepare_data(self, n_bins=51, result_binned_key='X_binned' ):
        
        #assume gene names are stored in var.index
        self.adata.var['gene_name'] = self.adata.var.index
        print('min max', np.min(self.adata.X), np.max(self.adata.X))
        preprocessor = Preprocessor(
            #  whether to bin the raw data and to what number of bins
            binning = n_bins, 
            # the key in adata.layers to store the binned data
            result_binned_key = result_binned_key,  
        )

        if issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()
            
        preprocessor(self.adata, batch_key = self.batch_key) 
        self.log.info('processed scGPT dataset')
        self.prepared = True




