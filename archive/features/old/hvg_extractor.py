import logging
import scanpy as sc
from features.pca_extractor import EmbeddingExtractor

logger = logging.getLogger('ml_logger')

class HVGExtractor(EmbeddingExtractor):
    def __init__(self, params):
        super().__init__(params)
        logger.info(f'HVGExtractor ({self.params})')
        self.n_top_genes = self.params.get('n_top_genes', 2000)
        self.flavor = self.params.get('flavor', 'seurat')
        self.batch = self.params.get('batch', 'batch')

    @staticmethod
    def validate_config(params):
        assert 'n_top_genes' in params['params'], "Missing 'n_top_genes' in parameters"
        assert 'flavor' in params['flavor'], "Missing 'flavor' in parameters"

    def fit_transform(self, data_loader):
        data =data_loader.adata
        sc.pp.highly_variable_genes(data, flavor=self.flavor, subset=False, batch_key=self.batch, n_top_genes=self.n_top_genes)
        data.obsm["X_hvg"] = data.X[:, data.var.highly_variable.values]
        return data.obsm["X_hvg"].copy()
