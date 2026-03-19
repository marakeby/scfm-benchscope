import scanpy as sc
import logging

from features.extractor import EmbeddingExtractor

logger = logging.getLogger('ml_logger')


class PCAEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self, params):
        super().__init__(params)
        logger.info(f'PCAEmbeddingExtractor ({self.params})')
        self.n_components = self.params.get('n_components', 2)
        self.hvg_params = self.params.get('HVG', None)

    @staticmethod
    def validate_config(params):
        assert 'params' in params and 'n_components' in params['params'], "Missing 'n_components' in parameters"

    def fit_transform(self, data_loader):
        data = data_loader.adata
        if self.hvg_params:
            self.n_top_genes = self.hvg_params.get('n_top_genes', 2000)
            self.flavor = self.hvg_params.get('flavor', 'seurat')
            logger.info(f"Selecting top {self.n_top_genes} highly variable genes...")
            sc.pp.highly_variable_genes(data, n_top_genes=self.n_top_genes, flavor=self.flavor , subset=True)

        logger.info("Scaling the data...")
        sc.pp.scale(data, max_value=10)

        logger.info(f"Running PCA with {self.n_components } components...")
        sc.tl.pca(data, n_comps=self.n_components )

        return data.obsm['X_pca'].copy()
