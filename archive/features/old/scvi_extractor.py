import scanpy as sc
import logging
import scvi
scvi.settings.seed = 9627

from features.extractor import EmbeddingExtractor
from utils.logs_ import get_logger
logger = get_logger()


class scVIEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self, params):
        super().__init__(params)
        logger.info(f'scVIEmbeddingExtractor ({self.params})')
        self.batch_key = self.params.get('batch_key', None)
        self.n_layers = self.params.get('n_layers', None)
        self.n_latent = self.params.get('n_latent', None)
        self.gene_likelihood = self.params.get('gene_likelihood', None)
        self.max_epochs = self.params.get('max_epochs', None)
        self.layer_name = self.params.get('layer_name', None)
        # self.layer_name = self.params.get('layer_name', None)
        # self.njobs = self.params.get('njobs', 0)
        self.batch_size = self.params.get('batch_size', 16)
        self.hvg_params = self.params.get('hvg_params', False)
        
    @staticmethod
    def validate_config(params):
        assert 'params' in params and params is not None, "Missing 'params' in parameters"
        assert 'batch_key' in params['params'], "Missing 'batch_key' in parameters"
        assert  'n_layers' in params['params'], "Missing 'n_layers' in parameters"
        assert  'n_latent' in params['params'], "Missing 'n_latent' in parameters"
        assert  'gene_likelihood' in params['params'], "Missing 'gene_likelihood' in parameters"

    def fit_transform(self, data_loader):
        data = data_loader.adata

        if self.hvg_params:
            self.n_top_genes = self.hvg_params.get('n_top_genes', 2000)
            self.flavor = self.hvg_params.get('flavor', 'seurat')
            logger.info(f"Selecting top {self.n_top_genes} highly variable genes...")
            sc.pp.highly_variable_genes(data, n_top_genes=self.n_top_genes, flavor=self.flavor , subset=True)

        scvi.model.SCVI.setup_anndata(data, layer=self.layer_name, batch_key=self.batch_key)

        model = scvi.model.SCVI(data, n_layers=self.n_layers, n_latent=self.n_latent, gene_likelihood=self.gene_likelihood)

        # model._data_loader_kwargs["num_workers"] = self.njobs
        
        model.train(self.max_epochs, batch_size = self.batch_size)
        data.obsm["X_scVI"] = model.get_latent_representation()
        # embedding_col = 'X_scVI'

        return data.obsm['X_scVI'].copy()
