from utils.logs_ import get_logger
logger  =get_logger()
class EmbeddingExtractor:
    def __init__(self, params):
        self.method = params['method']
        self.params = params.get('params', {})

    @staticmethod
    def validate_config(params):
        assert 'method' in params, "Missing required parameter: 'method'"

    def fit_transform(self, data):
        logger.info(f"Extracting embeddings using {self.method} with {self.params}")
        return f"embedding_{data}"