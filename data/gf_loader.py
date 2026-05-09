import os
import pickle
from os.path import join
from data.data_loader import H5ADLoader


def get_gene_name_ensembl_map(dict_dir):

    ENSEMBL_DICTIONARY_FILE = join(dict_dir, 'gene_name_id_dict.pkl')

    def invert_dict(dict_obj):
        return {v: k for k, v in dict_obj.items()}

    with open(ENSEMBL_DICTIONARY_FILE, "rb") as f:
        gene_to_ensembl_dict  = pickle.load(f)
        ensembl_to_gene_dict = invert_dict(gene_to_ensembl_dict)

    return ensembl_to_gene_dict, gene_to_ensembl_dict

class GFLoader(H5ADLoader):
    """Geneformer-oriented loader; ``map_ensembl`` / ``prepare_data`` live on ``H5ADLoader``."""

    def __init__(self, params):
        super().__init__(params)
        self.log.info(f'GFLoader {params}')
