from .sbnmf import SBNMF
from .subnmf import SUBNMF
from .dbnmfard import DBNMFARD
from .dubnmfard import DUBNMFARD
from .gdubnmfard import GDUBNMFARD
from .custom import CUSTOM

class ModelFactory():
    """Docstring for ModelStrategy. """

    @staticmethod
    def create_model(model: str) -> "Model":
        if model == 'SBNMF':
            return SBNMF
        if model == 'SUBNMF':
            return SUBNMF
        if model == 'DBNMFARD':
            return DBNMFARD
        if model == 'DUBNMFARD':
            return DUBNMFARD
        if model == 'GDUBNMFARD':
            return GDUBNMFARD
        return CUSTOM
