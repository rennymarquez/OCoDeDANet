from .dbnmfard import DBNMFARD
from .dubnmfard import DUBNMFARD
from .gdbnmfard import GDBNMFARD
from .gdubnmfard import GDUBNMFARD
from .custom import CUSTOM

class ModelFactory():
    """Docstring for ModelStrategy. """

    @staticmethod
    def create_model(model: str) -> "Model":
        if model == 'DBNMFARD':
            return DBNMFARD
        if model == 'DUBNMFARD':
            return DUBNMFARD
        if model == 'GDBNMFARD':
            return GDBNMFARD
        if model == 'GDUBNMFARD':
            return GDUBNMFARD
        return CUSTOM
