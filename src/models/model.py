import abc

class Model(abc.ABC):
    """Docstring for ModelStrategy. """

    @abc.abstractmethod
    def update(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def convergence(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def diff(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def comm(self, method: str) -> "numpy.ndarray":
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def N(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def K(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def update_data(self) -> None:
        raise NotImplementedError
