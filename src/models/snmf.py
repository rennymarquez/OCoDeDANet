from dataclasses import dataclass
import numpy as np
import networkx as nx
from .model import Model

@dataclass
class SNMFConfig():
    """Docstring for SNMFConfig. """
    matrix_seed: int = 1
    tolerance: float = 1e-5
    eps: float = 2.223e-16
    initial_K: int = None

class SNMF(Model):
    """Docstring for SNMF. """
    config = SNMFConfig()
    def __init__(self, data: "Data", time: int, config: "Config") -> None:
        """TODO: to be defined. """
        self._data = data
        config.model_params(self.config)

        adj_matrix = nx.to_numpy_array(data.get_graph(time))
        adj_matrix = np.matrix(adj_matrix)

        np.random.seed(seed=self.config.matrix_seed)
        self._vars = {
                "V": adj_matrix,
                "W": np.asmatrix(np.random.rand(adj_matrix.shape[0], \
                        self._data.get_number_clusters(time))),
                "H": np.asmatrix(np.random.rand(self._data.get_number_clusters(time), \
                        adj_matrix.shape[1])),
                }

        self._div = np.linalg.norm(self._vars["V"] - self._vars["W"]*self._vars["H"])
        print(self.config)

    def update(self) -> None:
        """TODO: Docstring for update.
        :returns: TODO

        """
        self._div = np.linalg.norm(self._vars["V"] - self._vars["W"]*self._vars["H"])
        self._vars["H"] = self.upd_h()
        self._vars["W"] = self.upd_w()

    def upd_h(self) -> "numpy.matrix":
        """TODO: Docstring for update.
        :returns: TODO

        """
        return np.multiply(self._vars["H"], \
                (self._vars["W"].T * (self._vars["V"] / ((self._vars["W"] * self._vars["H"]) + \
                self.config.eps))) / (self._vars["W"].T * np.ones(self._vars["V"].shape) + \
                self.config.eps))

    def upd_w(self) -> "numpy.matrix":
        """TODO: Docstring for update.
        :returns: TODO

        """
        return np.multiply(self._vars["W"], \
                ((self._vars["V"] / ((self._vars["W"] * self._vars["H"]) + \
                self.config.eps)) * self._vars["H"].T) / \
                ((np.ones(self._vars["V"].shape) * self._vars["H"].T) + self.config.eps))

    def convergence(self) -> bool:
        """TODO: Docstring for convergence.
        :returns: TODO

        """
        return np.abs(self.diff) < self.config.tolerance

    def comm(self, method: str = "W") -> "numpy.ndarray":
        if method == "W":
            return np.array(np.argmax(self._vars["W"] == np.amax(self._vars["W"], axis=1), \
                    axis=1).T)[0]
        return np.array(np.argmax(self._vars["H"].T == np.amax(self._vars["H"].T, axis=1), \
                axis=1).T)[0]

    @property
    def diff(self) -> float:
        """TODO: Docstring for diff.
        :returns: TODO

        """
        return self._div - np.linalg.norm(self._vars["V"] - self._vars["W"]*self._vars["H"])

    @property
    def name(self) -> str:
        """TODO: Docstring for name.
        :returns: TODO

        """
        return "SNMF"

    @property
    def N(self) -> int:
        """TODO: Docstring for N.
        :returns: TODO

        """
        return self._vars["V"].shape[0]

    @property
    def K(self) -> int:
        """TODO: Docstring for K.
        :returns: TODO

        """
        return self._vars["W"].shape[1]

    def update_data(self) -> None:
        """TODO: Docstring for update_data.
        :returns: TODO

        """
        pass
