from dataclasses import dataclass
import numpy as np
import networkx as nx
from .model import Model

@dataclass
class SBNMFConfig():
    """Docstring for SBNMFConfig. """
    eps: float = 2.223e-16
    tolerance: float = 1e-5
    matrix_seed: int = 1
    a: int = 5
    b: int = 2
    initial_K: int = None

class SBNMF(Model):
    """Docstring for SBNMF. """
    config = SBNMFConfig()
    def __init__(self, data: "Data", time: int, config: "Config") -> None:
        """TODO: to be defined. """
        self._data = data
        config.model_params(self.config)

        adj_matrix = nx.to_numpy_array(data.get_graph(time))
        adj_matrix = np.matrix(adj_matrix)
        num_k = self.config.initial_K if self.config.initial_K else adj_matrix.shape[0] // 2

        np.random.seed(seed=self.config.matrix_seed)
        self._vars = {
                "V": adj_matrix,
                "W": np.asmatrix(np.random.rand(adj_matrix.shape[0], num_k)),
                "H": np.asmatrix(np.random.rand(num_k, adj_matrix.shape[1])),
                "B": np.diag(np.ones(num_k))
                }
        self._div = np.linalg.norm(self._vars["V"] - self._vars["W"]*self._vars["H"])
        print(self.config)

    def update(self) -> None:
        """TODO: Docstring for update.
        :returns: TODO

        """
        self._div = np.linalg.norm(self._vars["V"] - self._vars["W"] * self._vars["H"])
        self._vars["H"] = self.upd_h()
        self._vars["W"] = self.upd_w()
        self._vars["B"] = (self.N + self.config.a - 1) * np.identity(self.K) / \
                np.array((np.multiply(self._vars["W"], self._vars["W"]).sum(axis=0) \
                + np.multiply(self._vars["H"].T, self._vars["H"].T).sum(axis=0)) / 2 + \
                self.config.b)

    def upd_h(self) -> "numpy.matrix":
        """TODO: Docstring for upd_h.
        :returns: TODO

        """
        return np.multiply(self._vars["H"] / \
                (self._vars["W"].T * np.ones(self._vars["V"].shape) + \
                self._vars["B"] * self._vars["H"] + self.config.eps), \
                self._vars["W"].T * (self._vars["V"] / ((self._vars["W"] * self._vars["H"]) + \
                self.config.eps)))

    def upd_w(self) -> "numpy.matrix":
        """TODO: Docstring for upd_w.
        :returns: TODO

        """
        return np.multiply(self._vars["W"] / \
                (np.ones(self._vars["V"].shape) * self._vars["H"].T + \
                self._vars["W"] * self._vars["B"] + self.config.eps), \
                (self._vars["V"] / ((self._vars["W"] * self._vars["H"]) \
                + self.config.eps)) * self._vars["H"].T)

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
        return "SBNMF"

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
