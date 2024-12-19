from dataclasses import dataclass
import numpy as np
import networkx as nx
from .model import Model

@dataclass
class GDUBNMFARDConfig():
    """Docstring for GDUBNMFARDConfig. """
    eps: float = 2.223e-16
    matrix_seed: int = 1
    tolerance: float = 1e-5
    alpha: float = 0.5
    a: int = 5
    b: int = 2
    initial_K: int = None

class GDUBNMFARD(Model):
    """Docstring for GDUBNMFARD. """
    config = GDUBNMFARDConfig()
    def __init__(self, data: "Data", time: int, config: "Config") -> None:
        """TODO: to be defined. """
        self._data = data
        self._time = time
        config.model_params(self.config)
        self.alpha = self.config.alpha if time > 0 else 1

        adj_matrix = nx.to_numpy_array(data.get_graph(time), nodelist=sorted(data.get_graph(time).nodes))
        adj_matrix = np.matrix(adj_matrix)
        attr_matrix = data.get_attributes(time)
        num_k = self.config.initial_K if self.config.initial_K else adj_matrix.shape[0] // 2
        mat_w_t1, mat_h_t1, mat_g_t1 = data.get_past_data(time, num_k)
        if time != 0:
            num_k = mat_w_t1.shape[1]

        np.random.seed(seed=self.config.matrix_seed)
        self._vars = {
                "V": adj_matrix,
                "F": attr_matrix,
                "W_1": mat_w_t1,
                "H_1": mat_h_t1,
                "G_1": mat_g_t1,
                "W": np.asmatrix(np.random.rand(adj_matrix.shape[0], num_k)),
                "H": np.asmatrix(np.random.rand(num_k, adj_matrix.shape[1])),
                "G": np.asmatrix(np.random.rand(attr_matrix.shape[0], num_k)),
                "B": np.diag(np.ones(num_k))
                }
        self._div = np.linalg.norm(self._vars["V"] - self._vars["W"] * self._vars["H"])
        print(self.config)

    def update(self) -> None:
        """TODO: Docstring for update.
        :returns: TODO

        """
        self._div = np.linalg.norm(self._vars["V"] - self._vars["W"]*self._vars["H"])
        self._vars["H"] = self.upd_h()
        self._vars["W"] = self.upd_w()
        self._vars["G"] = self.upd_g()

        self._vars["B"] = (self.N + self._vars["F"].shape[0]/2 + \
                self.config.a - 1) * np.identity(self.K) / \
                np.array((np.multiply(self._vars["W"], self._vars["W"]).sum(axis=0) + \
                np.multiply(self._vars["G"], self._vars["G"]).sum(axis=0) + \
                np.multiply(self._vars["H"].T, self._vars["H"].T).sum(axis=0)) / 2 + self.config.b)

    def upd_w(self) -> "numpy.matrix":
        """TODO: Docstring for upd_w.
        :returns: TODO

        """
        return (np.multiply(self.alpha * self._vars["W"], \
                (self._vars["V"] / (self._vars["W"] * self._vars["H"] + self.config.eps)) * \
                self._vars["H"].T) + (1 - self.alpha) * self._vars["W_1"]) / \
                (self.alpha * np.ones(self._vars["V"].shape) * self._vars["H"].T + \
                (1 - self.alpha) * np.ones(self._vars["W"].shape) + \
                self.alpha * self._vars["W"] * self._vars["B"] + self.config.eps)

    def upd_h(self) -> "numpy.matrix":
        """TODO: Docstring for upd_h.
        :returns: TODO

        """
        return (np.multiply(self.alpha * self._vars["H"], \
                self._vars["W"].T * (self._vars["V"] / (self._vars["W"] * self._vars["H"] + \
                self.config.eps)) + self._vars["G"].T * (self._vars["F"] / \
                (self._vars["G"] * self._vars["H"] + self.config.eps))) + \
                (1 - self.alpha) * self._vars["H_1"]) / \
                (self.alpha * self._vars["W"].T * np.ones(self._vars["V"].shape) + \
                self.alpha * self._vars["G"].T * np.ones(self._vars["F"].shape) \
                + (1 - self.alpha) * np.ones(self._vars["H"].shape) + \
                self.alpha * self._vars["B"] * self._vars["H"] + self.config.eps)

    def upd_g(self) -> "numpy.matrix":
        """TODO: Docstring for upd_g.
        :returns: TODO

        """
        return (np.multiply(self.alpha * self._vars["G"], \
                (self._vars["F"] / (self._vars["G"] * self._vars["H"] + self.config.eps)) * \
                self._vars["H"].T) + (1 - self.alpha) * self._vars["G_1"]) / \
                (self.alpha * np.ones(self._vars["F"].shape) * self._vars["H"].T + \
                (1 - self.alpha) * np.ones(self._vars["G"].shape) + \
                self.alpha * self._vars["G"] * self._vars["B"] + self.config.eps)

    def convergence(self) -> bool:
        """TODO: Docstring for convergence.
        :returns: TODO

        """
        return np.abs(self.diff) < self.config.tolerance

    def comm(self, method: str = "W", threshold: float = 0.5, overlapping: bool = False) -> "numpy.ndarray":
        M = self._vars["W"] if method == "W" else self._vars["H"].T
        if overlapping:
            M /= M.sum(axis=1)
            values = np.argwhere((M == np.amax(M, axis=1)) | (M > threshold))
            mapping = {idx: val for idx, val in enumerate(sorted(self._data.get_graph(self._time).nodes))}
            # print(np.vectorize(mapping.get)(values[:,0]))
            values[:,0] = np.vectorize(mapping.get)(values[:,0])
            # print(values)
            return values
        return np.array(np.argmax(M, axis=1).T)[0]

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
        return "GDUBNMFARD"

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
        self._data.set_past_data(self._time, self._vars["W"], self._vars["H"], self._vars["G"])
