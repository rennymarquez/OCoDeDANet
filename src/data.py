import numpy as np
import networkx as nx

class Data():
    """Docstring for Data. """

    def __init__(self, path: "pathlib.Path") -> None:
        """TODO: to be defined.

        :path: TODO

        """
        self._path = path
        self._past_matrix = {}
        print(f'Solving data {self._path}')

    def get_idx(self, time: int) -> "numpy.ndarray":
        """TODO: Docstring for get_idx.

        :time: TODO
        :returns: TODO

        """
        path = self.get_path('Data*Index.txt')
        return np.genfromtxt(path, dtype=int, delimiter=',', max_rows=1, skip_header=time)

    def get_graph(self, time: int) -> "networkx.classes.graph.Graph":
        """TODO: Docstring for get_graph.

        :time: TODO
        :returns: TODO

        """
        path = self.get_path(f'Data*t{time}.gml')
        #return nx.read_gml(path, label='id')
        return nx.read_gml(path, destringizer=int)

    def get_attributes(self, time: int) -> "numpy.matrix":
        """TODO: Docstring for get_attributes.

        :time: TODO
        :returns: TODO

        """
        #path = self.get_path(f'GTData*t{time}')
        #path = self.get_path(f'Data*H3Attrt{time}.csv')
        #cls = np.unique(np.concatenate([np.genfromtxt(self.get_path(f'GTData*t{t}'), dtype=int) \
        #        for t in range(self.get_snapshots())]))
        #attr = np.matrix(np.genfromtxt(path, dtype=int))
        #attr = np.matrix(list(map(lambda x:[int(cl in set(x)) \
        #        for cl in cls], np.array(attr.T)))).T
        path = self.get_path(f'Data*H3Attrt{time}.csv')
        #path = self.get_path(f'Data*Attrt{time}.csv')
        attr = np.matrix(np.genfromtxt(path, dtype=int, delimiter=",")).T
        return attr

    def get_number_clusters(self, time: int) -> "numpy.int64":
        """TODO: Docstring for get_number_clusters.

        :time: TODO
        :returns: TODO

        """
        path = self.get_path('KData*.txt')
        return np.genfromtxt(path, dtype=int)[time]

    def get_ground_truth(self, time: int) -> "numpy.ndarray":
        """TODO: Docstring for get_ground_truth.

        :time: TODO
        :returns: TODO

        """
        path = self.get_path(f'GTData*t{time}')
        return np.genfromtxt(path, dtype=int)

    def get_overlapping_gt(self, time: int):
        """TODO: Docstring for get_overlapping_gt.

        :time: TODO
        :returns: TODO

        """
        path = self.get_path(f'OGTData*t{time}')
        with open(path, 'r') as f:
            gt = {comm: list(map(int, line.split())) for comm, line in enumerate(f.readlines())}
        return gt

    def get_snapshots(self) -> int:
        """TODO: Docstring for get_snapshots.
        :returns: TODO

        """
        path = self.get_path('KData*.txt')
        return len(np.genfromtxt(path, dtype=int))

    def get_path(self, regex: str):
        """TODO: Docstring for get_path.

        :regex: TODO
        :returns: TODO

        """
        return list(self._path.glob(regex))[0]

    def get_past_data(self, time: int, n_comms: int) -> tuple:
        """TODO: Docstring for get_past_data.

        :time: TODO
        :K: TODO
        :returns: TODO

        """
        if time == 0:
            adj_matrix = nx.to_numpy_array(self.get_graph(time))
            attr_matrix = self.get_attributes(time)
            w_time_1 = np.asmatrix(np.zeros((adj_matrix.shape[0], n_comms)))
            h_time_1 = np.asmatrix(np.zeros((n_comms, adj_matrix.shape[1])))
            g_time_1 = np.asmatrix(np.zeros((attr_matrix.shape[0], n_comms)))
            return w_time_1, h_time_1, g_time_1

        idx = self.get_idx(time)
        idx_1 = self.get_idx(time - 1)
        matrix_w, matrix_h, matrix_g = self._past_matrix[time - 1]
        mapping = {i: ix for i, ix in map(lambda x: (x, np.where(idx_1 == x)[0][0] \
                if len(np.where(idx_1 == x)[0]) > 0 else -1), idx)}
        w_time_1 = np.append(matrix_w, np.zeros((1, matrix_w.shape[1])), axis = 0 \
                )[[i for i in map(lambda x: mapping[x], idx)]]

        h_time_1 = np.append(matrix_h, np.zeros((matrix_h.shape[0],1)), axis = 1 \
                )[:,[i for i in map(lambda x: mapping[x], idx)]]
        g_time_1 = matrix_g

        return w_time_1, h_time_1, g_time_1

    def set_past_data(self, time: int, w_matrix: "numpy.matrix", h_matrix: "numpy.matrix", g_matrix: "numpy.matrix" = None ) -> None:
        """TODO: Docstring for set_past_data.

        :W: TODO
        :H: TODO
        :time: TODO
        :returns: TODO

        """
        self._past_matrix[time] = (w_matrix, h_matrix, g_matrix)
