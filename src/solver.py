import time as tm
import csv
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community import modularity
from cdlib import algorithms, NodeClustering, evaluation
import numpy as np
from .models import ModelFactory

class Solver():
    """Docstring for Solver. """

    def __init__(self, data: "Data", config: "Config") -> None:
        """TODO: to be defined.

        :strategy: TODO

        """
        self._data = data
        self._config = config

    def solve(self) -> None:
        """TODO: Docstring for solve.
        :returns: TODO

        """
        for time in range(self._data.get_snapshots()):
            self.solve_timestep(time)

    def solve_timestep(self, time : float) -> None:
        """TODO: Docstring for solve_timestep.

        :time: TODO
        :returns: TODO

        """
        stime = tm.time()
        model = ModelFactory.create_model(self._config.model)(self._data, time, self._config)

        iteration = 0
        for iteration in tqdm(range(self._config.n_iters), leave=False):
            model.update()
            if model.convergence() and iteration > self._config.min_iters:
                break
        model.update_data()
        solve_time = tm.time() - stime
        print(f'\nTimestep: {time}')
        print(f'Break after {iteration} iterations')
        print(f'Delta: {abs(model.diff)}')
        #print(f'NMI: {self.nmi(model.comm("H"), self._data.get_ground_truth(time))}')
        print(f'NMI: {self.o2nmi(self._data.get_graph(time), model.comm("H", overlapping=True, threshold=0.3), self._data.get_overlapping_gt(time))}')
        self.save_results(model, solve_time, time, iteration)

    @staticmethod
    def nmi(comm: "numpy.ndarray", ground_truth: list) -> float:
        """TODO: Docstring for nmi.

        :comm: TODO
        :ground_truth: TODO
        :returns: TODO

        """
        return normalized_mutual_info_score(ground_truth, comm)

    @staticmethod
    def onmi(graph, comm: "numpy.ndarray", ground_truth: list, method: str = 'LFK') -> float:
        """TODO: Docstring for onmi.

        :comm: TODO
        :ground_truth: TODO
        :returns: TODO

        """
        community_nodes = {
            com: [node for node, com_val in enumerate(ground_truth) if com_val == com]
            for com in np.unique(ground_truth)
        }

        overlapping_comms = {
            com: [node for node, com_val in comm if com_val == com]
            for com in np.unique(comm[:,1])
        }

        gt = NodeClustering(communities={frozenset(community_nodes[c]) for c in community_nodes}, graph=graph)
        alg = NodeClustering(communities={frozenset(overlapping_comms[c]) for c in overlapping_comms}, graph=graph)

        if method == 'LFK':
            onmi = evaluation.overlapping_normalized_mutual_information_LFK(alg, gt).score
        else:
            onmi = evaluation.overlapping_normalized_mutual_information_MGH(alg, gt).score
        return onmi

    @staticmethod
    def o2nmi(graph, comm: "numpy.ndarray", ground_truth: list, method: str = 'LFK') -> float:
        """TODO: Docstring for o2nmi.

        :comm: TODO
        :ground_truth: TODO
        :returns: TODO

        """
        overlapping_comms = {
            com: [node for node, com_val in comm if com_val == com]
            for com in np.unique(comm[:,1])
        }

        gt = NodeClustering(communities={frozenset(ground_truth[c]) for c in ground_truth}, graph=graph)
        alg = NodeClustering(communities={frozenset(overlapping_comms[c]) for c in overlapping_comms}, graph=graph)
        #print(gt.communities)
        #print(alg.communities)

        if method == 'LFK':
            onmi = evaluation.overlapping_normalized_mutual_information_LFK(alg, gt).score
        else:
            onmi = evaluation.overlapping_normalized_mutual_information_MGH(alg, gt).score
        return onmi

    @staticmethod
    def modularity(graph, comm: "numpy.ndarray") -> float:
        """TODO: Docstring for modularity.

        :graph: TODO
        :comm: TODO
        :returns: TODO

        """
        mod = {i: [] for i in set(comm)}
        for idx, com in zip(graph, comm):
            mod[com].append(idx)
        return modularity(graph, mod.values())

    @staticmethod
    def omodularity(graph, comm: "numpy.ndarray") -> float:
        """TODO: Docstring for onmi.

        :graph: TODO
        :comm: TODO
        :returns: TODO

        """
        overlapping_comms = {
            com: [node for node, com_val in comm if com_val == com]
            for com in np.unique(comm[:,1])
        }

        alg = NodeClustering(communities={frozenset(overlapping_comms[c]) for c in overlapping_comms}, graph=graph)

        mod_overlap = evaluation.modularity_overlap(graph, alg).score
        return mod_overlap

    @staticmethod
    def density(graph, comms):
        """TODO: Docstring for density.

        :graph: TODO
        :comms: TODO
        :returns: TODO

        """
        comms_dict = {i: [] for i in set(comms)}
        for idx, com in zip(graph, comms):
            comms_dict[com].append(idx)
        density = 0
        for comm in comms_dict:
            density += len([(i,j) for (i,j) in graph.edges \
                    if i in comms_dict[comm] and j in comms_dict[comm]])
        density /= len(graph.edges)
        return density

    @staticmethod
    def entropy(graph, comms, attr, nattr=72, sizes=None, verbose=False):
        """TODO: Docstring for density.

        :graph: TODO
        :comms: TODO
        :attr: TODO
        :nattr: TODO
        :sizes: TODO
        :verbose: TODO
        :returns: TODO

        """
        nattr = nattr if nattr else attr.sum(axis=0).max()
        sizes = sizes if sizes else [ attr.shape[0] // nattr ] * nattr
        #print(nattr, sizes)
        alfa = 1/nattr

        print('len Attr: ', sizes) if verbose else None
        print('nattr: ', nattr) if verbose else None
        print('attr shape: ', attr.shape) if verbose else None
        print('alfa: ', alfa) if verbose else None

        # assert attr.sum(axis=0).max() == nattr, 'The sum of columns doesnt match'
        assert sum(sizes) == attr.shape[0], 'The sizes dont match'

        comms_dict = {i: [] for i in set(comms)}
        for idx, com in enumerate(comms):
            comms_dict[com].append(idx)

        entropy = 0
        for idx, m_size in enumerate(sizes):
            for m in range(m_size):
                for k in comms_dict:
                    card = len(comms_dict[k]) / len(graph.nodes)
                    for val in [0, 1]:
                        p = (attr[sum(sizes[:idx]) + m, comms_dict[k]] == val).sum()/len(comms_dict[k])
                        if p != 0:
                            entropy += card*(-p*np.log2(p))*alfa
                    print(f"""\nk: {k}\nm: {m}\nidx m: {sum(sizes[:idx])
                        + m}\np: {p}\nNk: {len(comms_dict[k])}\nN: {
                        len(graph.nodes)}""")  if verbose else None
        return entropy

    def save_results(self, model : "Model", solve_time: float, time: int, iterations: int) -> None:
        """TODO: Docstring for save_results.

        :model: TODO
        :solve_time: TODO
        :returns: TODO

        """
        with open(self._config.csvfile, 'a', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for thr in [0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
                writer.writerow([
                    self._config.data,
                    self.o2nmi(self._data.get_graph(time), model.comm("W", overlapping=True, threshold=thr), self._data.get_overlapping_gt(time)),
                    self.o2nmi(self._data.get_graph(time), model.comm("H", overlapping=True, threshold=thr), self._data.get_overlapping_gt(time)),
                    self.o2nmi(self._data.get_graph(time), model.comm("W", overlapping=True, threshold=thr), self._data.get_overlapping_gt(time), method = 'MGH'),
                    self.o2nmi(self._data.get_graph(time), model.comm("H", overlapping=True, threshold=thr), self._data.get_overlapping_gt(time), method = 'MGH'),
                    #self.nmi(model.comm("W"), self._data.get_ground_truth(time)),
                    #self.nmi(model.comm("H"), self._data.get_ground_truth(time)),
                    model.config.matrix_seed,
                    time,
                    iterations,
                    model.name,
                    self.modularity(self._data.get_graph(time), model.comm("W")),
                    self.modularity(self._data.get_graph(time), model.comm("H")),
                    self.omodularity(self._data.get_graph(time), model.comm("W", overlapping=True, threshold=thr)),
                    self.omodularity(self._data.get_graph(time), model.comm("H", overlapping=True, threshold=thr)),
                    self.density(self._data.get_graph(time), model.comm("W")),
                    self.density(self._data.get_graph(time), model.comm("H")),
                    self.entropy(self._data.get_graph(time), model.comm("W"), \
                            self._data.get_attributes(time)),
                    self.entropy(self._data.get_graph(time), model.comm("H"), \
                            self._data.get_attributes(time)),
                    len(set(model.comm("W"))),
                    len(set(model.comm("H"))),
                    model.N,
                    model.K,
                    self._config.alpha if self._config.model in ["DBNMFARD", "DUBNMFARD", "GDUBNMFARD", "DXBNMFARD", "AADBNMFARD", "CUSTOM"] else None,
                    solve_time,
                    tm.ctime(),
                    thr,
                    model.comm("W", overlapping=True, threshold=thr).shape[0] - model.N,
                    model.comm("H", overlapping=True, threshold=thr).shape[0] - model.N
                ])
        np.savetxt(self._config.data/f'comms_Wt{time}.csv', model.comm("W"),fmt='%d')
        np.savetxt(self._config.data/f'comms_Ht{time}.csv', model.comm("H"),fmt='%d')
        np.save(self._config.data/f'matrix_Wt{time}.npy', model._vars["W"])
        np.save(self._config.data/f'matrix_Ht{time}.npy', model._vars["H"])
        np.save(self._config.data/f'matrix_ATTRt{time}.npy', self._data.get_attributes(time))
