from qubogen.qubo import *
from .utils import *
import numpy as np


def test_qubo_number_partition():
    s = np.array([3, 1, 1, 2, 2, 1])
    sample, _ = qbsolv_from_ndarray(qubo_number_partition(s))
    assert s[sample[0] == 1].sum() == 5
    assert s[sample[0] == 0].sum() == 5


def test_qubo_max_cut():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    g = Graph(edges=edges, n_nodes=5)
    q = qubo_max_cut(g)
    assert q.tolist() == [[-2, 1, 1, 0, 0],
                          [1, -2, 0, 1, 0],
                          [1, 0, -3, 1, 1],
                          [0, 1, 1, -3, 1],
                          [0, 0, 1, 1, -2]]
    sample, _ = qbsolv_from_ndarray(q)
    assert is_contain(sample, np.array([0, 1, 1, 0, 0]))


def test_qubo_mvc():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    g = Graph(edges=edges, n_nodes=5)
    q = qubo_mvc(g, penalty=8.)
    sample, _ = qbsolv_from_ndarray(q)
    assert is_contain(sample, np.array([0, 1, 1, 0, 1]))


def test_qubo_wmvc():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    g = Graph(edges=edges, n_nodes=5, nodes=np.ones(5))
    q = qubo_wmvc(g, penalty=8.)
    sample, _ = qbsolv_from_ndarray(q)
    assert is_contain(sample, np.array([0, 1, 1, 0, 1]))


def test_qubo_set_pack():
    a = np.array([[1, 0, 1, 1],
                  [1, 1, 0, 0]])
    w = np.ones(4)
    q = qubo_set_pack(a, w, penalty=6.)
    assert q.tolist() == [[-1, 3, 3, 3],
                          [3, -1, 0, 0],
                          [3, 0, -1, 3],
                          [3, 0, 3, -1]]
    assert is_contain(qbsolv_from_ndarray(q)[0], np.array([0, 1, 1, 0]))


def test_qubo_max2sat():
    l = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 2],
                  [0, 2], [1, 2], [1, 3], [1, 2], [1, 2],
                  [2, 3], [2, 3]])
    s = np.array([[True, True], [True, False], [False, True], [False, False],
                  [False, True], [False, False], [True, False], [True, True],
                  [False, True], [False, False], [True, True], [False, False]])
    q = qubo_max2sat(Clauses(l, s))
    assert q.tolist() == [[1, 0, 0, 0],
                          [0, 0, -0.5, 0.5],
                          [0, -0.5, 0, 1],
                          [0, 0.5, 1, -2]]
    sample, _ = qbsolv_from_ndarray(q)
    assert is_contain(sample, np.array([0, 0, 0, 1]))


def test_qubo_spp():
    c = np.array([3, 2, 1, 1, 3, 2])
    a = np.array([[1, 0, 1, 0, 0, 1],
                  [0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 1, 1, 0],
                  [1, 1, 0, 1, 0, 1]])

    q = qubo_spp(c, a)
    assert q.tolist() == [[-17, 10, 10, 10, 0, 20],
                          [10, -18, 10, 10, 10, 20],
                          [10, 10, -29, 10, 20, 20],
                          [10, 10, 10, -19, 10, 10],
                          [0, 10, 20, 10, -17, 10],
                          [20, 20, 20, 10, 10, -28]]
    sample, _ = qbsolv_from_ndarray(q)
    assert is_contain(sample, np.array([1, 0, 0, 0, 1, 0]))


def test_qubo_graph_coloring():
    edges = [[0, 1], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [3, 4]]
    n_nodes, n_color = 5, 3
    g = Graph(edges=edges, n_nodes=n_nodes)
    ans = np.zeros(n_nodes * n_color)
    ans[[1, 3, 8, 10, 14]] = 1
    q = qubo_graph_coloring(g, n_color, penalty=4.)
    assert is_contain(qbsolv_from_ndarray(q)[0], ans)


def test_qubo_general01():
    c = np.array([6, 4, 8, 5, 5])
    a = np.array([[2, 2, 4, 3, 2],
                  [1, 2, 2, 1, 2],
                  [3, 3, 2, 4, 4]])
    b = np.array([7, 4, 5])
    s = np.array([-1, 0, 1])
    q = qubo_general01(c, a, b, s)
    sample, _ = qbsolv_from_ndarray(q)
    assert is_contain(sample[:, :5], np.array([1, 0, 0, 1, 1]))


def test_qubo_qap():
    flow = [[0, 5, 2],
            [5, 0, 3],
            [2, 3, 0]]
    distance = [[0, 8, 15],
                [8, 0, 13],
                [15, 13, 0]]
    q = qubo_qap(flow, distance, penalty=200.)
    sample, _ = qbsolv_from_ndarray(q)
    assert is_contain(sample, np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]))


def test_qubo_qkp():
    value = np.array([[2, 4, 3, 5],
                      [4, 5, 1, 3],
                      [3, 1, 2, 2],
                      [5, 3, 2, 4]])
    a = np.array([8, 6, 5, 3])
    b = 16
    q = qubo_qkp(value, a, b, penalty=10.)
    sample, _ = qbsolv_from_ndarray(q)
    n = len(a)
    assert is_contain(sample[:, :n], np.array([1, 0, 1, 1]))
