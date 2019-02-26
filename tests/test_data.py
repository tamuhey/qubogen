from qubogen.data import *
import numpy as np


def test_graph():
    edges = {(1, 2), (3, 4)}
    graph = Graph(edges, 5)
    assert graph.edges.shape == (2, 2)
    nxg = graph.to_networkx_graph()
    _graph = Graph.from_networkx(nxg)
    assert np.array_equal(graph.edges, _graph.edges)
    assert graph.n_nodes == _graph.n_nodes
