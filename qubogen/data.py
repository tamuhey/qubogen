from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import networkx as nx
from typing import Collection, Mapping, Sequence


@dataclass
class Graph:
    edges: np.ndarray  # edge is represented as (i, j)
    n_nodes: int
    init_value: Mapping[int, int] = None  # init_value[i] indicates init value of node i
    nodes: np.ndarray = None  # value of nodes. len(nodes) must be n_nodes

    def __post_init__(self):
        if self.nodes is not None:
            assert len(self.nodes) == self.n_nodes
        if self.edges is None:
            return
        if len(self.edges) == 0:
            self.edges = None
            return
        self.edges = np.array(list(self.edges))
        assert self.edges.max() < self.n_nodes
        assert self.edges.shape[1] == 2
        if self.init_value is not None:
            assert all(np.array(list(self.init_value)) < self.n_nodes)

    @classmethod
    def from_networkx(cls, graph: nx.Graph) -> Graph:
        return cls(edges=graph.edges, n_nodes=graph.number_of_nodes())

    def to_networkx_graph(self) -> nx.Graph:
        nxg = nx.Graph()
        nxg.add_nodes_from(range(self.n_nodes))
        nxg.add_edges_from(self.edges)
        return nxg


@dataclass
class Clauses:
    literals: np.ndarray[int]
    signs: np.ndarray[bool]

    def __post_init__(self):
        assert len(self.literals) == len(self.signs)
