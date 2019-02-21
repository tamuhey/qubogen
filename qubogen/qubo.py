from __future__ import annotations
from typing import Container
import numpy as np
from .utils import *
from .data import *


def qubo_number_partition(number_set: Container[float]) -> np.ndarray:
    """Return Q matrix of number partitioning problem
    
    Math:
        q_{ii} = s_{i}(s_{i}-c) \\
        q_{ij} = q_{ji} = s_{i}s_{j} \\
        c = \sum_i{s_{i}}
    """
    number_set = np.array(list(number_set))
    c = number_set.sum()
    q = np.outer(number_set, number_set)
    np.fill_diagonal(q, number_set * (number_set - c))
    return q


def qubo_max_cut(g: Graph) -> np.ndarray:
    n_nodes = g.n_nodes
    q = np.zeros((n_nodes, n_nodes))
    i, j = g.edges.T
    np.add.at(q, (i, i), -1)
    np.add.at(q, (j, j), -1)
    q[i, j] += 1
    q[j, i] += 1
    return q


def qubo_mvc(g: Graph, penalty: float = 10.):
    """The Minimum Vertex Cover Problem (MVC)
    
    Math:
        y = \sum_{j \in V}x_{j} + P\left(\sum_{(i,j) \in E}\left(1-x_{i}-x_{j}+x_{i}x_{j}\right)\right)
    """
    q = np.diagflat(np.ones(g.n_nodes))
    i, j = g.edges.T
    np.add.at(q, (i, i), -penalty)
    np.add.at(q, (j, j), -penalty)
    q[i, j] += penalty / 2.
    q[j, i] += penalty / 2.
    return q


def qubo_wmvc(g: Graph, penalty: float = 10.):
    """The Minimum Vertex Cover Problem (MVC)
    
    Math:
        y = \sum_{j \in V}w_{j}x_{j} + P\left(\sum_{(i,j) \in E}\left(1-x_{i}-x_{j}+x_{i}x_{j}\right)\right)
    """
    q = np.diagflat(np.ones(g.n_nodes) * g.nodes)
    i, j = g.edges.T
    np.add.at(q, (i, i), - penalty * g.nodes[i])
    np.add.at(q, (j, j), - penalty * g.nodes[j])
    q[i, j] += penalty / 2.
    q[j, i] += penalty / 2.
    return q


def qubo_set_pack(a: np.ndarray, weight: np.ndarray, penalty=8.) -> np.ndarray:
    """Set Packing Problem
    
    Math:
        min\ \sum_{j=1}^n w_{j}x_{j} \\
        st \\
        \sum_{j=1}^n a_{ij}x_{j} \le 1
    """
    assert len(a.shape) == 2
    assert a.shape[-1] == len(weight)
    q = -np.diagflat(weight)
    c = np.einsum("ij,ik->ijk", a, a).astype(np.float) / 2.  # constraint
    c *= (1 - np.eye(a.shape[-1]))[None, ...]  # zeros diag
    q += penalty * c.sum(0)
    return q


def qubo_max2sat(c: Clauses):
    n = c.literals.max() + 1
    q = np.zeros((n, n))
    i, j = c.literals.T
    si, sj = c.signs.T
    np.add.at(q, (i[sj], i[sj]), ((-1) ** si)[sj])
    np.add.at(q, (j[si], j[si]), ((-1) ** sj)[si])

    offdiag = np.zeros_like(q)
    np.add.at(offdiag, (i, j), (-1) ** (si ^ sj) / 2.)
    return offdiag + offdiag.T + q


def qubo_spp(cost: np.ndarray, set_flag: np.ndarray, p=10.):
    """The Set Partitioning Problem (SPP)
    
    Args:
        set_flag: 2-rank array. (i,j) = 1 if element i in set j.
    """
    assert cost.shape[0] == set_flag.shape[1]
    b = np.ones(set_flag.shape[0])
    return np.diagflat(cost - p * 2 * b.dot(set_flag)) + p * np.einsum("ij,ik->jk", set_flag, set_flag)


def qubo_graph_coloring(g: Graph, n_color, penalty=10.) -> np.ndarray:
    """Graph Coloring"""
    n = g.n_nodes
    q = np.zeros((n, n_color, n, n_color))

    # each nodes are colored by one color
    i = range(n)
    q[i, :, i, :] = penalty * (np.ones((n_color, n_color)) - 2 * np.eye(n_color))

    # adjacent nodes are not colored with same color
    i, j = g.edges.T[..., None]
    k = range(n_color)
    q[i, k, j, k] += penalty / 2.
    q[j, k, i, k] += penalty / 2.

    return q.reshape(n * n_color, n * n_color)


def qubo_general01(cost, a, b, sign, penalty=10.):
    """General 0/1 Programming
    
    Args:
        a, b: constraint (ax=b)
        sign: 0 is equality, 1 is >=, -1 is <=
    """
    slack = - sign * b
    slack[sign == 0] = 1.
    slack[sign == 1] += a[sign == 1, :].sum()
    slack = np.ceil(np.log2(slack)).astype(np.int)

    ni = a.shape[0]
    for i, ns in enumerate(slack):
        if ns:
            t = np.zeros((ni, ns))
            t[i] = -sign[i] * 2 ** np.arange(ns)
            a = np.concatenate([a, t], -1)

    return - np.diagflat(np.concatenate([cost, np.zeros(sum(slack))]) + 2 * penalty * b.dot(a)) + penalty * np.einsum(
        "ij,ik->jk", a, a)


def qubo_qap(flow: np.ndarray, distance: np.ndarray, penalty=10.):
    """Quadratic Assignment Problem (QAP)"""
    n = len(flow)
    q = np.einsum("ij,kl->ikjl", flow, distance).astype(np.float)

    i = range(len(q))
    q[i, :, i, :] += penalty
    q[:, i, :, i] += penalty
    q[i, i, i, i] -= 4 * penalty
    return q.reshape(n ** 2, n ** 2)


def qubo_qkp(value: np.ndarray, a: np.ndarray, b: float, penalty=10.) -> np.ndarray:
    """Quadratic Knapsack Problem (QKP)"""
    n = len(value)
    nslack = np.ceil(np.log2(b))
    slack = 2 ** (np.arange(nslack))
    a = np.concatenate([a, slack])
    q = penalty * (np.outer(a, a) - 2 * b * np.diag(a))
    q[:n, :n] -= value
    return q
