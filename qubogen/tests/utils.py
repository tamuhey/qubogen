import numpy as np
from typing import Tuple
from qubogen.utils import *
from dwave_qbsolv import QBSolv


def qbsolv_from_ndarray(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns lowest energy samples from dwave qbsolv.(for test util)"""
    res = QBSolv().sample_qubo(to_dict(q)).record
    i = res.energy == res.energy.min()
    return res.sample[i], res.energy[i]


def is_contain(x: np.ndarray, y: np.ndarray) -> bool:
    """If y \in x"""
    assert len(x.shape) - 1 == len(y.shape)
    return any(np.array_equal(xi, y) for xi in x)
