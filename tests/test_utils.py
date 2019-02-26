import numpy as np
from dwave_qbsolv import QBSolv
from typing import Set, Tuple, Container, Dict, Sequence
from qubogen.utils import *






def test_to_dict():
    q = np.array([[1, 2],
                  [2, 4]])
    assert to_dict(q) == {(0, 0): 1, (0, 1): 2, (1, 0): 2, (1, 1): 4}
