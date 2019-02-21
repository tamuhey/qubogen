import numpy as np
from typing import Tuple, Dict


def to_dict(q: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Returns q matrix in the form of dict from np.ndarray
    
    The dict can be directly passed to dwave_qbsolv.QBSolv
    """
    assert len(q.shape) == 2 and q.shape[0] == q.shape[1]
    return dict(zip(zip(*np.nonzero(q)), q[q != 0]))
