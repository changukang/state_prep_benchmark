import itertools
import random
from typing import List, Tuple

import cirq
import numpy as np

from state_preparation.state_samplers import (get_random_basis_state_vectors,
                                              get_random_sparse_state,
                                              get_random_state,
                                              get_random_state_by_rank_vector)


def partial_trace(state_vector: np.ndarray, sub_system: List[int]) -> np.ndarray:
    dims = int(np.log2(state_vector.size))
    state_vector = state_vector.reshape((2,) * dims)
    rho = np.outer(state_vector, np.conj(state_vector)).reshape(state_vector.shape * 2)
    ret_shape = (2 ** len(sub_system),)

    keep_rho = cirq.partial_trace(rho, sub_system).reshape((np.prod(ret_shape),) * 2)
    return keep_rho


def sub_system_rank(state_vector: np.ndarray, sub_system: List[int]):
    eigvals, _ = np.linalg.eigh(partial_trace(state_vector, sub_system))
    return len([e for e in eigvals if not np.isclose(e, 0)])


def schmidt_rank_vector(
    state_vector: np.ndarray, sub_systems: List[List[int]]
) -> Tuple[int, ...]:
    return tuple(
        sub_system_rank(state_vector, sub_system) for sub_system in sub_systems
    )


res = get_random_state_by_rank_vector([2, 2, 2], [4, 3, 4], seed=21313)

print(schmidt_rank_vector(res, [[0, 1], [2, 3], [4, 5]]))
