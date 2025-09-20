import itertools
import random
from typing import List, Tuple

import cirq
import numpy as np

from state_preparation.state_samplers import (
    get_random_basis_state_vectors,
    get_random_sparse_state,
    get_random_state,
    get_random_state_by_rank_vector,
)


def test_random_state():
    for seed in range(10):
        get_random_state(num_qubit=5, seed=seed)


def test_random_sparse_statea():
    for seed in range(10):
        sv = get_random_sparse_state(num_qubit=5, sparsity=3, seed=seed)
        assert (sv != 0).sum() == 3


def test_random_basis_state_vectors():
    for seed in range(10):
        get_random_basis_state_vectors(num_qubit=2, num_basis=2, seed=seed)


def test_random_state_by_rank_vector():
    def partial_trace(state_vector: np.ndarray, sub_system: List[int]) -> np.ndarray:
        dims = int(np.log2(state_vector.size))
        state_vector = state_vector.reshape((2,) * dims)
        rho = np.outer(state_vector, np.conj(state_vector)).reshape(
            state_vector.shape * 2
        )
        ret_shape = (2 ** len(sub_system),)

        keep_rho = cirq.partial_trace(rho, sub_system).reshape(
            (np.prod(ret_shape),) * 2
        )
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

    def random_int_3_tuple(
        low: int = 0, high: int = 10, seed: int = None
    ) -> tuple[int, int, int]:
        rng = random.Random(seed)
        return tuple(rng.randint(low, high) for _ in range(3))

    def partition_to_indices(partition: tuple[int, ...]):
        result = []
        current = 0
        for size in partition:
            if size == 1:
                result.append([current])
            else:
                result.append(list(range(current, current + size)))
            current += size
        return result

    for seed in range(5):
        qubit_partiton = random_int_3_tuple(1, 4, seed=seed)

        for rank_vector in itertools.product(
            *[list(range(2**i)) for i in qubit_partiton]
        ):
            rank_vector = list(r + 1 for r in rank_vector)
            if any(r <= 2 for r in rank_vector):
                continue
            sampled_sv = get_random_state_by_rank_vector(
                list(qubit_partiton), rank_vector, seed=42
            )
            rank_vec_res = schmidt_rank_vector(
                sampled_sv, sub_systems=partition_to_indices(qubit_partiton)
            )
            assert rank_vec_res == tuple(rank_vector)
