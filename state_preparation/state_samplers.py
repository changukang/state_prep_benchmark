from multiprocessing import Value
from turtle import shearfactor
from typing import List, Optional, Sequence

import cirq
import numpy as np
from sympy import rad

from state_preparation.utils import is_orthogonal


def get_random_state(num_qubit: int, seed: Optional[int] = None) -> np.ndarray:
    return cirq.testing.random_superposition(dim=2**num_qubit, random_state=seed)


def get_random_sparse_state(
    num_qubit: int,
    sparsity: int,
    seed: Optional[int] = None,
    complex: bool = True,
    uniform: bool = False,
) -> np.ndarray:
    if not (0 < sparsity <= 2**num_qubit):
        raise ValueError("sparsity must be in the range (0, 2**num_qubit]")

    rng = np.random.default_rng(seed)
    non_zero_terms = rng.choice(2**num_qubit, size=sparsity, replace=False)
    if uniform:
        amps = np.ones(sparsity)
    else:
        amps = (
            rng.random(sparsity) + 1j * rng.random(sparsity)
            if complex
            else rng.random(sparsity)
        )

    sv = np.zeros(2**num_qubit, dtype=np.complex128)
    sv[non_zero_terms] = amps
    sv /= np.linalg.norm(sv)

    cirq.validate_normalized_state_vector(sv, qid_shape=(2**num_qubit,))
    return sv


def get_random_basis_state_vectors(
    num_qubit: int, num_basis: int, seed: Optional[int] = None
) -> List[np.ndarray]:
    rand_unitary = cirq.testing.random_unitary(dim=2**num_qubit, random_state=seed)
    ret = [rand_unitary[:, i] for i in range(num_basis)]

    for r in ret:
        cirq.validate_normalized_state_vector(r, qid_shape=(2**num_qubit,))

    assert is_orthogonal(ret)

    return ret


def get_random_state_by_rank_vector(
    num_qubits_per_partition: Sequence[int], ranks: Sequence[int], seed: int = None
) -> np.ndarray:
    """
    Generalized random Tucker tensor generator.

    shape: tuple of ints, e.g. (n1, n2, ..., nk)
    ranks: tuple of ints, e.g. (d1, d2, ..., dk), same length as shape
    seed: optional random seed
    """

    if len(ranks) == 2 and len(set(ranks)) != 1:
        raise ValueError("For order-2 tensors, ranks must be the same.")

    if len(ranks) == 1:
        raise ValueError("For rank-1 tensors, use get_random_state instead.")

    rng = np.random.default_rng(seed)
    order = len(num_qubits_per_partition)

    if len(ranks) != order:
        raise ValueError("shape and ranks must have the same length")

    # core tensor
    G = rng.standard_normal(size=ranks)
    # random orthogonal factor matrices
    Us = []
    shapes = [2**n for n in num_qubits_per_partition]
    for n, d in zip(shapes, ranks):
        U, _ = np.linalg.qr(rng.standard_normal((n, d)))
        Us.append(U)

    # apply mode-k products
    T = G
    for mode, U in enumerate(Us):
        # tensordot along the 'mode' dimension of core with columns of U
        T = np.tensordot(T, U, axes=(mode, 1))
        # move new axis to correct position
        T = np.moveaxis(T, -1, mode)

    state_vector = T.reshape(-1)  # 1D flatten
    state_vector /= np.linalg.norm(state_vector)

    cirq.validate_normalized_state_vector(
        state_vector, qid_shape=(2 ** sum(num_qubits_per_partition),)
    )

    return state_vector
