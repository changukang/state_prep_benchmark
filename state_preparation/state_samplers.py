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


def random_core(rank_vector: Sequence[int], seed: int = None):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=rank_vector)  # i.i.d. N(0,1)
    G /= np.linalg.norm(G.ravel())
    return G


def get_random_state_by_rank_vector(
    qubit_partition: Sequence[int], rank_vector: Sequence[int], seed: int = None
):
    num_qubit = sum(qubit_partition)
    sampled_core = random_core(rank_vector, seed)
    sv_builder = np.zeros(shape=(2**num_qubit,), dtype=np.complex128)

    for core_idx in np.ndindex(*rank_vector):
        elt = sampled_core[core_idx]

        to_kron = [
            cirq.one_hot(
                index=basis_idx,
                shape=(2 ** (qubit_partition[part_idx]),),
                dtype=np.complex128,
            )
            for part_idx, basis_idx in enumerate(core_idx)
        ]
        basis_term_on_elt = cirq.kron(*to_kron, shape_len=1)
        sv_builder += elt * basis_term_on_elt

    rand_part_unitaries = [
        cirq.testing.random_unitary(dim=2**part_qubit_num, random_state=seed + idx)
        for idx, part_qubit_num in enumerate(qubit_partition)
    ]
    unitary_to_mult = cirq.kron(*rand_part_unitaries, shape_len=2)
    sv_builder = unitary_to_mult @ sv_builder

    cirq.validate_normalized_state_vector(sv_builder, qid_shape=(2**num_qubit,))

    return sv_builder
