from typing import List, Optional

import cirq
import numpy as np

from state_preparation.utils import is_orthogonal


def get_random_state(num_qubit: int, seed: Optional[int] = None) -> np.ndarray:
    return cirq.testing.random_superposition(dim=2**num_qubit, random_state=seed)


def get_random_sparse_state(
    num_qubit: int, sparsity: int, seed: Optional[int] = None
) -> np.ndarray:
    if not (0 < sparsity <= 2**num_qubit):
        raise ValueError("sparsity must be in the range (0, 2**num_qubit]")

    rng = np.random.default_rng(seed)
    non_zero_terms = rng.choice(2**num_qubit, size=sparsity, replace=False)
    random_complex = rng.random(sparsity) + 1j * rng.random(sparsity)

    sv = np.zeros(2**num_qubit, dtype=np.complex128)
    sv[non_zero_terms] = random_complex
    sv /= np.linalg.norm(sv)

    cirq.validate_normalized_state_vector(sv, qid_shape=(2**num_qubit,))
    return sv


def get_random_basis_state_vectors(
    num_qubit: int, num_basis: int, seed: int = None
) -> List[np.ndarray]:
    rand_unitary = cirq.testing.random_unitary(dim=2**num_qubit, random_state=seed)
    ret = [rand_unitary[:, i] for i in range(num_basis)]

    for r in ret:
        cirq.validate_normalized_state_vector(r, qid_shape=(2**num_qubit))

    assert is_orthogonal(ret)

    return ret
