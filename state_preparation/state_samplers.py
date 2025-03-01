from typing import Optional

import cirq
import numpy as np
from numpy.random import random_sample


def get_random_state(num_qubit: int, seed: Optional[int] = None) -> np.ndarray:
    return cirq.testing.random_superposition(dim=2**num_qubit, random_state=seed)


def get_random_sparse_state(
    num_qubit: int, sparsity: int, seed: Optional[int] = None
) -> np.ndarray:
    if 2**num_qubit < sparsity:
        raise ValueError(
            "Sparsity stads for number of non-zero terms in the state. "
            "Hence. must be sparsity < 2**num_qubit."
        )
    np.random.seed(seed)
    non_zero_terms = np.random.choice(
        list(range(2**num_qubit)), size=sparsity, replace=False
    ).tolist()
    assert len(non_zero_terms) == sparsity

    random_complex = random_sample((len(non_zero_terms),)) + 1j * random_sample(
        (len(non_zero_terms),)
    )
    sv_building = np.zeros(shape=(2**num_qubit,))
    for non_zero_term, amplitude in zip(non_zero_terms, random_complex):
        sv_building[non_zero_term] = amplitude
    sv = sv_building / np.linalg.norm(sv_building)
    cirq.validate_normalized_state_vector(sv, qid_shape=(2**num_qubit,))
    return sv
