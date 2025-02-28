from typing import Generator, Optional

import cirq
import numpy as np


def get_random_state(
    num_qubit: int, seed: Optional[int] = None
) -> np.ndarray:
    return cirq.testing.random_superposition(dim=2**num_qubit, random_state=seed)


def get_random_sparse_state(
    num_qubit: int, sparsity: int, seed: Optional[int] = None
) -> np.ndarray:
    if 2**num_qubit < sparsity : 
        raise ValueError("Sparsity stads for number of non-zero terms in the state. " 
                         "Hence. must be sparsity < 2**num_qubit.")
    