from re import sub
from typing import Sequence

import cirq
import test

from state_preparation.benchmark.states import SubsetSuperposition
from state_preparation.permutation.types import (
    Cycle,
    DisjointTranspositions,
    Permutation,
    SequentialTranspositions,
    Transposition,
    TranspositionsList,
    transformed_binary_matrix,
    transposition,
)
from state_preparation.state_samplers import get_random_sparse_state
import numpy as np


def test_permutation():
    n = 8
    sigma = Permutation.from_cycles(
        n,
        [
            [
                0,
                1,
                3,
                5,
            ]
        ],
    )
    assert sigma(0) == 1
    assert sigma(1) == 3
    assert sigma(3) == 5
    assert sigma(5) == 0

    assert sigma(7) == 7
    assert sigma(6) == 6
    assert sigma(2) == 2
    assert sigma(4) == 4


def test_cycle():
    sigma = Cycle.from_cycle(8, (2, 4, 6))

    assert sigma(2) == 4
    assert sigma(4) == 6
    assert sigma(6) == 2

    assert sigma(0) == 0
    assert sigma(1) == 1
    assert sigma(3) == 3

    assert sigma(5) == 5
    assert sigma(7) == 7


def test_cycle_decomposition_into_transpositions():
    sigma: Cycle = Cycle.from_cycle(8, (2, 4, 6))
    rho_prime, rho_double_prime = sigma.decompose_into_two_disjoint_transpositions()
    transpositions_to_apply = rho_prime + rho_double_prime

    def apply_transposition(x, tau: Sequence[Transposition]):
        for tau in transpositions_to_apply:
            x = tau(x)
        return x

    assert apply_transposition(2, transpositions_to_apply) == 4
    assert apply_transposition(4, transpositions_to_apply) == 6
    assert apply_transposition(6, transpositions_to_apply) == 2


def test_sequential_transpositions():

    for num_qubit, m in zip([7, 8], [8, 16]):
        sequential_transposes = list()
        for i in range(m):
            sequential_transposes.append(transposition(2**num_qubit, 2 * i, 2 * i + 1))

        s_transposes = SequentialTranspositions(sequential_transposes)
        qbits = cirq.LineQubit.range(num_qubit)
        qc = s_transposes.to_quantum_circuit(qbits)
        for seed in range(20):
            sv = get_random_sparse_state(num_qubit=num_qubit, sparsity=10, seed=seed)
            reordered_sv = s_transposes.apply_to_state(sv)
            res = qc.final_state_vector(initial_state=sv, qubit_order=qbits)
            assert np.isclose(reordered_sv, res, atol=1e-8).all()


def test_disjoint_transpositions():
    rng = np.random.default_rng(seed=42)
    random_permutation = tuple(rng.choice(range(32), size=20, replace=False))
    sigma: Cycle = Cycle.from_cycle(32, random_permutation)
    rho_prime, rho_double_prime = sigma.decompose_into_two_disjoint_transpositions()

    test_disjoint_transposition = DisjointTranspositions(rho_double_prime[:8])
    qc = test_disjoint_transposition.to_quantum_circuit_for_index_mapping(
        qubits=cirq.LineQubit.range(6)
    )
    transfomred = transformed_binary_matrix(
        test_disjoint_transposition.to_binary_matrix(6),
        qc,
        qubits=cirq.LineQubit.range(6),
    )


def test_transposition():
    sigma = Transposition(8, 0, 1)

    assert sigma(0) == 1
    assert sigma(1) == 0

    for i in range(2, 8):
        assert sigma(i) == i
