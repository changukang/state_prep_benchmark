from typing import Sequence

import cirq
import numpy as np
import pytest

from state_preparation.mcx.mcx_gates import CanonMCXGate, QulinMCXGate
from state_preparation.permutation.types import (
    Cycle,
    DisjointTranspositions,
    Permutation,
    SequentialTranspositions,
    Transposition,
    transposition,
)
from state_preparation.state_samplers import get_random_sparse_state


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


def test_inverse_permutation():
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

    sigma_inv = sigma.inverse()

    assert sigma_inv(1) == 0
    assert sigma_inv(3) == 1
    assert sigma_inv(5) == 3
    assert sigma_inv(0) == 5

    assert sigma_inv(7) == 7
    assert sigma_inv(6) == 6
    assert sigma_inv(2) == 2
    assert sigma_inv(4) == 4


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


def test_transposition():
    sigma = Transposition(8, 0, 1)

    assert sigma(0) == 1
    assert sigma(1) == 0

    for i in range(2, 8):
        assert sigma(i) == i


@pytest.mark.skip(reason="Not Implemented Yet")
def test_sequential_transpositions():

    for num_qubit, m in zip([7, 8], [8, 16]):
        sequential_transposes = list()
        for i in range(m):
            sequential_transposes.append(transposition(2**num_qubit, 2 * i, 2 * i + 1))

        s_transposes = SequentialTranspositions(sequential_transposes)

        qbits = cirq.LineQubit.range(num_qubit)
        qc = s_transposes.to_quantum_circuit(qbits, mcx_gate_type=QulinMCXGate)
        for seed in range(20):
            sv = get_random_sparse_state(num_qubit=num_qubit, sparsity=10, seed=seed)
            reordered_sv = s_transposes.apply_to_state(sv)
            res = qc.final_state_vector(initial_state=sv, qubit_order=qbits)
            assert np.isclose(reordered_sv, res, atol=1e-8).all()


@pytest.mark.skip(reason="Not Implemented Yet")
def test_disjoint_transpositions():
    rng = np.random.default_rng(seed=42)
    random_permutation = tuple(rng.choice(range(32), size=20, replace=False))
    sigma = Cycle.from_cycle(32, random_permutation)
    _, rho_double_prime = sigma.decompose_into_two_disjoint_transpositions()

    disjoint_transposition = DisjointTranspositions(rho_double_prime[:8])

    index_extraction_mapping: Permutation = (
        disjoint_transposition.induced_index_extraction_map()
    )

    for idx, elt in enumerate(disjoint_transposition.elements):
        assert index_extraction_mapping(elt) == idx

    seq_transposes = SequentialTranspositions.from_num_transposes(
        disjoint_transposition.n, disjoint_transposition.m
    )

    f = disjoint_transposition.induced_index_extraction_map()
    g = seq_transposes
    h = f.inverse()

    for elt in disjoint_transposition.elements:
        res = h(g(f(elt)))
        oracle = disjoint_transposition(elt)
        assert res == oracle

    qc = cirq.Circuit()
    qubits = cirq.LineQubit.range(6)
    qc += disjoint_transposition.index_extraction_map_qc(
        qubits, mcx_gate_type=CanonMCXGate
    )
    qc += seq_transposes.to_quantum_circuit(qubits, mcx_gate_type=CanonMCXGate)
    qc += (
        disjoint_transposition.index_extraction_map_qc(
            qubits, mcx_gate_type=CanonMCXGate
        )
        ** -1
    )

    for elt in disjoint_transposition.elements:
        initial_state = np.zeros(64, dtype=np.complex128)
        initial_state[elt] = 1.0
        res = qc.final_state_vector(initial_state=initial_state, qubit_order=qubits)
        oracle = disjoint_transposition(elt)
        for i in range(64):
            if i == oracle:
                assert np.isclose(res[i], 1.0)
            else:
                assert np.isclose(res[i], 0.0)


def test_decompose_permutation_into_two_disjoint_transpositions():
    sparsity = 20
    sv = get_random_sparse_state(num_qubit=6, sparsity=sparsity, seed=2025)
    nonzero_indicies = (np.where(sv > 1e-8)[0]).tolist()
    flag = np.zeros(sparsity, dtype=int)

    perm_building = Permutation.identity(2**6)

    for i in nonzero_indicies:
        if i < sparsity:
            flag[i] = 1

    for q in nonzero_indicies:
        if q >= sparsity:
            k = np.where(flag == 0)[0][0]
            flag[k] = 1
            perm_building = perm_building.compose(Transposition(2**6, k, q))

    permuted = [perm_building(i) for i in range(sparsity)]

    assert sorted(permuted) == sorted(nonzero_indicies)

    qc = perm_building.index_extraction_based_decomposition_qc(
        cirq.LineQubit.range(6), mcx_gate_type=CanonMCXGate
    )

    for i in range(2**6):
        initial_state = np.zeros(64, dtype=np.complex128)
        initial_state[i] = 1.0
        res = qc.final_state_vector(
            initial_state=initial_state, qubit_order=cirq.LineQubit.range(6)
        )
        nonzero = np.where(res > 1e-8)[0]
        oracle = perm_building(i)
        assert len(nonzero) == 1
        assert (nonzero[0] == oracle).all()
