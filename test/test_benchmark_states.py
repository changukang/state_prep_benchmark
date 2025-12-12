import cirq
import numpy as np

from state_preparation.benchmark.states import (BalancedHammingWeight,
                                                GraphPermutationSuperposition,
                                                HeadZeroSuperposition,
                                                SubsetSuperposition)

ZERO_KET = np.array([1, 0])
ONE_KET = np.array([0, 1])


def test_balanced_hamming_weight():
    balanced_hw = BalancedHammingWeight()

    n_2_sv = balanced_hw(n=2)

    # n: 2 -> 00, 11
    assert np.isclose(
        n_2_sv,
        np.sqrt(1 / 2) * (np.kron(ZERO_KET, ZERO_KET) + np.kron(ONE_KET, ONE_KET)),
    ).all()

    n_4_sv = balanced_hw(n=4)
    # n: 4 -> 0000, 0101, 0110, 1001, 1010, 1111
    assert np.isclose(
        n_4_sv,
        np.sqrt(1 / 6)
        * (
            cirq.kron(ZERO_KET, ZERO_KET, ZERO_KET, ZERO_KET)
            + cirq.kron(ZERO_KET, ONE_KET, ZERO_KET, ONE_KET)
            + cirq.kron(ZERO_KET, ONE_KET, ONE_KET, ZERO_KET)
            + cirq.kron(ONE_KET, ZERO_KET, ZERO_KET, ONE_KET)
            + cirq.kron(ONE_KET, ZERO_KET, ONE_KET, ZERO_KET)
            + cirq.kron(ONE_KET, ONE_KET, ONE_KET, ONE_KET)
        ),
    ).all()


def test_head_zero_superposition():
    balanced_hw = HeadZeroSuperposition()

    sv = balanced_hw(2, 3)
    assert np.isclose(
        sv,
        (1 / np.sqrt(2))
        * (
            cirq.kron(ZERO_KET, ZERO_KET)
            + (1 / np.sqrt(3))
            * (
                cirq.kron(ZERO_KET, ONE_KET)
                + cirq.kron(ONE_KET, ZERO_KET)
                + cirq.kron(ONE_KET, ONE_KET)
            )
        ),
    ).all()


def test_subset_superposition():
    subset_superposition = SubsetSuperposition()

    sv = subset_superposition(n=3, subset={0, 2})
    assert np.isclose(sv, (1 / np.sqrt(2)) * np.array([1, 0, 1, 0, 0, 0, 0, 0])).all()

    sv = subset_superposition(n=3, subset={1, 3, 5})
    assert np.isclose(sv, (1 / np.sqrt(3)) * np.array([0, 1, 0, 1, 0, 1, 0, 0])).all()


def test_graph_permutation_state():
    graph_permutaiton_superpose = GraphPermutationSuperposition()
    graph_edges = [(0, 1), (1, 2)]  # |101> in quantum state
    sv = graph_permutaiton_superpose(n=3, graph_edges=graph_edges)

    # S_3 = { (0, 1, 2), (0, 2, 1), (1, 0, 2),
    #       (1, 2, 0), (2, 0, 1), (2, 1, 0) }
    # each corresponds to
    # |101>, |110>, |011>, |011> , |110>, |101>
    # Occupation count:
    # |101> : 2
    # |110> : 2
    # |011> : 2
    # hence target state is
    # normalized of |101> + |110> + |011> (5, 6, 3)
    sv_targ = np.array([0, 0, 0, 1, 0, 1, 1, 0], dtype=np.complex128)
    sv_targ /= np.linalg.norm(sv_targ)

    assert np.isclose(sv, sv_targ).all(), f"Expected {sv_targ}, got {sv}"
