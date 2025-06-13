import cirq
import numpy as np

from state_preparation.benchmark.states import (
    BalancedHammingWeight,
    HeadZeroSuperposition,
)

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

    # for m = 3, the result state vector is ()
    sv = balanced_hw(3)
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
