import math
from itertools import combinations
from typing import List

import cirq
import numpy as np

from .abstract import BenchmarkStateVector


class BalancedHammingWeight(BenchmarkStateVector):
    # ref : https://mathoverflow.net/questions/301733/how-to-create-a-quantum-algorithm-that-produces-2-n-bit-sequences-with-equal-num
    # authored by : seungmin.jeon@kaist.ac.kr
    """
    Construct state vector for bit measure state (number of ones on the left half and number of ones on the right half are equal)

    Example
    -------
    n: 2 -> 00, 11
    n: 4 -> 0000, 0101, 0110, 1001, 1010, 1111
    n: 6 -> 000000, 001001, 001010, 001100, 010001, 010010, 010100, 100001, 100010, 100100, 011011, 011101, 011110, 101011, 101101, 101110, 110011, 110101, 110110, 111111
    """

    def __call__(self, n: int):
        sv = np.zeros(shape=(2**n,))

        valid_indices = []
        for i in range(2**n):
            bit_string = bin(i)[2:].zfill(n)  # Convert to binary and pad with zeros
            left_half = bit_string[: n // 2]
            right_half = bit_string[n // 2 :]

            if left_half.count("1") == right_half.count("1"):
                valid_indices.append(i)

        norm_factor = 1 / np.sqrt(len(valid_indices))
        for idx in valid_indices:
            sv[idx] = norm_factor

        return sv


class HeadZeroSuperposition(BenchmarkStateVector):
    # ref : https://quantumcomputing.stackexchange.com/q/4545/15277

    def __call__(self, m: int):
        num_qubit = math.ceil(np.log2(m + 1))

        sv_building = np.zeros(shape=2**num_qubit, dtype=np.complex128)
        sv_building += cirq.one_hot(index=0, shape=(2**num_qubit,), dtype=np.complex128)
        for i in range(1, m + 1):
            sv_building += (1 / np.sqrt(m)) * cirq.one_hot(
                index=i, shape=(2**num_qubit,), dtype=np.complex128
            )

        sv_building /= np.sqrt(2)

        cirq.validate_normalized_state_vector(sv_building, qid_shape=(2**num_qubit))

        return sv_building


class Unary(BenchmarkStateVector):
    # ref : https://quantumcomputing.stackexchange.com/q/4545/15277

    def __call__(self, amplitudes: List[float]):
        num_qubit = len(amplitudes)

        sv_building = np.zeros(shape=2**num_qubit, dtype=np.complex128)
        sv_building += cirq.one_hot(index=0, shape=(2**num_qubit,), dtype=np.complex128)
        for i, amp in enumerate(amplitudes):
            sv_building += amp * cirq.one_hot(
                index=2**i, shape=(2**num_qubit,), dtype=np.complex128
            )

        sv_building /= np.linalg.norm(sv_building)

        cirq.validate_normalized_state_vector(sv_building, qid_shape=(2**num_qubit,))

        return sv_building


class Dicke(BenchmarkStateVector):
    # ref : https://quantumcomputing.stackexchange.com/q/4545/15277

    def __call__(self, n: int, k: int):
        sv = np.zeros(2**n, dtype=np.complex128)
        for ones_pos in combinations(range(n), k):
            idx = 0
            for pos in ones_pos:
                idx |= 1 << (n - pos - 1)
            sv[idx] = 1
        sv /= np.linalg.norm(sv)
        cirq.validate_normalized_state_vector(sv, qid_shape=(2**n,))
        return sv
