import math
import random
from itertools import combinations
from typing import Any, Dict, List, Set

import cirq
import numpy as np

from .abstract import BenchmarkStateVector
from .dicke_util import dicke_U_n_k


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

    def sample_parameters(cls, n: int, seed: int) -> Dict[str, Any]:
        return {}

    def __call__(self, n: int):
        sv = np.zeros(shape=(2**n,))
        assert n % 2 == 0, "n must be even"

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

    def get_known_circuit(self, n: int) -> cirq.Circuit:

        raise NotImplementedError(
            "State preparation before dicke-state application needs to be reconsidered."
        )

        def get_bindary_diff_X_strings(
            bin1: str, bin2: str, qbits: List[cirq.LineQubit]
        ) -> List[cirq.GateOperation]:
            assert len(bin1) == len(bin2), "Binary strings must be of the same length"
            assert len(bin1) == len(
                qbits
            ), "Number of qubits must match the length of binary strings"
            ret = []
            for elt1, elt2 in zip(bin1, bin2):
                if elt1 != elt2:
                    ret.append(cirq.X(qbits[bin1.index(elt1)]))
            return ret

        def int_to_bin_str(x: int, length: int) -> str:
            return bin(x)[2:].zfill(length)

        def get_pivotal_binary(hamming_weight: int, num_qubit: int) -> str:
            return "0" * (num_qubit - hamming_weight) + "1" * hamming_weight

        assert n % 2 == 0, "n must be even"

        qc = cirq.Circuit()
        qbits = cirq.LineQubit.range(n)
        qc.append([cirq.H(qbits[i]) for i in range(n // 2)])
        qc.append([cirq.CX(qbits[i], qbits[i + n // 2]) for i in range(n // 2)])

        for i in range(2 ** (n // 2)):
            hamming_weight = bin(i).count("1")
            curr_binary = int_to_bin_str(i, n // 2)
            pivotal = get_pivotal_binary(hamming_weight, n // 2)
            if curr_binary != pivotal:
                qc.append(
                    [
                        x_string.controlled_by(
                            *qbits[: n // 2],
                            control_values=[int(x) for x in curr_binary],
                        )
                        for x_string in get_bindary_diff_X_strings(
                            curr_binary, pivotal, qbits[n // 2 :]
                        )
                    ]
                )

        dicke_part = cirq.Circuit(dicke_U_n_k(n // 2, n // 2 - 1))
        dicke_part = dicke_part.transform_qubits(lambda q: q + n // 2)
        qc.append(dicke_part)
        return qc


class HeadZeroSuperposition(BenchmarkStateVector):
    # ref : https://quantumcomputing.stackexchange.com/q/4545/15277

    def sample_parameters(cls, n: int, seed: int) -> Dict[str, Any]:
        return {}

    def __call__(self, n: int, m: int):

        req_num_qubit = math.ceil(np.log2(m + 1))
        assert n >= req_num_qubit

        sv_building = np.zeros(shape=2**n, dtype=np.complex128)
        sv_building += cirq.one_hot(index=0, shape=(2**n,), dtype=np.complex128)
        for i in range(1, m + 1):
            sv_building += (1 / np.sqrt(m)) * cirq.one_hot(
                index=i, shape=(2**n,), dtype=np.complex128
            )

        sv_building /= np.sqrt(2)

        cirq.validate_normalized_state_vector(sv_building, qid_shape=(2**n))

        return sv_building


class SubsetSuperposition(BenchmarkStateVector):
    # https://quantumcomputing.stackexchange.com/questions/27864/creating-a-uniform-superposition-of-a-subset-of-basis-states/39868
    def sample_parameters(cls, n: int, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        subset_size = rng.randint(1, n)
        subset = set(random.sample(range(n), subset_size))
        return {"subset": subset}

    def __call__(self, n: int, subset: Set[int]):
        assert all(0 <= i < 2**n - 1 for i in subset)
        sv = np.zeros(2**n, dtype=np.complex128)
        for idx in subset:
            sv[idx] = 1
        sv /= np.linalg.norm(sv)
        cirq.validate_normalized_state_vector(sv, qid_shape=(2**n,))
        return sv


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


def hamming_weight(x: int) -> int:
    return bin(x).count("1")


class PreBHWState(BenchmarkStateVector):

    def __call__(self, n: int):
        assert n % 2 == 0
        sv = np.zeros(shape=(2**n,), dtype=np.complex128)
        for i in range(2 ** int(n / 2)):
            left_term = cirq.one_hot(
                index=i, shape=(2 ** (n // 2),), dtype=np.complex128
            )
            right_term = cirq.one_hot(
                index=sum(2**i for i in range(hamming_weight(i))),
                shape=(2 ** (n // 2),),
                dtype=np.complex128,
            )
            sv += cirq.kron(left_term, right_term).reshape(-1)

        sv /= np.linalg.norm(sv)
        return sv
