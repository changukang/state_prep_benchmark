from typing import List

import cirq
import numpy as np
from cirq import GateOperation, LineQubit


def building_block(n: int, m: int) -> List[GateOperation]:
    # Assume n qubits are given.
    # 'm' represents the offset from the last (n_th_qubit) site.
    # The building block will be applied to:
    # - n_th_qubit
    # - (n - m)-th qubit
    # - (n - m + 1)-th qubit
    # Valid for m > 0
    # For m = 1, the gate will be applied to n_th_qubit and (n - 1)-th qubit.
    assert m > 0

    n_th_qubit = LineQubit(n - 1)  # n_th_qubit is the last qubit
    ret = list()
    ry_gate_angle = 2 * np.arccos(np.sqrt(m / n))
    if m == 1:
        target_qubit = LineQubit(n - 1 - 1)
        ret.append(cirq.CX(target_qubit, n_th_qubit))
        ret.append(
            cirq.Ry(rads=ry_gate_angle).controlled(num_controls=1)(
                n_th_qubit, target_qubit
            )
        )
        ret.append(cirq.CX(target_qubit, n_th_qubit))
    else:
        splitted_qubit = LineQubit(n - m - 1)  # (n-m)-th qubit
        shift_qubit = LineQubit(n - m)  # (n-m+1)-th qubit
        ret.append(cirq.CX(splitted_qubit, n_th_qubit))
        ret.append(
            cirq.Ry(rads=ry_gate_angle).controlled(num_controls=2)(
                n_th_qubit, shift_qubit, splitted_qubit
            )
        )
        ret.append(cirq.CX(splitted_qubit, n_th_qubit))

    return ret


def split_and_cyclice_shift(n: int, k: int) -> List[GateOperation]:
    ret = list()
    for m in range(1, k + 1):
        ret += building_block(n, m)
    return ret


def dicke_U_n_k(n: int, k: int) -> List[cirq.GateOperation]:
    assert n >= k
    if n == 1 and k == 1:
        return list()
    elif n == k:
        return split_and_cyclice_shift(k, k - 1) + dicke_U_n_k(k - 1, k - 1)
    else:
        return split_and_cyclice_shift(n, k) + dicke_U_n_k(n - 1, k)
