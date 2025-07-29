import cirq
from cirq import LineQubit, GateOperation
from typing import List
import numpy as np


def building_block(n: int, l: int) -> List[GateOperation]:
    # Assume n qubits are given.
    # 'l' represents the offset from the last (n_th_qubit) site.
    # The building block will be applied to:
    # - n_th_qubit
    # - (n - l)-th qubit
    # - (n - l + 1)-th qubit
    # Valid for l > 0
    # For l = 1, the gate will be applied to n_th_qubit and (n - 1)-th qubit.
    assert l > 0

    n_th_qubit = LineQubit(n - 1)  # n_th_qubit is the last qubit
    ret = list()
    ry_gate_angle = 2 * np.arccos(np.sqrt(l / n))
    if l == 1:
        target_qubit = LineQubit(n - 1 - 1)
        ret.append(cirq.CX(target_qubit, n_th_qubit))
        ret.append(
            cirq.Ry(rads=ry_gate_angle).controlled(num_controls=1)(
                n_th_qubit, target_qubit
            )
        )
        ret.append(cirq.CX(target_qubit, n_th_qubit))
    else:
        splitted_qubit = LineQubit(n - l - 1)  # (n-l)-th qubit
        shift_qubit = LineQubit(n - l)  # (n-l+1)-th qubit
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
    for l in range(1, k + 1):
        ret += building_block(n, l)
    return ret


def dicke_U_n_k(n: int, k: int) -> List[cirq.GateOperation]:
    assert n >= k
    if n == 1 and k == 1:
        return list()
    elif n == k:
        return split_and_cyclice_shift(k, k - 1) + dicke_U_n_k(k - 1, k - 1)
    else:
        return split_and_cyclice_shift(n, k) + dicke_U_n_k(n - 1, k)


if __name__ == "__main__":
    # n_th_qubit = LineQubit(4)
    # l = 2
    # operations = building_block(n_th_qubit, l)
    # qc = (cirq.Circuit(operations))
    # print(qc)
    # li = [0, 1, 3] if l==1 else [0,1,2,3,7]
    # for i  in li:
    #     in_ =cirq.one_hot(index =i , shape = (2**cirq.num_qubits(qc),), dtype=np.complex128)
    #     print(cirq.dirac_notation(in_))
    #     res=cirq.final_state_vector(qc, initial_state=in_)
    #     print(cirq.dirac_notation(res))
    #     print("----")
    # input()
    qc = cirq.Circuit(dicke_U_n_k(5, 3))
    in_ = cirq.one_hot(index=1, shape=(2 ** cirq.num_qubits(qc),), dtype=np.complex128)
    print(cirq.dirac_notation(in_))
    print(cirq.dirac_notation(cirq.final_state_vector(qc, initial_state=in_)))
