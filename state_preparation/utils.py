from contextlib import contextmanager
from time import perf_counter

import cirq
import numpy as np


def num_qubit(state_vector: np.ndarray) -> int:
    log2_res = np.log2(state_vector.shape[0])
    in_int = int(log2_res)
    if in_int != int(log2_res):
        raise ValueError(f"Invalid Quantum State Vector : {state_vector}")
    return in_int


def num_cnot_for_cirq_circuit(qc: cirq.Circuit) -> int:
    cnt = 0
    for gate_op in qc.all_operations():
        if gate_op._num_qubits_() == 2:
            if gate_op.gate not in [cirq.CX, cirq.CNOT]:
                raise ValueError(f"Invalid two-qubit gate encountered {gate_op}")
            cnt += 1
        elif gate_op._num_qubits_() == 1:
            continue
        elif isinstance(gate_op.gate, cirq.ops.global_phase_op.GlobalPhaseGate):
            continue
        else:
            raise ValueError(
                f"Invalid >3-qubit gate encountered {gate_op.gate} of gate type {type(gate_op.gate)}"
            )
    return cnt


def validate_result_cirq_circuit(circuit: cirq.Circuit):
    # TODO : enhance here
    for gate_op in circuit.all_operations():
        if gate_op._num_qubits_() == 2:
            if gate_op.gate not in [cirq.CX, cirq.CNOT]:
                raise ValueError(f"Invalid two-qubit gate encountered {gate_op}")
        elif gate_op._num_qubits_() == 1:
            continue
        elif isinstance(gate_op.gate, cirq.ops.global_phase_op.GlobalPhaseGate):
            continue
        else:
            raise ValueError(
                f"Invalid >3-qubit gate encountered {gate_op.gate} of gate type {type(gate_op.gate)}"
            )


def validate_result_qiskit_circuit(circuit: cirq.Circuit):
    raise NotImplementedError


# ref : https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
@contextmanager
def catchtime():
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
