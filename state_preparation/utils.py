import cirq
import numpy as np
import qiskit


def num_qubit(state_vector: np.ndarray) -> int:
    log2_res = np.log2(state_vector.shape[0])
    in_int = int(log2_res)
    if in_int != int(log2_res):
        raise ValueError(f"Invalid Quantum State Vector : {state_vector}")
    return in_int


def validate_result_cirq_circuit(circuit: cirq.Circuit):
    pass


def validate_result_qiskit_circuit(circuit: cirq.Circuit):
    pass
