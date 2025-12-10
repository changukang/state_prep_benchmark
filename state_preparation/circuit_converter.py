import cirq
import numpy as np
import qiskit
import qiskit.qasm2
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import circuit_from_qasm


def qiskit2cirq_by_qasm(qiskit_qc: qiskit.QuantumCircuit) -> cirq.Circuit:
    qasm_str = qiskit.qasm2.dumps(qiskit_qc)
    cirq_qc = circuit_from_qasm(qasm_str)
    return cirq_qc


def qiskit2cirq(qiskit_qc: qiskit.QuantumCircuit, do_reverse=False) -> cirq.Circuit:
    cirq_qc = cirq.Circuit()
    for gate in qiskit_qc.data:
        # TODO : check in matrix rather than the name for the robustness
        if gate.operation.name == "u3":
            theta, phi, lam = gate.operation.params
            theta, phi, lam = (
                float(theta) / np.pi,
                float(phi) / np.pi,
                float(lam) / np.pi,
            )
            qubit = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            cirq_qc.append(QasmUGate(theta, phi, lam)(qubit))
        elif gate.operation.name == "cx":
            control = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            target = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[1]))
            cirq_qc.append(cirq.CNOT(control, target))
        elif gate.operation.name == "h":
            qubit = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            cirq_qc.append(cirq.H(qubit))
        elif gate.operation.name == "x":
            qubit = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            cirq_qc.append(cirq.X(qubit))
        elif gate.operation.name == "p":
            exponent = gate.operation.params[0] / np.pi
            qubit = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            cirq_qc.append(cirq.ZPowGate(exponent=exponent)(qubit))
        elif gate.operation.name == "tdg":
            qubit = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            cirq_qc.append((cirq.T**-1)(qubit))
        elif gate.operation.name == "t":
            qubit = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            cirq_qc.append((cirq.T)(qubit))
        else:
            raise ValueError(f"Unexpected gate type: {gate[0]}")
    # TODO : remove following do_reverse routine for generality
    # either call of qclib or qiskit should handle the endian consistency
    if do_reverse:
        qubits = cirq.LineQubit.range(len(qiskit_qc.qubits))
        rev_qubits = list(reversed(qubits))
        qubit_map = dict(zip(qubits, rev_qubits))
        cirq_qc = cirq_qc.transform_qubits(qubit_map)
    return cirq_qc
