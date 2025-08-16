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


# TODO : remove following later
# def qiskit2cirq(qiskit_qc: qiskit.QuantumCircuit) -> cirq.Circuit:
#     qasm_str = qiskit.qasm2.dumps(qiskit_qc)
#     cirq_qc = circuit_from_qasm(qasm_str)
#     return cirq_qc


def qiskit2cirq(qiskit_qc: qiskit.QuantumCircuit, do_reverse=False) -> cirq.Circuit:
    cirq_qc = cirq.Circuit()
    for gate in qiskit_qc.data:
        if gate[0].name == "u3":
            theta, phi, lam = gate[0].params
            theta, phi, lam = (
                float(theta) / np.pi,
                float(phi) / np.pi,
                float(lam) / np.pi,
            )
            qubit = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))

            cirq_qc.append(QasmUGate(theta, phi, lam)(qubit))
        elif gate[0].name == "cx":
            control = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            target = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[1]))
            cirq_qc.append(cirq.CNOT(control, target))
        else:
            raise ValueError("Unexpected gate type: {}".format(gate))
    # TODO : remove following do_reverse routine for generality
    # either call of qclib or qiskit should handle the endian consistency
    if do_reverse:
        qubits = cirq.LineQubit.range(len(qiskit_qc.qubits))
        rev_qubits = list(reversed(qubits))
        qubit_map = dict(zip(qubits, rev_qubits))
        cirq_qc = cirq_qc.transform_qubits(qubit_map)
    return cirq_qc
