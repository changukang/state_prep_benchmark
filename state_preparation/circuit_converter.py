import cirq
import qiskit
import qiskit.qasm2
from cirq.contrib.qasm_import import circuit_from_qasm


def qiskit2cirq(qiskit_qc: qiskit.QuantumCircuit) -> cirq.Circuit:
    qasm_str = qiskit.qasm2.dumps(qiskit_qc)
    cirq_qc = circuit_from_qasm(qasm_str)
    return cirq_qc
