import cirq
import cirq.circuits
import numpy as np
import qiskit
import qiskit.qasm2
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import circuit_from_qasm
from cirq.transformers.analytical_decompositions import \
    decompose_multi_controlled_rotation


def qiskit2cirq_by_qasm(qiskit_qc: qiskit.QuantumCircuit) -> cirq.Circuit:
    qasm_str = qiskit.qasm2.dumps(qiskit_qc)
    cirq_qc = circuit_from_qasm(qasm_str)
    return cirq_qc


def rccx_decomposition(control1, control2, target):
    return [
        cirq.H(target),
        cirq.T(target),
        cirq.CX(control2, target),
        (cirq.T**-1)(target),
        cirq.CX(control1, target),
        cirq.T(target),
        cirq.CX(control2, target),
        (cirq.T**-1)(target),
        cirq.H(target),
    ]


class OptimalToffoli(cirq.Gate):
    def num_qubits(self) -> int:
        return 3

    def _unitary_(self):
        return cirq.unitary(cirq.CCX)

    def _decompose_(self, qubits):
        # from https://en.wikipedia.org/wiki/Toffoli_gate
        control1, control2, target = qubits
        yield (cirq.H(target))
        yield (cirq.CX(control2, target))
        yield ((cirq.T**-1)(target))
        yield (cirq.CX(control1, target))
        yield (cirq.T(target))
        yield (cirq.CX(control2, target))
        yield ((cirq.T**-1)(target))
        yield (cirq.CX(control1, target))
        yield (cirq.T(control2))
        yield (cirq.T(target))
        yield (cirq.H(target))
        yield (cirq.CX(control1, control2))
        yield ((cirq.T**-1)(control2))
        yield (cirq.T(control1))
        yield (cirq.CX(control1, control2))


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
        elif gate.operation.name == "cp":
            control = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            target = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[1]))
            exponent = gate.operation.params[0] / np.pi
            cirq_qc.append(
                cirq.ZPowGate(exponent=exponent / 2)(control)
            )  # for matching phase
            cirq_qc.append(cirq.CX(control, target))
            cirq_qc.append(cirq.ZPowGate(exponent=-exponent / 2)(target))
            cirq_qc.append(cirq.CX(control, target))
            cirq_qc.append(cirq.ZPowGate(exponent=exponent / 2)(target))
        elif gate.operation.name == "rccx":
            control1 = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            control2 = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[1]))
            target = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[2]))
            cirq_qc.append(rccx_decomposition(control1, control2, target))
        elif gate.operation.name == "rccx_dg":
            control1 = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            control2 = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[1]))
            target = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[2]))
            cirq_qc.append(
                cirq.Circuit(rccx_decomposition(control1, control2, target)) ** -1
            )
        elif gate.operation.name == "ccx":
            control1 = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[0]))
            control2 = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[1]))
            target = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[2]))
            cirq_qc.append(
                cirq.decompose_once(OptimalToffoli()(control1, control2, target))
            )
        elif gate.operation.name == "mcphase":
            controls = [
                cirq.LineQubit(qiskit_qc.qubits.index(q)) for q in gate.qubits[:-1]
            ]
            target = cirq.LineQubit(qiskit_qc.qubits.index(gate.qubits[-1]))
            exponent = gate.operation.params[0] / np.pi
            mat = cirq.unitary(cirq.ZPowGate(exponent=exponent))
            decomposed_ops = decompose_multi_controlled_rotation(mat, controls, target)
            into_optimal_toffoli = list()
            for op in decomposed_ops:
                if op.gate == cirq.TOFFOLI:
                    into_optimal_toffoli += cirq.decompose_once(
                        OptimalToffoli()(*op.qubits)
                    )
                else:
                    into_optimal_toffoli.append(op)

            assert all(
                len(op.qubits) <= 2 for op in into_optimal_toffoli
            ), "Decomposed operations must be at most 2-qubit gates"
            cirq_qc.append(into_optimal_toffoli)

            # cirq_qc.append(
            #     cirq.ControlledGate(
            #         sub_gate=cirq.ZPowGate(exponent=exponent),
            #         num_controls=len(controls),
            #     )(*controls, target)
            # )

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
