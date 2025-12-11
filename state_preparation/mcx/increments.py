from typing import Sequence
import cirq


def aux_increment_with_borrowed_qubits(
    target: Sequence[cirq.LineQubit], borrowed: Sequence[cirq.LineQubit]
) -> cirq.Circuit:
    qc = cirq.Circuit()
    for idx, q in enumerate(borrowed[:-1]):
        qc.append(cirq.CX(q, target[idx]))
        qc.append(cirq.CX(borrowed[idx + 1], borrowed[idx]))
        qc.append(cirq.CCX(borrowed[idx], target[idx], borrowed[idx + 1]))

    qc.append(cirq.CX(borrowed[-1], target[-1]))

    for idx, q in reversed(list(enumerate(borrowed[:-1]))):
        qc.append(cirq.CCX(borrowed[idx], target[idx], borrowed[idx + 1]))
        qc.append(cirq.CX(borrowed[idx + 1], borrowed[idx]))
        qc.append(cirq.CX(borrowed[idx + 1], target[idx]))
    return qc


def increment_with_borrowed_qubits(
    target: Sequence[cirq.LineQubit], borrowed: Sequence[cirq.LineQubit]
) -> cirq.Circuit:

    assert len(target) == len(borrowed)

    qc = cirq.Circuit()
    for q in target:
        qc.append(cirq.CX(borrowed[0], q))

    for q in borrowed[1:]:
        qc.append(cirq.X(q))

    qc.append(cirq.X(target[-1]))

    qc += aux_increment_with_borrowed_qubits(target, borrowed)

    for q in borrowed[1:]:
        qc.append(cirq.X(q))
    qc += aux_increment_with_borrowed_qubits(target, borrowed)

    for q in target:
        qc.append(cirq.CX(borrowed[0], q))

    return qc


def decrement_with_borrowed_qubits(
    target: Sequence[cirq.LineQubit], borrowed: Sequence[cirq.LineQubit]
) -> cirq.Circuit:

    assert len(target) == len(borrowed)
    qc = cirq.Circuit()

    for q in target:
        qc.append(cirq.X(q))
    qc += increment_with_borrowed_qubits(target, borrowed)

    for q in target:
        qc.append(cirq.X(q))

    return qc
