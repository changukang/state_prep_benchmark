from itertools import product
from typing import Type

import cirq
import numpy as np
import pytest

from state_preparation.circuit_converter import qiskit2cirq
from state_preparation.mcx.increments import (decrement_with_borrowed_qubits,
                                              increment_with_borrowed_qubits)
from state_preparation.mcx.mcx_gates import (CirqStandardMCXGate,
                                             ItenDirtyMCXGate,
                                             KGDirtyOneMCXGate,
                                             KGDirtyTwoMCXGate, MCXGateBase,
                                             QulinMCXGate,
                                             SelectiveOptimalMCXGate,
                                             ValeMCXGate)
from state_preparation.state_samplers import get_random_state
from state_preparation.utils import (keep_ftn_for_cirq_decompose,
                                     num_cnot_for_cirq_circuit)


def twos_complement_to_int(bitstring: str) -> int:
    n = len(bitstring)
    value = int(bitstring, 2)
    if bitstring[0] == "1":  # negative
        value -= 1 << n
    return value


def test_increment_with_borrowed_qubits():
    qubits = cirq.LineQubit.range(8)
    targets = qubits[1::2]  # Odd-indexed qubits as target qubits
    borrowed = qubits[0::2]  # Even-indexed qubits as borrowed qubits
    qc = increment_with_borrowed_qubits(targets, borrowed)

    for target_bits, borrowed_bits in zip(
        range(2 ** len(targets)), range(2 ** len(borrowed))
    ):
        target_state = format(target_bits, f"0{len(targets)}b")
        borrowed_state = format(borrowed_bits, f"0{len(borrowed)}b")
        interleaved_state = "".join(b + t for b, t in zip(borrowed_state, target_state))
        input_state = cirq.one_hot(
            index=int(interleaved_state, 2),
            shape=(2 ** len(qubits),),
            dtype=np.complex128,
        )
        res = qc.final_state_vector(initial_state=input_state)
        result_bits = format(np.argmax(np.abs(res) ** 2), f"0{len(qubits)}b")
        res_target_bits = result_bits[1::2]  # Extract odd-indexed bits

        n = len(target_state)
        orig_val = twos_complement_to_int(target_state[::-1])
        wrapped = (orig_val + 1) & ((1 << n) - 1)
        # Reinterpret wrapped as signed n-bit two's complement
        if wrapped >= (1 << (n - 1)):
            expected_val = wrapped - (1 << n)
        else:
            expected_val = wrapped
        res_val = twos_complement_to_int(res_target_bits[::-1])
        assert expected_val == res_val, f"Expected {expected_val}, but got {res_val}"

        res_borrowed_bits = result_bits[0::2]

        assert res_borrowed_bits == borrowed_state


def test_decrement_with_borrowed_qubits():
    qubits = cirq.LineQubit.range(8)
    targets = qubits[1::2]  # Odd-indexed qubits as target qubits
    borrowed = qubits[0::2]  # Even-indexed qubits as borrowed qubits
    qc = decrement_with_borrowed_qubits(targets, borrowed)

    for target_bits, borrowed_bits in zip(
        range(2 ** len(targets)), range(2 ** len(borrowed))
    ):
        target_state = format(target_bits, f"0{len(targets)}b")
        borrowed_state = format(borrowed_bits, f"0{len(borrowed)}b")
        interleaved_state = "".join(b + t for b, t in zip(borrowed_state, target_state))
        input_state = cirq.one_hot(
            index=int(interleaved_state, 2),
            shape=(2 ** len(qubits),),
            dtype=np.complex128,
        )
        res = qc.final_state_vector(initial_state=input_state)
        result_bits = format(np.argmax(np.abs(res) ** 2), f"0{len(qubits)}b")
        res_target_bits = result_bits[1::2]  # Extract odd-indexed bits

        n = len(target_state)
        orig_val = twos_complement_to_int(target_state[::-1])
        res_val = twos_complement_to_int(res_target_bits[::-1])

        n = len(target_state)
        orig_val = twos_complement_to_int(target_state[::-1])
        wrapped = (orig_val - 1) & ((1 << n) - 1)
        # Reinterpret wrapped as signed n-bit two's complement
        if wrapped >= (1 << (n - 1)):
            expected_val = wrapped - (1 << n)
        else:
            expected_val = wrapped
        res_val = twos_complement_to_int(res_target_bits[::-1])
        assert expected_val == res_val, f"Expected {expected_val}, but got {res_val}"


def test_qulin_cirq():
    from qiskit.synthesis import synth_mcx_noaux_hp24

    for num_control in [2, 3, 4, 5]:
        qiskit_qc = synth_mcx_noaux_hp24(num_control)
        num_qubit = num_control + 1
        cirq_circuit = qiskit2cirq(qiskit_qc)
        oracle_qc = cirq.Circuit(
            cirq.X.controlled(num_control)(*cirq.LineQubit.range(num_qubit))
        )
        for bitstring in (format(i, f"0{num_qubit}b") for i in range(2**num_qubit)):
            int_val = int(bitstring, 2)
            input_state = cirq.one_hot(
                index=int_val,
                shape=(2**num_qubit,),
                dtype=np.complex128,
            )
            res_state = cirq_circuit.final_state_vector(
                initial_state=input_state, qubit_order=cirq.LineQubit.range(num_qubit)
            )
            oracle_state = oracle_qc.final_state_vector(
                initial_state=input_state, qubit_order=cirq.LineQubit.range(num_qubit)
            )
            assert np.allclose(res_state, oracle_state), f"Failed for input {bitstring}"


def test_qulin_mcx_gate():
    qubits = cirq.LineQubit.range(6)
    qulin_gate = QulinMCXGate(num_controls=5)(*qubits)
    # vale_gate = ValeMCXGate(num_controls=5)(*qubits)
    canon_mcx = cirq.X.controlled(5)(*qubits)

    assert cirq.approx_eq(cirq.unitary(qulin_gate), cirq.unitary(canon_mcx), atol=1e-8)
    # assert cirq.approx_eq(cirq.unitary(vale_gate), cirq.unitary(canon_mcx), atol=1e-8)

    res_qulin = cirq.decompose(
        cirq.Circuit(qulin_gate), keep=keep_ftn_for_cirq_decompose
    )
    res_qulin_decompose_once = cirq.decompose_once(qulin_gate)

    # res_vale = cirq.decompose(cirq.Circuit(vale_gate), keep=keep_ftn_for_cirq_decompose)
    res_canon_mcx = cirq.decompose(
        cirq.Circuit(canon_mcx), keep=keep_ftn_for_cirq_decompose
    )
    print(num_cnot_for_cirq_circuit(cirq.Circuit(res_qulin)))
    # print(num_cnot_for_cirq_circuit(cirq.Circuit(res_vale)))
    print(num_cnot_for_cirq_circuit(cirq.Circuit(res_canon_mcx)))


def test_qulin():
    qubits = cirq.LineQubit.range(2)
    qulin_gate = QulinMCXGate(num_controls=1)(*qubits)
    canon_mcx = cirq.X.controlled(1)(qubits[1], qubits[0])

    print(cirq.unitary(qulin_gate))
    print(cirq.unitary(canon_mcx))


@pytest.mark.parametrize(
    "mcx_gate",
    [
        QulinMCXGate,
        ItenDirtyMCXGate,
        ValeMCXGate,
        CirqStandardMCXGate,
        KGDirtyTwoMCXGate,
        KGDirtyOneMCXGate,
    ],
)
def test_mcx_with_aux_qubits(mcx_gate: Type[MCXGateBase]):

    for num_control, num_available_aux, seed in product(
        [3, 4, 5, 6], [5, 6], range(10)
    ):
        num_qubit = num_control + 1 + num_available_aux
        init_sv = get_random_state(num_qubit=num_qubit, seed=seed)
        qubits = cirq.LineQubit.range(num_qubit)
        qc = cirq.Circuit(
            cirq.decompose_once(
                mcx_gate.from_available_aux_qubits(
                    main_qubits=qubits[: num_control + 1],
                    available_aux_qubits=qubits[num_control + 1 :],
                )
            )
        )

        oracle_qc = cirq.Circuit(
            cirq.X(qubits[num_control]).controlled_by(*qubits[:num_control])
        )

        res = qc.final_state_vector(
            initial_state=init_sv, qubit_order=qubits, dtype=np.complex128
        )
        res_oracle = oracle_qc.final_state_vector(
            initial_state=init_sv, qubit_order=qubits, dtype=np.complex128
        )
        print("On MCX Gate:", mcx_gate.__name__)
        print(
            f"Num CNOTs: on num control : {num_control}", num_cnot_for_cirq_circuit(qc)
        )
        assert np.allclose(res, res_oracle, atol=1e-8)


def test_select_mcx_gate():
    qubits = cirq.LineQubit.range(6)
    main_qubits = qubits[:2]
    aux_qubits = qubits[4:]
    res = SelectiveOptimalMCXGate.from_available_aux_qubits(main_qubits, aux_qubits)
    print(res)
