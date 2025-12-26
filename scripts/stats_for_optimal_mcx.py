from typing import Dict, Tuple, Type

import cirq.circuits

from state_preparation.gates.mcx.types import (
    CirqStandardMCXGate,
    ItenDirtyMCXGate,
    KGDirtyOneMCXGate,
    KGDirtyTwoMCXGate,
    MCXGateBase,
    QulinMCXGate,
    ValeMCXGate,
)
from state_preparation.gates.utils import NotEnoughAuxQubits
from state_preparation.utils import num_cnot_for_cirq_circuit


def run(
    max_num_total_qubits: int, assertion_check: bool = False
) -> Dict[Tuple[int, int], Type[MCXGateBase]]:
    results = {}
    for num_total_qubits in range(4, max_num_total_qubits + 1):
        for num_main_qubits in range(4, num_total_qubits + 1):
            num_aux_qubits = num_total_qubits - num_main_qubits

            print(
                f"Num main qubits: {num_main_qubits}, num aux qubits: {num_aux_qubits}"
            )
            collected = list()
            for type_mcx in [
                QulinMCXGate,
                ItenDirtyMCXGate,
                ValeMCXGate,
                CirqStandardMCXGate,
                KGDirtyTwoMCXGate,
                KGDirtyOneMCXGate,
            ]:
                qc = cirq.Circuit()
                assert issubclass(type_mcx, MCXGateBase)
                try:
                    mcx_gate_op = type_mcx.from_available_aux_qubits(
                        main_qubits=cirq.LineQubit.range(num_main_qubits),
                        available_aux_qubits=cirq.LineQubit.range(
                            num_main_qubits, num_total_qubits
                        ),
                    )
                except NotEnoughAuxQubits as e:
                    print(f"Skipping {type_mcx} due to not enough aux qubits: {e}")
                    continue

                qc = cirq.Circuit(cirq.decompose_once(mcx_gate_op))
                print(
                    f"On mcx_type : {type_mcx}, num CNOTs: ",
                    num_cnot_for_cirq_circuit(qc),
                )

                collected.append((type_mcx, num_cnot_for_cirq_circuit(qc)))

            results[(num_main_qubits, num_aux_qubits)] = min(
                collected, key=lambda x: x[1]
            )

    return results


if __name__ == "__main__":
    res = run(max_num_total_qubits=12, assertion_check=True)

    # (num_main_qubits, num_aux_qubits) : (Best MCX Gate Type, Num CNOTs)
    oracle = {
        (4, 0): (ItenDirtyMCXGate, 14),
        (4, 1): (ItenDirtyMCXGate, 14),
        (5, 0): (ValeMCXGate, 36),
        (4, 2): (ItenDirtyMCXGate, 14),
        (5, 1): (KGDirtyOneMCXGate, 30),
        (6, 0): (QulinMCXGate, 96),
        (4, 3): (ItenDirtyMCXGate, 14),
        (5, 2): (ItenDirtyMCXGate, 26),
        (6, 1): (KGDirtyOneMCXGate, 42),
        (7, 0): (QulinMCXGate, 136),
        (4, 4): (ItenDirtyMCXGate, 14),
        (5, 3): (ItenDirtyMCXGate, 26),
        (6, 2): (KGDirtyTwoMCXGate, 42),
        (7, 1): (KGDirtyOneMCXGate, 54),
        (8, 0): (QulinMCXGate, 192),
        (4, 5): (ItenDirtyMCXGate, 14),
        (5, 4): (ItenDirtyMCXGate, 26),
        (6, 3): (ItenDirtyMCXGate, 34),
        (7, 2): (KGDirtyTwoMCXGate, 54),
        (8, 1): (KGDirtyOneMCXGate, 66),
        (9, 0): (QulinMCXGate, 264),
        (4, 6): (ItenDirtyMCXGate, 14),
        (5, 5): (ItenDirtyMCXGate, 26),
        (6, 4): (ItenDirtyMCXGate, 34),
        (7, 3): (KGDirtyTwoMCXGate, 54),
        (8, 2): (KGDirtyTwoMCXGate, 66),
        (9, 1): (KGDirtyOneMCXGate, 78),
        (10, 0): (QulinMCXGate, 344),
        (4, 7): (ItenDirtyMCXGate, 14),
        (5, 6): (ItenDirtyMCXGate, 26),
        (6, 5): (ItenDirtyMCXGate, 34),
        (7, 4): (ItenDirtyMCXGate, 42),
        (8, 3): (KGDirtyTwoMCXGate, 66),
        (9, 2): (KGDirtyTwoMCXGate, 78),
        (10, 1): (KGDirtyOneMCXGate, 90),
        (11, 0): (QulinMCXGate, 464),
        (4, 8): (ItenDirtyMCXGate, 14),
        (5, 7): (ItenDirtyMCXGate, 26),
        (6, 6): (ItenDirtyMCXGate, 34),
        (7, 5): (ItenDirtyMCXGate, 42),
        (8, 4): (KGDirtyTwoMCXGate, 66),
        (9, 3): (KGDirtyTwoMCXGate, 78),
        (10, 2): (KGDirtyTwoMCXGate, 90),
        (11, 1): (KGDirtyOneMCXGate, 102),
        (12, 0): (QulinMCXGate, 576),
    }

    assert res == oracle
