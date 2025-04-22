import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Final, List, Union

import cirq
import cirq.circuits
import numpy as np
import qiskit

from state_preparation.circuit_converter import qiskit2cirq
from state_preparation.utils import (
    num_cnot_for_cirq_circuit,
    validate_result_cirq_circuit,
)

if TYPE_CHECKING:
    from .algorithms import StatePreparation

logger = logging.getLogger(__name__)

AVAILABLE_RESULT_ITEMS: Final[str] = ["num_cnot", "depth", "elapsed_time"]


def item_result_expr_render(result: Any) -> str:
    if type(result) is float:
        return str(round(result, 3))
    return str(result)


@dataclass
class StatePreparationResult:
    state_prep_engine: "StatePreparation"
    target_sv: np.ndarray
    circuit: Union[cirq.Circuit, qiskit.QuantumCircuit]
    elapsed_time: float

    @property
    def available_result_item(self):
        return AVAILABLE_RESULT_ITEMS

    @property
    def _result_item_rank(self) -> Dict[str, int]:
        return {item: idx for idx, item in enumerate(self.available_result_item)}

    def _export_to_row_data(self, result_items: List[str]) -> List[str]:
        # should be exclusively used for rich.table.Table().add_row()
        if result_items:
            for item in result_items:
                if item not in self.available_result_item:
                    raise ValueError(f"Invalid Result Item {item}")
        else:
            result_items = AVAILABLE_RESULT_ITEMS

        sorted_result_items = sorted(
            result_items, key=lambda x: self._result_item_rank[x]
        )
        return [
            item_result_expr_render(getattr(self, item)) for item in sorted_result_items
        ]

    @cached_property
    def cirq_circuit(self) -> cirq.Circuit:
        cirq_circuit = (
            qiskit2cirq(self.circuit.reverse_bits())
            if isinstance(self.circuit, qiskit.QuantumCircuit)
            else self.circuit
        )
        normalized_cirq_qc = cirq.merge_single_qubit_gates_to_phxz(cirq_circuit)
        validate_result_cirq_circuit(normalized_cirq_qc)
        return normalized_cirq_qc

    @cached_property
    def qiskit_circuit(self) -> qiskit.QuantumCircuit:
        raise NotImplementedError

    def _get_noramlized_cirq_circuit(self):
        raise NotImplementedError

    @property
    def num_cnot(self):
        return num_cnot_for_cirq_circuit(self.cirq_circuit)

    @property
    def depth(self):
        return len(self.cirq_circuit)
