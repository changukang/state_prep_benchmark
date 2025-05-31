import logging
import statistics
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Final, List, Union

import cirq
import numpy as np
import qiskit

from state_preparation.circuit_converter import qiskit2cirq
from state_preparation.utils import (
    num_cnot_for_cirq_circuit,
    validate_result_cirq_circuit,
)

if TYPE_CHECKING:
    from .algorithms import StatePreparationBase

logger = logging.getLogger(__name__)
NUM_CNOT = "num_cnot"
DEPTH = "depth"
ELAPSED_TIME = "elapsed_time"
AVAILABLE_RESULT_ITEMS: Final[List[str]] = [NUM_CNOT, DEPTH, ELAPSED_TIME]


def item_result_expr_render(result: Any) -> str:
    if type(result) is float:
        return str(round(result, 3))
    return str(result)


@dataclass
class StatePreparationResult:
    state_prep_engine: "StatePreparationBase"
    target_sv: np.ndarray
    circuit: Union[cirq.Circuit, qiskit.QuantumCircuit]
    elapsed_time: float

    available_result_item: ClassVar[List[str]] = AVAILABLE_RESULT_ITEMS
    result_items_rank: ClassVar[Dict[str, int]] = {
        item: idx for idx, item in enumerate(AVAILABLE_RESULT_ITEMS)
    }

    def _export_to_row_data(self, result_items: List[str]) -> List[str]:
        # should be exclusively used for rich.table.Table().add_row()
        if result_items:
            for item in result_items:
                if item not in StatePreparationResult.available_result_item:
                    raise ValueError(f"Invalid Result Item {item}")
        else:
            result_items = StatePreparationResult.available_result_item

        sorted_result_items = sorted(
            result_items, key=lambda x: StatePreparationResult.result_items_rank[x]
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
    def num_cnot(self) -> int:
        return num_cnot_for_cirq_circuit(self.cirq_circuit)

    @property
    def depth(self) -> int:
        return len(self.cirq_circuit)


@dataclass
class StatePreparationResultStatistics:
    id: str
    results: List[StatePreparationResult]

    def __post_init__(self):
        # check that
        target_state_prep_engine = self.results[0].state_prep_engine
        if not all(
            res.state_prep_engine == target_state_prep_engine for res in self.results
        ):
            raise ValueError("All State Preparation Engine must be same for statistics")

    def _export_to_row_data(self, result_items: List[str]) -> List[str]:
        # should be exclusively used for rich.table.Table().add_row()
        if result_items:
            for item in result_items:
                if item not in StatePreparationResult.available_result_item:
                    raise ValueError(f"Invalid Result Item {item}")
        else:
            result_items = StatePreparationResult.available_result_item

        sorted_result_items = sorted(
            result_items, key=lambda x: StatePreparationResult.result_items_rank[x]
        )

        return [
            item_result_expr_render(
                statistics.mean([getattr(res, item) for res in self.results])
            )
            for item in sorted_result_items
        ]
