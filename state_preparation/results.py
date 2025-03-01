import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Union

import cirq
import cirq.circuits
import numpy as np
import qiskit

if TYPE_CHECKING:
    from .state_prep_algorithms import StatePreparation

logger = logging.getLogger(__name__)


@dataclass
class StatePreparationResult:
    state_prep_engine: "StatePreparation"
    target_sv: np.ndarray
    circuit: Union[cirq.Circuit, qiskit.QuantumCircuit]
    elapsed_time: float

    def export(self):
        raise NotImplementedError

    @cached_property
    def cirq_circuit(self) -> cirq.Circuit:
        raise NotImplementedError

    @cached_property
    def qiskit_circuit(self) -> qiskit.QuantumCircuit:
        raise NotImplementedError

    @property
    def num_cnot(self):
        pass

    @property
    def depth(self):
        pass
