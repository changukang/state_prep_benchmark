import logging
import time
from dataclasses import dataclass
from functools import cached_property
from typing import Union

import cirq
import numpy as np
import qiskit
from qiskit import transpile
from qiskit_aer import AerSimulator
from abc import ABC, abstractmethod

from qclib.state_preparation import LowRankInitialize

logger = logging.getLogger(__name__)


class StatePreparation(ABC):
    
    @abstractmethod
    def run(
        self, state_vector: np.ndarray, target_object: str = "cirq"
    ) -> "StatePreparationResult": ...

class LowRankStatePrep(StatePreparation):
    # implementation of https://arxiv.org/abs/2111.03132

    def run(self, state_vector: np.ndarray, target_object: str = "cirq"):
        logger.info(f"State to Prepare : {cirq.dirac_notation(state_vector)}")
        logger.info(f"num qubit : {int(np.log2(state_vector.shape[0]))}")
        logger.info("Running LowRankStatePrep")
        backend = AerSimulator()
        circuit = LowRankInitialize(state_vector).definition
        transpiled_circuit = transpile(
            circuit, backend, basis_gates=["u3", "cx"], optimization_level=3
        )
        return transpiled_circuit


class IsometryBased(StatePreparation):
    pass


@dataclass
class StatePreparationResult:
    def __init__(
        self,
        state_prep_engine: StatePreparation,
        goal_sv: np.ndarray,
        circuit: Union[cirq.Circuit, qiskit.QuantumCircuit],
        elapsed_time: float,
    ):
        self.elapsed_time = None
        self.num_cnot = None
        self.cnot_depth = None
        self.circuit = None

    def export(self):
        raise NotImplementedError

    @staticmethod
    def validate_circuit(circuit):
        pass

    @cached_property
    def cirq_circuit(self):
        pass

    @cached_property
    def qiskit_circuit(self):
        pass

    @cached_property
    def num_cnot(self):
        pass

    @cached_property
    def depth(self):
        pass
