import logging
import time
from dataclasses import dataclass
import qiskit

import cirq
import numpy as np
import xyz
from qclib.state_preparation import LowRankInitialize
from qiskit import transpile
from qiskit_aer import AerSimulator
from sp.qiskit_to_cirq import qiskit2cirq
from sp.special_states import GenearlizdeWTypeState, GenearlizdeWTypeStateWithPLU
from sp.special_three_qubit_states import ThreeQubitWType
from sp.symbolic_expression.w_type import *
from sp.utils import get_number_of_cnot
from functools import cached_property

logger = logging.getLogger(__name__)


class StatePreparation:

    def run(
        self, state_vector: np.ndarray, target_object: str = "cirq"
    ) -> "StatePreparationResult": ...


class QuantumXYZ(StatePreparation):
    # implementation of https://arxiv.org/abs/2401.01009

    def run(self, state_vector: np.ndarray, target_object: str = "cirq"):
        logger.info(f"State to Prepare : {cirq.dirac_notation(state_vector)}")
        logger.info(f"num qubit : {int(np.log2(state_vector.shape[0]))}")
        logger.info("Running XYZ")
        target_state = xyz.quantize_state(state_vector)
        # synthesize the state
        start_time = time.time()
        circuit = xyz.prepare_state(target_state, map_gates=True, verbose_level=0)
        end_time = time.time()
        synthesized_qc = xyz.to_qiskit(circuit)
        backend = AerSimulator()
        t_circuit = transpile(
            synthesized_qc, backend, basis_gates=["u1", "u2", "u3", "cx"]
        )
        return t_circuit, (end_time - start_time)


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
        elapsed_time : float
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
    