import logging
import time
from dataclasses import dataclass

import cirq
import numpy as np
import xyz
from qiskit import transpile
from qiskit_aer import AerSimulator

from qclib.state_preparation import LowRankInitialize
from sp.qiskit_to_cirq import qiskit2cirq
from sp.special_states import GenearlizdeWTypeState, GenearlizdeWTypeStateWithPLU
from sp.special_three_qubit_states import ThreeQubitWType
from sp.symbolic_expression.w_type import *
from sp.utils import get_number_of_cnot

logger = logging.getLogger(__name__)


class StatePreparation:

    def run(self, state_vector: np.ndarray) -> "StatePreparationResult": ...


class QuantumXYZ(StatePreparation):

    def run(self, state_vector: np.ndarray, target_object: str = "cirq"):
        logger.info("State to Prepare")
        logger.info(cirq.dirac_notation(state_vector))
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


@dataclass
class StatePreparationResult:

    def __init__(
        self,
        state_prep_engine: StatePreparation,
        goal_sv: np.ndarray,
        circuit: cirq.Circuit,
    ):
        self.elapsed_time = None
        self.num_cnot = None
        self.cnot_depth = None

    def export(self):
        raise NotImplementedError
