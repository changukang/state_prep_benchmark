import logging
from abc import ABC, abstractmethod

import cirq
import numpy as np
import qiskit
from qiskit import transpile
from qiskit_aer import AerSimulator
from .utils import num_qubit, catchtime
from .circuit_converter import qiskit2cirq
from .results import StatePreparationResult
from qclib.state_preparation import LowRankInitialize

logger = logging.getLogger(__name__)
QISKIT = "qiskit"
CIRQ = "cirq"


class StatePreparation(ABC):

    @abstractmethod
    def run(
        self, state_vector: np.ndarray, target_object: str = CIRQ
    ) -> StatePreparationResult: ...


class LowRankStatePrep(StatePreparation):
    # implementation of https://arxiv.org/abs/2111.03132

    def run(self, state_vector: np.ndarray):
        sv_num_qubit = num_qubit(state_vector)
        if sv_num_qubit < 10:
            logger.info(f"State to Prepare : {cirq.dirac_notation(state_vector)}")
        logger.info(f"Num qubit : {sv_num_qubit}")

        logger.info("Running LowRankStatePrep")
        backend = AerSimulator()
        with catchtime() as time : 
            circuit = LowRankInitialize(state_vector).definition

        transpiled_circuit = transpile(
            circuit, backend, basis_gates=["u3", "cx"], optimization_level=3
        )
        assert isinstance(transpiled_circuit, qiskit.QuantumCircuit)

        return StatePreparationResult(
            state_prep_engine=type(self),
            target_sv=state_vector,
            circuit = transpiled_circuit,
            elapsed_time=time()
        )


class IsometryBased(StatePreparation):
    # qiskit implementation of https://arxiv.org/abs/1501.06911

    def run(self):
        qiskit_qc = qiskit.QuantumCircuit(n)
        qiskit_qc.append(qiskit.circuit.library.StatePreparation(params=sv), range(n))

        transpiled_qiskit_qc = transpile(qiskit_qc, basis_gates=["u3", "cx"])
        transpiled_cirq_qc = qiskit2cirq(transpiled_qiskit_qc)
