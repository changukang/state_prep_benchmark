import logging
from abc import ABC, abstractmethod
from typing import Optional

import cirq
import numpy as np
import qiskit
from qiskit import transpile
from qiskit.exceptions import QiskitError

from qclib.state_preparation import LowRankInitialize

from .results import StatePreparationResult
from .utils import catchtime, num_qubit

logger = logging.getLogger(__name__)
QISKIT = "qiskit"
CIRQ = "cirq"


class InvalidStatePreparationResult(Exception):

    def __init__(self, e: Optional[Exception] = None):
        msg = f"Exception was raised : {e}"
        super().__init__(msg)


class StatePreparationBase(ABC):

    def run(self, state_vector: np.ndarray) -> StatePreparationResult:
        self._pre_run(state_vector)
        result_qc = None
        try:
            result_qc = self._get_result(state_vector)
            return result_qc
        except QiskitError as e:
            raise InvalidStatePreparationResult(e)

    def _pre_run(self, state_vector: np.ndarray):
        sv_num_qubit = num_qubit(state_vector)
        if sv_num_qubit < 10:
            logger.info(f"State to Prepare : {cirq.dirac_notation(state_vector)}")
        logger.info(f"Num qubit : {sv_num_qubit}")

    @property
    @abstractmethod
    def name(self): ...

    @abstractmethod
    def _get_result(
        state_vector: np.ndarray,
    ) -> StatePreparationResult: ...

    @abstractmethod
    def __eq__(self, value) -> bool: ...


class LowRankStatePrep(StatePreparationBase):
    # implementation of https://arxiv.org/abs/2111.03132

    def __init__(self, skip_qc_validation: bool = False):
        self.skip_qc_validation = skip_qc_validation

    def _get_result(self, state_vector: np.ndarray) -> StatePreparationResult:
        sv_num_qubit = num_qubit(state_vector)
        if sv_num_qubit < 10:
            logger.info(f"State to Prepare : {cirq.dirac_notation(state_vector)}")
        logger.info(f"Num qubit : {sv_num_qubit}")

        logger.info("Running LowRankStatePrep")
        with catchtime() as time:
            circuit = LowRankInitialize(state_vector).definition

        transpiled_circuit = transpile(
            circuit, basis_gates=["u3", "cx"], optimization_level=0
        )
        assert isinstance(transpiled_circuit, qiskit.QuantumCircuit)
        return StatePreparationResult(
            state_prep_engine=type(self),
            target_sv=state_vector,
            circuit=transpiled_circuit,
            elapsed_time=time(),
            skip_qc_validation=self.skip_qc_validation,
        )

    @property
    def name(self):
        return "Low Rank"

    def __eq__(self, value):
        return type(self) is type(value)


class IsometryBased(StatePreparationBase):
    # qiskit implementation of https://arxiv.org/abs/1501.06911

    def __init__(self, skip_qc_validation: bool = False):
        self.skip_qc_validation = skip_qc_validation

    def _get_result(self, state_vector: np.ndarray):
        qiskit_qc = qiskit.QuantumCircuit(num_qubit(state_vector))
        qiskit_qc.append(
            qiskit.circuit.library.StatePreparation(params=state_vector),
            range(num_qubit(state_vector)),
        )
        with catchtime() as time:
            transpiled_qiskit_qc = transpile(
                qiskit_qc, basis_gates=["u3", "cx"], optimization_level=0
            )
        assert isinstance(transpiled_qiskit_qc, qiskit.QuantumCircuit)

        return StatePreparationResult(
            state_prep_engine=self,
            target_sv=state_vector,
            circuit=transpiled_qiskit_qc,
            elapsed_time=time(),
            skip_qc_validation=self.skip_qc_validation,
        )

    @property
    def name(self):
        return "Isometry"

    def __eq__(self, value):
        return type(self) is type(value)
