import logging
from abc import ABC, abstractmethod
from typing import Optional

import cirq
import numpy as np
import qiskit
import qiskit.circuit.library
import xyz
from qiskit import transpile
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector

from qclib.state_preparation import LowRankInitialize, UCGEInitialize

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
            raise e

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

        assert isinstance(circuit, qiskit.QuantumCircuit)
        transpiled_circuit = transpile(
            circuit, basis_gates=["u3", "cx"], optimization_level=0
        )

        return StatePreparationResult(
            state_prep_engine=self,
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


class UCGEBased(StatePreparationBase):
    # implementation of https://arxiv.org/abs/2409.05618

    def __init__(self, skip_qc_validation: bool = False):
        self.skip_qc_validation = skip_qc_validation

    def _get_result(self, state_vector: np.ndarray) -> StatePreparationResult:
        sv_num_qubit = num_qubit(state_vector)
        if sv_num_qubit < 10:
            logger.info(f"State to Prepare : {cirq.dirac_notation(state_vector)}")
        logger.info(f"Num qubit : {sv_num_qubit}")

        logger.info("Running UCGE")
        with catchtime() as time:
            circuit = UCGEInitialize(state_vector).definition

        transpiled_circuit = transpile(
            circuit, basis_gates=["u3", "cx"], optimization_level=0
        )
        assert isinstance(transpiled_circuit, qiskit.QuantumCircuit)
        return StatePreparationResult(
            state_prep_engine=self,
            target_sv=state_vector,
            circuit=transpiled_circuit,
            elapsed_time=time(),
            skip_qc_validation=self.skip_qc_validation,
        )

    @property
    def name(self):
        return "UCGE"

    def __eq__(self, value):
        return type(self) is type(value)


class IsometryBased(StatePreparationBase):
    # qiskit implementation of https://arxiv.org/abs/1501.06911

    def __init__(self, skip_qc_validation: bool = False):
        self.skip_qc_validation = skip_qc_validation

    def _get_result(self, state_vector: np.ndarray):
        qiskit_qc = qiskit.QuantumCircuit(num_qubit(state_vector))
        qiskit_qc.append(
            qiskit.circuit.library.StatePreparation(
                params=Statevector(data=state_vector)
            ),
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


class XYZ(StatePreparationBase):

    def __init__(self, skip_qc_validation: bool = False):
        self.skip_qc_validation = skip_qc_validation

    def _get_result(self, state_vector: np.ndarray):
        if np.iscomplexobj(state_vector) and np.any(np.imag(state_vector) != 0):
            raise ValueError(
                "State vector contains non-real elements; XYZ algorithm does not support complex-valued state vectors."
            )

        with catchtime() as time:
            logger.info("Starting XYZ")
            param = xyz.StatePreparationParameters(
                enable_compression=False,
                enable_m_flow=True,
                enable_n_flow=False,
                enable_exact_synthesis=True,
            )

            qc = xyz.prepare_state(
                state_vector, verbose_level=0, map_gates=True, param=param
            )
            logger.info(f"CNOT cost {qc.get_cnot_cost()}")
        qiskit_qc = xyz.to_qiskit(qc)
        transpiled_circuit = transpile(
            qiskit_qc, basis_gates=["u3", "cx"], optimization_level=0
        )
        return StatePreparationResult(
            state_prep_engine=self,
            target_sv=state_vector,
            circuit=transpiled_circuit,
            elapsed_time=time(),
            skip_qc_validation=self.skip_qc_validation,
        )

    @property
    def name(self):
        return "XYZ"

    def __eq__(self, value):
        return type(self) is type(value)
