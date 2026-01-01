import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Type

import cirq
import numpy as np
import qiskit
import qiskit.circuit.library
import xyz
from qiskit import transpile
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector

from qclib.state_preparation import LowRankInitialize, UCGEInitialize
from qclib.state_preparation.pivot import PivotInitialize
from state_preparation.gates.mcx.types import MCXGateBase
from state_preparation.permutation.types import Permutation, Transposition

from .results import StatePreparationResult
from .utils import (
    catchtime,
    get_global_phase_match,
    num_cnot_for_cirq_circuit,
    num_qubit,
)

logger = logging.getLogger(__name__)
QISKIT = "qiskit"
CIRQ = "cirq"


class InvalidStatePreparationResult(Exception):

    def __init__(self, e: Optional[Exception] = None):
        msg = f"Exception was raised : {e}"
        super().__init__(msg)


class StatePreparationBase(ABC):

    def run(self, state_vector: np.ndarray, **kwargs) -> StatePreparationResult:
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


def statevector_to_sparse_dict(
    statevector: np.ndarray,
    atol: float = 1e-12,
) -> Dict[str, complex]:
    """
    Convert a statevector (np.ndarray) into sparse dict form:
        {'001': amp1, '110': amp2, ...}

    Args:
        statevector: np.ndarray of shape (2**n,)
        atol: threshold below which amplitudes are treated as zero

    Returns:
        Dict[str, complex]
    """
    statevector = np.asarray(statevector)
    dim = statevector.shape[0]

    # sanity check
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("Length of statevector must be power of 2")

    sparse_data = {}
    for idx, amp in enumerate(statevector):
        if np.abs(amp) > atol:
            bitstring = format(idx, f"0{n_qubits}b")
            sparse_data[bitstring] = amp

    return sparse_data


class PivotStatePrep(StatePreparationBase):

    def __init__(self, skip_qc_validation: bool = False):
        self.skip_qc_validation = skip_qc_validation

    def _get_result(self, state_vector: np.ndarray) -> StatePreparationResult:

        logger.info("Running PivotStatePrep")

        data = statevector_to_sparse_dict(state_vector)

        with catchtime() as time:
            circuit = PivotInitialize(data).definition

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
        return "Pivot"

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
                state_vector, verbose_level=2, map_gates=True, param=param
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


class SandwichedPermutation(StatePreparationBase):
    """
    Implements algorithm from paper
        "Nearly Optimal Circuit Size for Sparse Quantum State Preparation"
        https://arxiv.org/pdf/2406.16142
    """

    def __init__(
        self,
        sub_state_preparation: Callable[[np.ndarray], StatePreparationResult],
        mcx_gate_type: Type[MCXGateBase],
        do_validation: bool = False,
    ):
        self.sub_state_preparation = sub_state_preparation
        self.mcx_gate_type = mcx_gate_type
        self.do_validation = do_validation

    def _get_result(
        self,
        state_vector: np.ndarray,
    ) -> StatePreparationResult:
        sv_num_qubit = num_qubit(state_vector)

        def get_target_perm(sv: np.ndarray) -> Permutation:
            nonzero_indicies = (np.where(sv > 1e-8)[0]).tolist()
            sparsity = len(nonzero_indicies)
            flag = np.zeros(sparsity, dtype=int)
            sv_num_qubit = num_qubit(sv)
            perm_building = Permutation.identity(2**sv_num_qubit)
            for i in nonzero_indicies:
                if i < sparsity:
                    flag[i] = 1

            for q in nonzero_indicies:
                if q >= sparsity:
                    k = np.where(flag == 0)[0][0]
                    flag[k] = 1
                    perm_building = perm_building.compose(
                        Transposition(2**sv_num_qubit, k, q)
                    )
            return perm_building

        perm = get_target_perm(state_vector)

        nonzero_indicies = (np.where(state_vector > 1e-8)[0]).tolist()
        sparsity = len(nonzero_indicies)
        dense_sv_to_prep = np.zeros(
            2 ** int(np.ceil(np.log2(sparsity))), dtype=np.complex128
        )

        inversed = perm.inverse()
        for i in nonzero_indicies:
            dense_sv_to_prep[inversed(i)] = state_vector[i]

        qc = cirq.Circuit()
        sub_prep_result = self.sub_state_preparation(dense_sv_to_prep)

        logger.info(
            f"Sub State Preparation #CNOT count : {num_cnot_for_cirq_circuit(sub_prep_result.cirq_circuit)}"
        )

        to_move = sv_num_qubit - num_qubit(dense_sv_to_prep)
        qc += sub_prep_result.cirq_circuit.transform_qubits(
            {
                cirq.LineQubit(i): cirq.LineQubit(i + to_move)
                for i in range(num_qubit(dense_sv_to_prep))
            }
        )
        qc += cirq.global_phase_operation(get_global_phase_match(dense_sv_to_prep, qc))

        perm_qc = perm.index_extraction_based_decomposition_qc(
            cirq.LineQubit.range(sv_num_qubit),
            mcx_gate_type=self.mcx_gate_type,
            do_validation=self.do_validation,
        )

        logger.info(
            f"Premutation Part #CNOT count : {num_cnot_for_cirq_circuit(perm_qc)}"
        )

        assert all(
            op.gate == cirq.CX for op in perm_qc.all_operations() if len(op.qubits) > 1
        )
        qc += perm_qc

        return StatePreparationResult(
            state_prep_engine=self,
            target_sv=state_vector,
            circuit=qc,
            elapsed_time=None,
            skip_qc_validation=True,
        )

    @property
    def name(self):
        return "SequentialSuperpositionSynthesis"

    def __eq__(self, value):
        return type(self) is type(value)
