import logging
import math
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Type,
)

import cirq
import cirq.circuits
import numpy as np
import qiskit
import qiskit.circuit.library
import qiskit.quantum_info
import qiskit.synthesis
from qiskit import transpile

import qclib
import qclib.isometry
import qclib.unitary
from state_preparation.circuit_converter import qiskit2cirq
from state_preparation.gates.mcp.types import MCPhaseGateBase
from state_preparation.gates.mcx.types import SelectiveOptimalMCXGate
from state_preparation.householder.types import HouseHolderBasedMapping
from state_preparation.results import StatePreparationResult
from state_preparation.utils import (
    get_global_phase_match,
    keep_ftn_for_cirq_decompose,
    num_cnot_for_cirq_circuit,
)

logger = logging.getLogger(__name__)


class IsometryBase:

    def __init__(self, isometry_matrix: np.ndarray | List[List[complex | float]]):
        if isinstance(isometry_matrix, np.ndarray):
            if isometry_matrix.ndim != 2:
                raise ValueError("isometry_matrix must be a 2D array.")
            self._isometry_matrix = isometry_matrix
        elif isinstance(isometry_matrix, list) and all(
            isinstance(row, list) for row in isometry_matrix
        ):
            self._isometry_matrix = np.array(isometry_matrix)
        else:
            raise ValueError("isometry_matrix must be a 2D list or a 2D numpy.ndarray.")

        num_rows = len(self.isometry_matrix)
        if not (num_rows > 0 and (num_rows & (num_rows - 1)) == 0):
            raise ValueError(
                "The number of rows in isometry_matrix must be a power of 2."
            )

    @property
    def isometry_matrix(self) -> np.ndarray:
        return self._isometry_matrix

    @property
    def domain_num_qubit(self) -> int:
        return math.ceil(np.log2(self.isometry_matrix.shape[1]))

    @property
    def codomain_num_qubit(self) -> int:
        return math.ceil(np.log2(self.isometry_matrix.shape[0]))

    def to_quantum_circuit(**kwargs) -> cirq.Circuit:
        raise NotImplementedError()


class HoulseHolderBasedDenseIsometry(IsometryBase):

    def to_quantum_circuit(
        self,
        state_preparation: Callable[[np.ndarray], StatePreparationResult],
        main_qubits: Sequence[cirq.Qid],
        available_aux_qubits: Sequence[cirq.Qid],
        mcp_gate: Type[MCPhaseGateBase],
        mcx_gate: Type[MCPhaseGateBase] = SelectiveOptimalMCXGate,
    ) -> cirq.Circuit:

        assert len(main_qubits) == self.domain_num_qubit

        qc = cirq.Circuit()

        curr_V = self.isometry_matrix

        for idx in range(self.isometry_matrix.shape[1]):

            idx_sv = cirq.one_hot(
                index=idx, shape=(2**self.codomain_num_qubit,), dtype=np.complex128
            )
            curr_targ = curr_V[:, idx]
            if (
                idx == self.isometry_matrix.shape[1] - 1
                and self.domain_num_qubit == self.codomain_num_qubit
            ):
                assert cirq.equal_up_to_global_phase(
                    curr_targ, idx_sv
                ), "Only the last element should be non-zero."

                continue

            hh_based = HouseHolderBasedMapping(curr_targ, idx_sv, strict=True)
            hh_based_qc = hh_based.to_quantum_circuit(
                state_preparation=state_preparation,
                main_qubits=main_qubits,
                available_aux_qubits=available_aux_qubits,
                mcp_gate=mcp_gate,
                mcx_gate=mcx_gate,
            )
            logger.info("Householder for column", idx)
            print("The cnot num is", num_cnot_for_cirq_circuit(hh_based_qc))

            qc += hh_based_qc

            curr_V = hh_based_qc.unitary() @ curr_V

        ret_qc = qc**-1
        diag_angles_radians = list()

        for res_col, targ_col in zip(
            ret_qc.unitary().T, self.isometry_matrix.T, strict=True
        ):
            assert cirq.equal_up_to_global_phase(res_col, targ_col)
            if np.allclose(res_col, targ_col):
                diag_angles_radians.append(0)
            else:
                phase_difference = np.angle(np.vdot(res_col, targ_col))
                diag_angles_radians.append(phase_difference)
        if len(diag_angles_radians) < 2**self.domain_num_qubit:
            diag_angles_radians += [0] * (
                2**self.domain_num_qubit - len(diag_angles_radians)
            )

        digonal_gate_decomposed = cirq.decompose(
            cirq.DiagonalGate(diag_angles_radians)(*main_qubits),
            keep=keep_ftn_for_cirq_decompose,
        )
        return cirq.Circuit(digonal_gate_decomposed) + ret_qc


def get_extending_orthgonal_vector(
    orthogonals: List[np.ndarray], m_qubit_num: int
) -> List[np.ndarray]:
    sv_mat = np.stack(orthogonals, axis=0).conjugate()
    from scipy.linalg import null_space

    num_of_required_additions = (2**m_qubit_num) - len(orthogonals)
    assert num_of_required_additions >= 0

    ns: np.ndarray = null_space(sv_mat)

    if ns.size == 0:
        return []

    ns = ns.reshape((sv_mat.shape[1], -1))
    columns = [ns[:, i] for i in range(ns.shape[1])]
    nomralized = list()
    for column in columns[:num_of_required_additions]:
        column /= np.linalg.norm(column)
        nomralized.append(column)
        cirq.validate_normalized_state_vector(column, qid_shape=(sv_mat.shape[1],))

    assert len(nomralized) == num_of_required_additions
    for i, vec1 in enumerate(nomralized):
        for j, vec2 in enumerate(nomralized):
            inner_product = np.vdot(vec1, vec2)
            if i == j:
                assert np.isclose(inner_product, 1.0), f"Vector {i} is not normalized."
            else:
                assert np.isclose(
                    inner_product, 0.0
                ), f"Vectors {i} and {j} are not orthogonal."

    return nomralized


def to_extended_isometry(
    isometry_matrix: np.ndarray, targ_codomain_qubit_num: Optional[int] = None
) -> np.ndarray:
    # NOTE: The isometry matrix may not always fit the qubit dimensions directly
    # (e.g., a 3x5 matrix representing an isometry from 2 qubits to 3 qubits).
    # However, the Qiskit isometry decomposition function only accepts square matrices
    # with dimensions that are powers of 2. Therefore, it may be necessary to extend
    # the isometry matrix by appending orthogonal vectors to make it compatible.
    domain_qubit_num = int(np.ceil(np.log2(isometry_matrix.shape[0])))
    codomain_qubit_num = targ_codomain_qubit_num or int(
        np.ceil(np.log2(isometry_matrix.shape[1]))
    )
    iso_vecs = [isometry_matrix[:, i] for i in range(isometry_matrix.shape[1])]
    extended_iso_vecs = iso_vecs + get_extending_orthgonal_vector(
        iso_vecs, codomain_qubit_num
    )

    assert len(extended_iso_vecs) == 2**codomain_qubit_num
    extended_iso_mat = np.stack(extended_iso_vecs, axis=1)
    assert extended_iso_mat.shape == (2**domain_qubit_num, 2**codomain_qubit_num)

    return extended_iso_mat


class QiskitIsometry(IsometryBase):

    def to_quantum_circuit(
        self,
        main_qubits: List[cirq.Qid],
        force_unitary_synthesis_method: bool = False,
        **kwargs,
    ) -> cirq.Circuit:
        isometry_shape = self.isometry_matrix.shape
        assert 2**self.codomain_num_qubit == isometry_shape[0]

        if (
            isometry_shape[0] == isometry_shape[1]
            and self.domain_num_qubit == self.codomain_num_qubit
        ) or force_unitary_synthesis_method:
            unitary_to_apply = None
            if not cirq.is_unitary(self.isometry_matrix):
                unitary_to_apply = to_extended_isometry(
                    self.isometry_matrix, targ_codomain_qubit_num=len(main_qubits)
                )
            else:
                unitary_to_apply = self.isometry_matrix

            assert cirq.is_unitary(unitary_to_apply)

            # NOTE : need to recheck if this gives some bias in a fair comparison with QCLIB...
            circuit = qiskit.synthesis.qs_decomposition(
                unitary_to_apply, opt_a1=True, opt_a2=True
            )
        else:
            scheme = (
                "csd"
                if self.isometry_matrix.shape[0] // 2 == self.isometry_matrix.shape[1]
                else "ccd"
            )
            extended_isometry = to_extended_isometry(self.isometry_matrix)
            circuit = qclib.isometry.decompose(extended_isometry, scheme=scheme)

        circuit_after_transpile = transpile(
            circuit, basis_gates=["u3", "cx"], optimization_level=0
        )
        cirq_qc = qiskit2cirq(circuit_after_transpile, do_reverse=True)

        cirq_qc_ret: cirq.Circuit = cirq_qc + cirq.global_phase_operation(
            get_global_phase_match(self.isometry_matrix[:, 0], cirq_qc)
        )

        return cirq_qc_ret.transform_qubits(lambda q: main_qubits[q.x])
