import logging

import cirq
import numpy as np

from state_preparation.algorithms import PivotStatePrep
from state_preparation.gates.mcp.types import CanonMCPhaseGate
from state_preparation.isometry import HoulseHolderBasedDenseIsometry, QiskitIsometry
from state_preparation.utils import num_cnot_for_cirq_circuit


def test_hh_based_isometry():
    logging.getLogger("qiskit.passmanager.base_tasks").setLevel(logging.WARNING)

    num_qubit = 3
    random_matrix = cirq.testing.random_unitary(2**num_qubit, random_state=2025)

    test_hh_based_isometry = HoulseHolderBasedDenseIsometry(
        state_preparation=PivotStatePrep().run,
        mcp_gate=CanonMCPhaseGate,
    )
    qc = test_hh_based_isometry.to_quantum_circuit(
        isometry_matrix=random_matrix,
        main_qubits=cirq.LineQubit.range(num_qubit),
        aux_qubits=[],
    )

    for i in range(4):
        basis_vector = np.zeros(2**num_qubit, dtype=np.complex128)
        basis_vector[i] = 1.0

        res = qc.final_state_vector(
            initial_state=basis_vector, qubit_order=cirq.LineQubit.range(num_qubit)
        )

        expected = random_matrix @ basis_vector

        assert cirq.equal_up_to_global_phase(res, expected, atol=1e-8)


def test_qiskit_based_isometry():
    logging.getLogger("qiskit.passmanager.base_tasks").setLevel(logging.WARNING)

    num_qubit = 3
    random_matrix = cirq.testing.random_unitary(2**num_qubit, random_state=2025)

    for num_cols in [2, 5, 8]:
        test_hh_based_isometry = QiskitIsometry(force_unitary_synthesis_method=True)
        qc = test_hh_based_isometry.to_quantum_circuit(
            isometry_matrix=random_matrix[:, :num_cols],
            main_qubits=cirq.LineQubit.range(num_qubit),
            aux_qubits=[],
        )

        for i in range(num_cols):
            basis_vector = np.zeros(2**num_qubit, dtype=np.complex128)
            basis_vector[i] = 1.0

            res = qc.final_state_vector(
                initial_state=basis_vector, qubit_order=cirq.LineQubit.range(num_qubit)
            )

            expected = random_matrix @ basis_vector

            assert cirq.equal_up_to_global_phase(res, expected, atol=1e-8)
