import logging
from typing import Callable, Sequence, Type

import cirq
import cirq.circuits
import numpy as np

from state_preparation.gates.mcp.types import MCPhaseGateBase
from state_preparation.gates.mcx.types import SelectiveOptimalMCXGate
from state_preparation.results import StatePreparationResult
from state_preparation.utils import num_cnot_for_cirq_circuit

logger = logging.getLogger(__name__)


class HouseHolder:

    def __init__(self, state_vector: np.ndarray, phi: float = np.pi):

        cirq.validate_normalized_state_vector(
            state_vector, qid_shape=state_vector.shape
        )

        self.v = state_vector
        self.phi = phi

    @property
    def matrix(self) -> np.ndarray:
        dim = self.v.shape[0]
        iden = np.eye(dim, dtype=np.complex128)
        return iden + (np.exp(1j * self.phi) - 1.0) * np.outer(
            self.v, np.conjugate(self.v)
        )

    def to_quantum_circuit(
        self,
        state_preparation: Callable[[np.ndarray], StatePreparationResult],
        main_qubits: Sequence[cirq.Qid],
        available_aux_qubits: Sequence[cirq.Qid],
        mcp_gate: Type[MCPhaseGateBase],
        mcx_gate: Type[MCPhaseGateBase] = SelectiveOptimalMCXGate,
    ) -> cirq.Circuit:
        qc = cirq.Circuit()
        assert len(main_qubits) == int(np.log2(self.v.shape[0]))

        if np.allclose(self.v, [1] + [0] * (self.v.shape[0] - 1)):
            sub_prep_qc = cirq.Circuit()
        else:
            sub_prep_qc = state_preparation(self.v).cirq_circuit

        logger.info(
            f"HouseHolder state preparation circuit #CNOT: {num_cnot_for_cirq_circuit(sub_prep_qc)} "
        )
        qc += sub_prep_qc**-1
        if self.phi == np.pi:
            qc.append(cirq.X(main_qubits[-1]))
            qc.append(cirq.H(main_qubits[-1]))
            qc.append(
                mcx_gate.from_available_aux_qubits(
                    main_qubits=main_qubits,
                    available_aux_qubits=available_aux_qubits,
                    control_values=[0] * (len(main_qubits) - 1),
                )
            )
            qc.append(cirq.H(main_qubits[-1]))
            qc.append(cirq.X(main_qubits[-1]))

        else:
            mcp_gate_op = mcp_gate.from_available_aux_qubits(
                phi=self.phi,
                main_qubits=main_qubits,
                available_aux_qubits=available_aux_qubits,
                control_values=[0] * (len(main_qubits) - 1),
                phase_on_zero=True,
            )
            qc.append(cirq.decompose_once(mcp_gate_op))

        qc += sub_prep_qc

        return qc


class HouseHolderBasedMapping(HouseHolder):

    def __init__(self, v: np.ndarray, w: np.ndarray, strict: bool = False):

        if np.allclose(v, w):
            raise ValueError("v and w must not be equal")
        # v, w must be normalized state vectors
        cirq.validate_normalized_state_vector(v, qid_shape=v.shape)
        cirq.validate_normalized_state_vector(w, qid_shape=w.shape)

        vdot_vw = np.vdot(v, w)

        if strict:
            ephi = (vdot_vw - 1) / (1 - np.conjugate(vdot_vw))
            phi = np.angle(ephi)
            u = v - w
            norm_u = np.linalg.norm(u)
            u = u / norm_u

            # A householder reflection maps |v> to |w>
            super().__init__(state_vector=u, phi=phi)
        else:
            # θ = π − arg(<v|w>)
            vdot_vw = np.vdot(v, w)
            if np.isclose(vdot_vw, 0.0):
                theta = 0.0
            else:
                theta = np.pi - np.angle(vdot_vw)

            # |u> = (|v> − e^{iθ}|w>) / || |v> − e^{iθ}|w> ||
            u = v - np.exp(1j * theta) * w
            norm_u = np.linalg.norm(u)

            u = u / norm_u

            # A householder reflection maps |v> to  e^{iθ}|w>.
            # where the θ is defined as above.
            super().__init__(state_vector=u)
