from typing import Callable

import cirq
import cirq.circuits
import numpy as np

from state_preparation.results import StatePreparationResult


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
        I = np.eye(dim, dtype=np.complex128)
        return I + (np.exp(1j * self.phi) - 1.0) * np.outer(
            self.v, np.conjugate(self.v)
        )

    def to_quantum_circuit(
        self, state_preparation: Callable[[np.ndarray], StatePreparationResult]
    ) -> cirq.Circuit:
        pass


class HouseHolderBasedMapping(HouseHolder):

    def __init__(self, v: np.ndarray, w: np.ndarray, strict: bool = False):

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
