from ast import Not
from curses.ascii import ctrl
from typing import Sequence, Union

import cirq
import numpy as np
from cirq.transformers.analytical_decompositions.controlled_gate_decomposition import (
    _decompose_su,
    decompose_multi_controlled_rotation,
)

from state_preparation.gates.utils import MCUGateBase


class MCPhaseGateBase(MCUGateBase):

    def __init__(
        self,
        phi: float,
        num_controls,
        control_values=None,
        num_aux_qubits=0,
        phase_on_zero: bool = False,
    ):

        self.phi = phi
        # Results a gate with e^{iφ} phase.
        if phase_on_zero:
            sub_gate = cirq.MatrixGate(matrix=np.array([[np.exp(1j * phi), 0], [0, 1]]))
        else:
            sub_gate = cirq.ZPowGate(exponent=phi / np.pi)
        super().__init__(
            sub_gate,
            num_controls,
            control_values,
            num_aux_qubits,
        )

    def _decompose_(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_available_aux_qubits(
        cls,
        phi: float,
        main_qubits: Sequence[cirq.LineQubit],
        available_aux_qubits: Sequence[cirq.LineQubit],
        control_values: Sequence[int] = None,
        phase_on_zero: bool = False,
    ) -> cirq.GateOperation:
        """
        main_qubits will be composed as controls and target
        """
        num_control = len(main_qubits) - 1
        free_qubits = cls._select_free_qubits(
            num_ctrl=num_control,
            available_aux_qubits=available_aux_qubits,
        )

        mcx_gate = cls(
            phi=phi,
            num_controls=num_control,
            control_values=control_values,
            num_aux_qubits=len(free_qubits),
            phase_on_zero=phase_on_zero,
        )

        return mcx_gate(*(main_qubits + free_qubits))


class CanonMCPhaseGate(MCPhaseGateBase):

    @classmethod
    def required_aux_qubits_num(cls, num_ctrl: int) -> Union[int, None]:
        return 0

    @classmethod
    def max_valid_aux_qubits_num(
        cls, num_ctrl: int, num_available_aux_qubits: int
    ) -> int:
        return 0

    def _decompose_(self, qubits):
        ctrl_qubits = qubits[: self.num_controls]
        target_qubit = qubits[self.num_controls]

        yield from self._flip_controls(ctrl_qubits)
        yield from decompose_multi_controlled_rotation(
            cirq.unitary(self.sub_gate), ctrl_qubits, target_qubit
        )
        yield from self._flip_controls(ctrl_qubits)

    def _circuit_diagram_info_(self, args):
        return [
            "NC" if cv == 0 else "C"
            for cv in (self.control_values or [1] * self.num_controls)
        ] + [f"CanonP({self.phi:.2f})"]
