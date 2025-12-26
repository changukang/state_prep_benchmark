from typing import Sequence, Union

import cirq


class NotEnoughAuxQubits(Exception):
    """Exception raised when there are not enough auxiliary qubits available."""

    def __init__(self, required: int, available: int, num_controls: int):
        self.required = required
        self.available = available
        self.num_controls = num_controls
        super().__init__(
            f"Not enough auxiliary qubits. Required: {required}, Available: {available}, "
            f"for {num_controls} control qubits."
        )


class MCUGateBase(cirq.Gate):
    @classmethod
    def required_aux_qubits_num(cls, num_ctrl: int) -> Union[int, None]:
        raise NotImplementedError

    @classmethod
    def max_valid_aux_qubits_num(
        cls, num_ctrl: int, num_available_aux_qubits: int
    ) -> int:
        raise NotImplementedError

    def _flip_controls(self, qubits):
        for i, cv in enumerate(self.control_values):
            if cv == 0:
                yield cirq.X(qubits[i])

    def _num_qubits_(self) -> int:
        return self.num_controls + self.num_aux_qubits + 1

    @classmethod
    def _select_free_qubits(
        cls,
        num_ctrl: int,
        available_aux_qubits: Sequence[cirq.LineQubit],
    ) -> Sequence[cirq.LineQubit]:
        required_aux_num = cls.required_aux_qubits_num(num_ctrl=num_ctrl)

        if required_aux_num is not None:
            if len(available_aux_qubits) < required_aux_num:
                raise NotEnoughAuxQubits(
                    required=required_aux_num,
                    available=len(available_aux_qubits),
                    num_controls=num_ctrl,
                )
            return available_aux_qubits[:required_aux_num]

        num_max_valid = cls.max_valid_aux_qubits_num(
            num_ctrl=num_ctrl,
            num_available_aux_qubits=len(available_aux_qubits),
        )
        assert num_max_valid is not None
        return available_aux_qubits[:num_max_valid]

    def __init__(
        self,
        sub_gate: cirq.Gate,
        num_controls: int,
        control_values: Sequence[int] = None,
        num_aux_qubits: int = 0,
    ):
        self.num_controls = num_controls
        self.control_values = (
            control_values if control_values is not None else [1] * num_controls
        )

        self.num_aux_qubits = num_aux_qubits
        self.sub_gate = sub_gate
