from typing import Sequence, Union

import cirq
from cirq.transformers.analytical_decompositions import decompose_multi_controlled_x
from qiskit.synthesis import (
    synth_mcx_1_dirty_kg24,
    synth_mcx_2_dirty_kg24,
    synth_mcx_n_dirty_i15,
    synth_mcx_noaux_hp24,
    synth_mcx_noaux_v24,
)

from state_preparation.circuit_converter import qiskit2cirq
from state_preparation.utils import (  # Using technique from Braceno et al.
    keep_ftn_for_cirq_decompose,
)


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


def get_qulin_mcx_in_cirq_no_aux(num_control: int) -> cirq.Circuit:
    qiskit_qc = synth_mcx_noaux_hp24(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


def get_kg_dirty_one_mcx_in_cirq_no_aux(num_control: int) -> cirq.Circuit:
    qiskit_qc = synth_mcx_1_dirty_kg24(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


def get_kg_dirty_two_mcx_in_cirq_no_aux(num_control: int) -> cirq.Circuit:
    qiskit_qc = synth_mcx_2_dirty_kg24(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


def get_iten_dirty_mcx_in_cirq(num_control: int) -> cirq.Circuit:
    qiskit_qc = synth_mcx_n_dirty_i15(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


def get_vale_mcx_in_cirq_no_aux(num_control: int) -> cirq.Circuit:
    qiskit_qc = synth_mcx_noaux_v24(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    decomposed = cirq.decompose(cirq_circuit, keep=keep_ftn_for_cirq_decompose)
    return cirq.Circuit(decomposed)


class MCXGateBase(cirq.Gate):

    @classmethod
    def required_aux_qubits_num(cls, num_ctrl: int) -> Union[int, None]:
        raise NotImplementedError

    @classmethod
    def max_valid_aux_qubits_num(
        cls, num_ctrl: int, num_available_aux_qubits: int
    ) -> int:
        raise NotImplementedError

    def __init__(
        self,
        num_controls: int,
        control_values: Sequence[int] = None,
        num_aux_qubits: int = 0,
    ):
        super().__init__()
        self.num_controls = num_controls
        self.control_values = (
            control_values if control_values is not None else [1] * num_controls
        )

        self.num_aux_qubits = num_aux_qubits

    def _flip_controls(self, qubits):
        for i, cv in enumerate(self.control_values):
            if cv == 0:
                yield cirq.X(qubits[i])

    def _num_qubits_(self) -> int:
        return self.num_controls + self.num_aux_qubits + 1

    @classmethod
    def from_available_aux_qubits(
        cls,
        main_qubits: Sequence[cirq.LineQubit],
        available_aux_qubits: Sequence[cirq.LineQubit],
        control_values: Sequence[int] = None,
    ) -> cirq.GateOperation:
        """
        main_qubits will be composed as controls and target
        """
        num_control = len(main_qubits) - 1

        free_qubits = None
        if (
            required_aux_num := cls.required_aux_qubits_num(num_ctrl=num_control)
        ) is not None:
            if len(available_aux_qubits) < required_aux_num:
                raise NotEnoughAuxQubits(
                    required=required_aux_num,
                    available=len(available_aux_qubits),
                    num_controls=num_control,
                )

            free_qubits = available_aux_qubits[:required_aux_num]

        else:
            num_max_valid = cls.max_valid_aux_qubits_num(
                num_ctrl=num_control,
                num_available_aux_qubits=len(available_aux_qubits),
            )
            assert num_max_valid is not None
            free_qubits = available_aux_qubits[:num_max_valid]

        mcx_gate = cls(
            num_controls=num_control,
            control_values=control_values,
            num_aux_qubits=len(free_qubits),
        )
        return mcx_gate(*(main_qubits + free_qubits))

    def _decompose_(self, qubits, get_mcx_in_cirq):

        assert len(qubits) == self.num_controls + self.num_aux_qubits + 1

        if self.num_aux_qubits == 0:  # no aux qubits to use
            mcx_qc: cirq.Circuit = get_mcx_in_cirq(self.num_controls)
        else:
            mcx_qc: cirq.Circuit = get_mcx_in_cirq(self.num_controls)

        original_qubits = sorted(mcx_qc.all_qubits(), key=lambda q: q.x)

        qubit_transformed = mcx_qc.transform_qubits(
            lambda q: qubits[original_qubits.index(q)]
        )

        for op in qubit_transformed.all_operations():
            if isinstance(op.gate, cirq.Gate) and op.gate.num_qubits() == 2:
                assert op.gate == cirq.CX

        yield from self._flip_controls(qubits)
        yield qubit_transformed
        yield from self._flip_controls(qubits)

    def _circuit_diagram_info_(self, args):
        return ["C"] * self.num_controls + ["X"]


class CanonMCXGate(MCXGateBase):

    @classmethod
    def required_aux_qubits_num(cls, num_ctrl):
        return 0

    def _decompose_(self, qubits):
        yield cirq.X(qubits[-1]).controlled_by(
            *qubits[:-1], control_values=self.control_values
        )


class QulinMCXGate(MCXGateBase):
    # Wrapper for Qulin https://dl.acm.org/doi/pdf/10.1145/3656436 for Cirq implementation
    @classmethod
    def required_aux_qubits_num(cls, num_ctrl):
        return 0

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_qulin_mcx_in_cirq_no_aux)


class KGDirtyOneMCXGate(MCXGateBase):
    # Khattar and Gidney,
    # Rise of conditionally clean ancillae for optimizing quantum circuits arXiv:2407.17966
    @classmethod
    def required_aux_qubits_num(cls, num_ctrl):
        return 1

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_kg_dirty_one_mcx_in_cirq_no_aux)


class KGDirtyTwoMCXGate(MCXGateBase):
    # Khattar and Gidney,
    # Rise of conditionally clean ancillae for optimizing quantum circuits arXiv:2407.17966

    @classmethod
    def required_aux_qubits_num(cls, num_ctrl):
        return 2

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_kg_dirty_two_mcx_in_cirq_no_aux)


class ItenDirtyMCXGate(MCXGateBase):
    # Iten et. al.,
    # Quantum Circuits for Isometries, Phys. Rev. A 93, 032318 (2016), arXiv:1501.06911
    @classmethod
    def required_aux_qubits_num(cls, num_ctrl):
        if num_ctrl >= 4:
            return num_ctrl - 2
        else:
            return 0

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_iten_dirty_mcx_in_cirq)


class ValeMCXGate(MCXGateBase):
    @classmethod
    def required_aux_qubits_num(cls, num_ctrl):
        return 0

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_vale_mcx_in_cirq_no_aux)


class CirqStandardMCXGate(MCXGateBase):

    @classmethod
    def required_aux_qubits_num(cls, num_ctrl: int) -> None:
        return None

    @classmethod
    def max_valid_aux_qubits_num(
        cls, num_ctrl: int, num_available_aux_qubits: int
    ) -> int:
        return num_available_aux_qubits

    def _decompose_(self, qubits):
        qubits = list(qubits)
        controls = qubits[: self.num_controls]
        target = qubits[self.num_controls]
        aux_qubits = qubits[self.num_controls + 1 :]

        yield from self._flip_controls(qubits)
        # optimize following?
        yield from cirq.decompose(
            decompose_multi_controlled_x(
                controls=controls, target=target, free_qubits=aux_qubits
            ),
            keep=keep_ftn_for_cirq_decompose,
        )
        yield from self._flip_controls(qubits)


oracle = {
    (4, 0): (ItenDirtyMCXGate, 14),
    (4, 1): (ItenDirtyMCXGate, 14),
    (5, 0): (ValeMCXGate, 36),
    (4, 2): (ItenDirtyMCXGate, 14),
    (5, 1): (KGDirtyOneMCXGate, 30),
    (6, 0): (QulinMCXGate, 96),
    (4, 3): (ItenDirtyMCXGate, 14),
    (5, 2): (ItenDirtyMCXGate, 26),
    (6, 1): (KGDirtyOneMCXGate, 42),
    (7, 0): (QulinMCXGate, 136),
    (4, 4): (ItenDirtyMCXGate, 14),
    (5, 3): (ItenDirtyMCXGate, 26),
    (6, 2): (KGDirtyTwoMCXGate, 42),
    (7, 1): (KGDirtyOneMCXGate, 54),
    (8, 0): (QulinMCXGate, 192),
    (4, 5): (ItenDirtyMCXGate, 14),
    (5, 4): (ItenDirtyMCXGate, 26),
    (6, 3): (ItenDirtyMCXGate, 34),
    (7, 2): (KGDirtyTwoMCXGate, 54),
    (8, 1): (KGDirtyOneMCXGate, 66),
    (9, 0): (QulinMCXGate, 264),
    (4, 6): (ItenDirtyMCXGate, 14),
    (5, 5): (ItenDirtyMCXGate, 26),
    (6, 4): (ItenDirtyMCXGate, 34),
    (7, 3): (KGDirtyTwoMCXGate, 54),
    (8, 2): (KGDirtyTwoMCXGate, 66),
    (9, 1): (KGDirtyOneMCXGate, 78),
    (10, 0): (QulinMCXGate, 344),
    (4, 7): (ItenDirtyMCXGate, 14),
    (5, 6): (ItenDirtyMCXGate, 26),
    (6, 5): (ItenDirtyMCXGate, 34),
    (7, 4): (ItenDirtyMCXGate, 42),
    (8, 3): (KGDirtyTwoMCXGate, 66),
    (9, 2): (KGDirtyTwoMCXGate, 78),
    (10, 1): (KGDirtyOneMCXGate, 90),
    (11, 0): (QulinMCXGate, 464),
    (4, 8): (ItenDirtyMCXGate, 14),
    (5, 7): (ItenDirtyMCXGate, 26),
    (6, 6): (ItenDirtyMCXGate, 34),
    (7, 5): (ItenDirtyMCXGate, 42),
    (8, 4): (KGDirtyTwoMCXGate, 66),
    (9, 3): (KGDirtyTwoMCXGate, 78),
    (10, 2): (KGDirtyTwoMCXGate, 90),
    (11, 1): (KGDirtyOneMCXGate, 102),
    (12, 0): (QulinMCXGate, 576),
}


class OptimalToffoli(MCXGateBase):
    def __init__(self, control_values=None):
        super().__init__(2, control_values, 0)

    def num_qubits(self) -> int:
        return 3

    def _unitary_(self):
        return cirq.unitary(cirq.CCX)

    def _decompose_(self, qubits):
        # from https://en.wikipedia.org/wiki/Toffoli_gate
        control1, control2, target = qubits
        yield from self._flip_controls(qubits)
        yield (cirq.H(target))
        yield (cirq.CX(control2, target))
        yield ((cirq.T**-1)(target))
        yield (cirq.CX(control1, target))
        yield (cirq.T(target))
        yield (cirq.CX(control2, target))
        yield ((cirq.T**-1)(target))
        yield (cirq.CX(control1, target))
        yield (cirq.T(control2))
        yield (cirq.T(target))
        yield (cirq.H(target))
        yield (cirq.CX(control1, control2))
        yield ((cirq.T**-1)(control2))
        yield (cirq.T(control1))
        yield (cirq.CX(control1, control2))
        yield from self._flip_controls(qubits)


class SelectiveOptimalMCXGate(MCXGateBase):
    """
    This gate selects the optimal MCX gate implementation based on the number of
    auxiliary qubits available.
    """

    def __init__(
        self,
        selected_mcx_gate_type: MCXGateBase,
        num_controls: int,
        control_values: Sequence[int] = None,
        num_aux_qubits: int = 0,
    ):
        super().__init__(
            num_controls=num_controls,
            control_values=control_values,
            num_aux_qubits=num_aux_qubits,
        )
        self.selected_mcx_gate_type = selected_mcx_gate_type

    @classmethod
    def from_available_aux_qubits(
        cls,
        main_qubits: Sequence[cirq.LineQubit],
        available_aux_qubits: Sequence[cirq.LineQubit],
        control_values: Sequence[int] = None,
    ) -> Union[cirq.GateOperation, cirq.Circuit]:
        selected_mcx_gate_type = oracle.get(
            (len(main_qubits), len(available_aux_qubits))
        )

        assert (
            len(main_qubits) - 1 == len(control_values)
            if control_values is not None
            else True
        )

        if len(main_qubits) == 2:
            return cirq.Circuit(
                cirq.X(main_qubits[1]).controlled_by(
                    main_qubits[0], control_values=control_values
                )
            )
        if len(main_qubits) == 3:
            return cirq.Circuit(
                cirq.decompose_once(
                    OptimalToffoli(control_values=control_values)(*(main_qubits))
                )
            )

        if selected_mcx_gate_type is None:
            raise NotImplementedError(
                f"No optimal MCX gate found for {len(main_qubits)-1} control qubits "
                f"and {len(available_aux_qubits)} auxiliary qubits."
            )

        selected_mcx_gate_type = selected_mcx_gate_type[0]

        assert issubclass(selected_mcx_gate_type, MCXGateBase)

        return selected_mcx_gate_type.from_available_aux_qubits(
            main_qubits=main_qubits,
            available_aux_qubits=available_aux_qubits,
            control_values=control_values,
        )

    def _decompose_(self, qubits):
        yield from self.selected_mcx_gate_type._decompose_(qubits)
