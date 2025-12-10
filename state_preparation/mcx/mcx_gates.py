from ast import Call
from typing import Callable, Sequence, Union
import cirq
from state_preparation.circuit_converter import qiskit2cirq
from qiskit.synthesis import (
    synth_mcx_noaux_hp24,
    synth_mcx_noaux_v24,
    synth_mcx_1_dirty_kg24,
    synth_mcx_2_dirty_kg24,
    synth_mcx_n_dirty_i15,
)
from cirq.transformers.analytical_decompositions import (
    decompose_multi_controlled_x,
)

from state_preparation.utils import (
    keep_ftn_for_cirq_decompose,
)  # Using technique from Braceno et al.


def get_qulin_mcx_in_cirq_no_aux(num_control: int) -> cirq.Gate:
    qiskit_qc = synth_mcx_noaux_hp24(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


def get_kg_dirty_one_mcx_in_cirq_no_aux(num_control: int) -> cirq.Gate:
    qiskit_qc = synth_mcx_1_dirty_kg24(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


def get_kg_dirty_two_mcx_in_cirq_no_aux(num_control: int) -> cirq.Gate:
    qiskit_qc = synth_mcx_2_dirty_kg24(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


def get_iten_dirty_mcx_in_cirq(num_control: int) -> cirq.Gate:
    qiskit_qc = synth_mcx_n_dirty_i15(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


def get_vale_mcx_in_cirq_no_aux(num_control: int) -> cirq.Gate:
    qiskit_qc = synth_mcx_noaux_v24(num_control)
    cirq_circuit = qiskit2cirq(qiskit_qc)
    return cirq_circuit


class MCXGateBase(cirq.Gate):
    required_aux_qubits_num: Callable[[int], Union[int, None]] = ...

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
        return self.num_controls + type(self).required_aux_qubits_num(self.num_controls) + 1

    def _decompose_(self, qubits, get_mcx_in_cirq):

        assert len(qubits) == self.num_controls + self.num_aux_qubits + 1

        if self.num_aux_qubits == 0:  # no aux qubits to use

            mcx_qc: cirq.Circuit = get_mcx_in_cirq(self.num_controls)

        else:
            raise NotImplementedError("This Path is not implemented yet.")

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
    required_aux_qubits_num = lambda num_ctrl: 0

    def _num_qubits_(self) -> int:
        return self.num_controls + self

    def _decompose_(self, qubits):
        yield cirq.X(qubits[-1]).controlled_by(
            *qubits[:-1], control_values=self.control_values
        )


class QulinMCXGate(MCXGateBase):
    # Wrapper for Qulin https://dl.acm.org/doi/pdf/10.1145/3656436 for Cirq implementation
    required_aux_qubits_num = lambda num_ctrl: 0

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_qulin_mcx_in_cirq_no_aux)


class KGDirtyOneMCXGate(MCXGateBase):
    # Khattar and Gidney,
    # Rise of conditionally clean ancillae for optimizing quantum circuits arXiv:2407.17966
    required_aux_qubits_num = lambda num_ctrl: 1

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_kg_dirty_one_mcx_in_cirq_no_aux)


class KGDirtyTwoMCXGate(MCXGateBase):
    # Khattar and Gidney,
    # Rise of conditionally clean ancillae for optimizing quantum circuits arXiv:2407.17966

    required_aux_qubits_num = lambda num_ctrl: 2

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_kg_dirty_two_mcx_in_cirq_no_aux)


class ItenDirtyMCXGate(MCXGateBase):
    # Iten et. al.,
    # Quantum Circuits for Isometries, Phys. Rev. A 93, 032318 (2016), arXiv:1501.06911
    required_aux_qubits_num = lambda num_ctrl: num_ctrl - 2

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_iten_dirty_mcx_in_cirq)


class ValeMCXGate(MCXGateBase):
    required_aux_qubits_num = lambda _: 0

    def _decompose_(self, qubits):
        yield from super()._decompose_(qubits, get_vale_mcx_in_cirq_no_aux)


class CirqStandardMCXGate(MCXGateBase):
    required_aux_qubits_num = lambda _: None

    @property
    def max_aux_qubits_num(self) -> int:
        return 100 * 1000  # effectively large number

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
