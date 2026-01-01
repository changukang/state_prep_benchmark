from itertools import product

import cirq
import numpy as np

from state_preparation.algorithms import PivotStatePrep
from state_preparation.gates.mcp.types import CanonMCPhaseGate
from state_preparation.householder.types import HouseHolder, HouseHolderBasedMapping
from state_preparation.state_samplers import get_random_sparse_state, get_random_state


def test_house_holder_based_mapping():

    for seed in range(10):
        v = get_random_state(5, seed)
        w = get_random_sparse_state(5, 10, seed + 100)

        hh_based = HouseHolderBasedMapping(v, w, strict=False)
        mapped_result = hh_based.matrix @ v
        theta = np.pi - np.angle(np.vdot(v, w))

        assert np.isclose(np.exp(1j * theta) * w, mapped_result).all()


def test_house_holder_based_mapping_strict():

    for seed in range(10):
        v = get_random_state(5, seed)
        w = get_random_sparse_state(5, 10, seed + 100)

        hh_based = HouseHolderBasedMapping(v, w, strict=True)
        mapped_result = hh_based.matrix @ v

        assert np.isclose(w, mapped_result).all()


def test_house_holder_qc():
    for num_qubit in [4, 5]:
        for seed, phi in product(range(5), [np.pi, np.pi / 3]):
            v = get_random_state(num_qubit, seed)

            hh = HouseHolder(state_vector=v, phi=phi)
            qc = hh.to_quantum_circuit(
                state_preparation=PivotStatePrep().run,
                main_qubits=cirq.LineQubit.range(num_qubit),
                available_aux_qubits=[],
                mcp_gate=CanonMCPhaseGate,
            )

            res = cirq.final_state_vector(
                qc, qubit_order=cirq.LineQubit.range(num_qubit), initial_state=v
            )

            assert np.allclose(res, np.exp(1j * phi) * v)
            assert np.allclose(hh.matrix, cirq.unitary(qc))


def test_house_holder_based_map_qc():
    for num_qubit in [4, 5]:
        for seed in range(5):
            v = get_random_state(num_qubit, seed)
            w = get_random_state(num_qubit, seed + 100 * 2)

            hh_based_map = HouseHolderBasedMapping(v=v, w=w, strict=True)

            qc = hh_based_map.to_quantum_circuit(
                state_preparation=PivotStatePrep().run,
                main_qubits=cirq.LineQubit.range(num_qubit),
                available_aux_qubits=[],
                mcp_gate=CanonMCPhaseGate,
            )

            res = qc.final_state_vector(
                initial_state=v, qubit_order=cirq.LineQubit.range(num_qubit)
            )

            assert np.allclose(res, w)
