from mimetypes import init

import cirq
import numpy as np

from state_preparation.gates.mcp.types import CanonMCPhaseGate


def test_mcp():

    phi = np.pi
    mcp_gate = CanonMCPhaseGate(phi, num_controls=2)
    qc = cirq.Circuit(cirq.decompose_once(mcp_gate(*cirq.LineQubit.range(3))))

    oracle = cirq.unitary(
        (cirq.ZPowGate(exponent=phi / np.pi))(cirq.LineQubit(2)).controlled_by(
            cirq.LineQubit(0), cirq.LineQubit(1)
        )
    )
    qc_res = cirq.unitary(qc)

    assert np.allclose(qc_res, oracle)
