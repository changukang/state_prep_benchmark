import logging

import cirq
import numpy as np
import pytest

from state_preparation.algorithms import (
    XYZ,
    IsometryBased,
    LowRankStatePrep,
    PivotStatePrep,
    SandwichedPermutation,
    UCGEBased,
)
from state_preparation.gates.mcx.types import SelectiveOptimalMCXGate
from state_preparation.results import StatePreparationResult
from state_preparation.state_samplers import get_random_sparse_state, get_random_state


@pytest.mark.parametrize(
    "prepare_engine",
    [
        LowRankStatePrep(),
        PivotStatePrep(),
        UCGEBased(),
        IsometryBased(),
    ],
)
def test_prepare_engines(prepare_engine):
    logging.getLogger("qiskit.passmanager.base_tasks").setLevel(logging.WARNING)
    for seed in range(10):
        if isinstance(prepare_engine, PivotStatePrep):
            # PivotStatePrep is optimized for sparse states, producing quantum circuits with fewer moments,
            # making it easier to validate the final state vector.
            sv = get_random_sparse_state(num_qubit=9, sparsity=20, seed=seed)
        else:
            sv = get_random_state(num_qubit=5, seed=seed)

        state_prep_res: StatePreparationResult = prepare_engine.run(sv)
        qc = state_prep_res.cirq_circuit
        sv_from_result = cirq.final_state_vector(qc, dtype=np.complex128)
        assert cirq.equal_up_to_global_phase(sv, sv_from_result)
        print("Number of CNOTs:", state_prep_res.num_cnot)


def test_xyz():
    for seed in range(5):
        # NOTE : due to duration time of the algorithm, we use a sparse state
        sv = get_random_sparse_state(
            num_qubit=9, sparsity=30, seed=seed, complex=False, uniform=True
        )
        prepare = XYZ()
        state_prep_res = prepare.run(sv)
        qc = state_prep_res.cirq_circuit
        sv_from_result = cirq.final_state_vector(qc, dtype=np.complex128)
        assert cirq.equal_up_to_global_phase(sv, sv_from_result)


def test_sandwiched_permutation():
    qclib_low_rank_state_prep = LowRankStatePrep().run
    for num_qubit in [9]:
        for sparsity in [100, 200, 500]:
            print(
                f"Test Sandwiched Permutation : num_qubit={num_qubit}, sparsity={sparsity}"
            )
            sv = get_random_sparse_state(
                num_qubit=num_qubit, sparsity=sparsity, seed=2025
            )

            res = SandwichedPermutation(
                sub_state_preparation=qclib_low_rank_state_prep,
                # mcx_gate_type=QulinMCXGate,
                # mcx_gate_type=ValeMCXGate,
                # mcx_gate_type=CirqStandardMCXGate,
                # mcx_gate_type=ItenDirtyMCXGate,
                mcx_gate_type=SelectiveOptimalMCXGate,
            ).run(sv)
            assert cirq.equal_up_to_global_phase(
                sv, cirq.final_state_vector(res.cirq_circuit, dtype=np.complex128)
            )
            print("On Num qubit :", num_qubit, "sparsity:", sparsity)
            print("NUM CNOTS:", res.num_cnot)
