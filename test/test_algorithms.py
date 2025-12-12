import cirq
import numpy as np

from state_preparation.algorithms import (XYZ, IsometryBased, LowRankStatePrep,
                                          SandwichedPermutation, UCGEBased)
from state_preparation.mcx.mcx_gates import SelectiveOptimalMCXGate
from state_preparation.results import StatePreparationResult
from state_preparation.state_samplers import (get_random_sparse_state,
                                              get_random_state)


def test_low_rank():
    for seed in range(10):
        sv = get_random_state(num_qubit=5, seed=seed)
        prepare = LowRankStatePrep()
        state_prep_res: StatePreparationResult = prepare.run(sv)
        qc = state_prep_res.cirq_circuit
        sv_from_result = cirq.final_state_vector(qc, dtype=np.complex128)
        assert cirq.equal_up_to_global_phase(sv, sv_from_result)


def test_ucge():
    for seed in range(10):
        sv = get_random_state(num_qubit=5, seed=seed)
        prepare = UCGEBased()
        state_prep_res: StatePreparationResult = prepare.run(sv)
        qc = state_prep_res.cirq_circuit
        sv_from_result = cirq.final_state_vector(qc, dtype=np.complex128)
        assert cirq.equal_up_to_global_phase(sv, sv_from_result)


def test_isometry():
    for seed in range(10):
        sv = get_random_state(num_qubit=5, seed=seed)
        prepare = IsometryBased()
        state_prep_res = prepare.run(sv)
        qc = state_prep_res.cirq_circuit
        sv_from_result = cirq.final_state_vector(qc, dtype=np.complex128)
        assert cirq.equal_up_to_global_phase(sv, sv_from_result)


def test_xyz():
    for seed in range(5):
        # NOTE : due to duration time of the algorithm, we use a sparse state
        sv = get_random_sparse_state(num_qubit=8, sparsity=5, seed=seed, complex=False)
        prepare = XYZ()
        state_prep_res = prepare.run(sv)
        qc = state_prep_res.cirq_circuit
        sv_from_result = cirq.final_state_vector(qc, dtype=np.complex128)
        assert cirq.equal_up_to_global_phase(sv, sv_from_result)


def test_sandwiched_permutation():
    qclib_low_rank_state_prep = LowRankStatePrep().run
    for num_qubit in [5]:
        for sparsity in [5, 10, 15]:
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
            print("finished state preparation circuit.")
            assert cirq.equal_up_to_global_phase(
                sv, cirq.final_state_vector(res.cirq_circuit, dtype=np.complex128)
            )
            print("On Num qubit :", num_qubit, "sparsity:", sparsity)
            print("NUM CNOTS:", res.num_cnot)
