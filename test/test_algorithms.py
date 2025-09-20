import cirq
import numpy as np
import pytest

from state_preparation.algorithms import XYZ, IsometryBased, LowRankStatePrep, UCGEBased
from state_preparation.results import StatePreparationResult
from state_preparation.state_samplers import get_random_sparse_state, get_random_state


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
