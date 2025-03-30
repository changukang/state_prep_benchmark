from state_preparation.algorithms import IsometryBased, LowRankStatePrep
from state_preparation.state_samplers import get_random_state
from state_preparation.results import StatePreparationResult
import cirq
import numpy as np

def test_low_rank():
    for seed in range(10):
        sv = get_random_state(num_qubit=5, seed=seed)
        prepare = LowRankStatePrep()
        state_prep_res : StatePreparationResult = prepare.run(sv)
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
