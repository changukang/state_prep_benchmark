import random
from typing import List

from state_preparation.algorithms import IsometryBased, LowRankStatePrep
from state_preparation.runner import run_state_preparations
from state_preparation.state_samplers import get_random_sparse_state
from state_preparation.statevector import StateVectorWithInfo


def test_run_state_preparations():
    state_vectors = []
    for seed in range(30):
        sparsity = random.randint(1, 6)
        sv = get_random_sparse_state(num_qubit=7, sparsity=sparsity**2, seed=seed)
        state_vectors.append(sv)
    state_preparations = [LowRankStatePrep(), IsometryBased()]
    run_state_preparations(
        state_vectors, state_preparations, result_items=["num_cnot", "depth"]
    )


def test_run_state_preparations_with_sv_info():
    state_vectors: List[StateVectorWithInfo] = []
    for seed in range(30):
        sparsity = random.randint(1, 6)
        sv = get_random_sparse_state(num_qubit=7, sparsity=sparsity**2, seed=seed)
        state_vectors.append(
            StateVectorWithInfo(
                state_vector=sv,
                info={"name": f"seed={seed}/num_qubit={7}", "sparsity": sparsity**2},
            )
        )
    state_preparations = [LowRankStatePrep(), IsometryBased()]
    run_state_preparations(
        state_vectors, state_preparations, result_items=["num_cnot", "depth"]
    )
