from state_preparation.algorithms import IsometryBased, LowRankStatePrep
from state_preparation.runner import run_state_preparations
from state_preparation.state_samplers import get_random_state, get_random_sparse_state


def test_run_state_preparations():
    state_vectors = [
        get_random_sparse_state(num_qubit=6, sparsity=13, seed=seed)
        for seed in range(5)
    ]
    state_preparations = [LowRankStatePrep(), IsometryBased()]
    run_state_preparations(
        state_vectors, state_preparations, result_items=["num_cnot", "depth"]
    )
