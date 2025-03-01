from state_preparation.state_samplers import (get_random_sparse_state,
                                              get_random_state)


def test_random_state():
    for seed in range(10):
        get_random_state(num_qubit=5, seed=seed)


def test_random_sparse_statea():
    for seed in range(10):
        get_random_sparse_state(num_qubit=5, sparsity=3, seed=seed)
