from state_preparation.state_samplers import (get_random_basis_state_vectors,
                                              get_random_sparse_state,
                                              get_random_state)


def test_random_state():
    for seed in range(10):
        get_random_state(num_qubit=5, seed=seed)


def test_random_sparse_statea():
    for seed in range(10):
        sv = get_random_sparse_state(num_qubit=5, sparsity=3, seed=seed)
        assert (sv != 0).sum() == 3


def test_random_basis_state_vectors():
    for seed in range(10):
        get_random_basis_state_vectors(num_qubit=2, num_basis=2, seed=seed)
