from state_preparation.state_prep_algorithms import IsometryBased, LowRankStatePrep
from state_preparation.state_samplers import get_random_state


def test_low_rank():
    for seed in range(10):
        sv = get_random_state(num_qubit=5, seed=seed)
        prepare = LowRankStatePrep()
        state_prep_res = prepare.run(sv)
        print(state_prep_res)


def test_isometry():
    for seed in range(10):
        sv = get_random_state(num_qubit=5, seed=seed)
        prepare = IsometryBased()
        state_prep_res = prepare.run(sv)
        print(state_prep_res)
