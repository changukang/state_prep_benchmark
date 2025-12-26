import numpy as np

from state_preparation.householder.types import HouseHolderBasedMapping
from state_preparation.state_samplers import get_random_sparse_state, get_random_state


def test_house_holder_based_mapping():

    for seed in range(10):
        v = get_random_state(5, seed)
        w = get_random_sparse_state(5, 10, seed + 100)

        hh_based = HouseHolderBasedMapping(v, w, strict=False)
        mapped_result = hh_based.matrix @ v
        theta = np.pi - np.angle(np.vdot(v, w))

        assert np.isclose(np.exp(1j * theta) * w, mapped_result).all()


def test_house_holder_based_mapping_strict():

    for seed in range(10):
        v = get_random_state(5, seed)
        w = get_random_sparse_state(5, 10, seed + 100)

        hh_based = HouseHolderBasedMapping(v, w, strict=True)
        mapped_result = hh_based.matrix @ v

        assert np.isclose(w, mapped_result).all()
