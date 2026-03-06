import logging

from state_preparation.benchmark.lcu_prep_states import LcuPrepStatesBenchmark
from state_preparation.utils import sparsity
import numpy as np

logging.basicConfig(level=logging.INFO)

# each cid is from CCBDB
# NOTE : following lcu prep states must have 630 non-zero terms,
# Ref: https://arxiv.org/pdf/2007.11624
cid_list = [
    "157350",
    "16211014",
    "5460607",
    "62714",
    "6397184",
]


def test():

    for cid in cid_list:
        sv = LcuPrepStatesBenchmark(cid)()
        nonzero_amps = sv[sv != 0]

        assert nonzero_amps.size > 0
        assert np.isclose(np.linalg.norm(sv), 1.0)
        assert sparsity(sv) == 630
