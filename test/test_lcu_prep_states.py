import logging

from state_preparation.benchmark.lcu_prep_states import LcuPrepStatesBenchmark

logging.basicConfig(level=logging.INFO)


def test():
    LcuPrepStatesBenchmark("157350")
