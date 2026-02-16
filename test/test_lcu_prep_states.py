import logging

from state_preparation.benchmark.lcu_prep_states import LcuPrepStatesBenchmark

logging.basicConfig(level=logging.INFO)

# each multiplicty is from CCBDB
cid_to_multiplicity = {
    "157350": 2,
    "16211014": 1,
    "5460607": 3,
    "62714": 1,
    "6397184": 1,
    "139073": 1,
    "123164": 3,
    "123329": 2,
    "139760": 2,
    "962": 1,
}


def test():

    for cid, multiplicity in cid_to_multiplicity.items():
        print(f"Running LCU prep state benchmark for CID {cid}")
        LcuPrepStatesBenchmark(cid, multiplicity=multiplicity)
