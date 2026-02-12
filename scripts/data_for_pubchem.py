import logging
from state_preparation.benchmark.lcu_prep_states import LcuPrepStatesBenchmark
import json



cid_to_multiplicity = {
    # "157350": 2,
    # "16211014": 1,
    # "5460607": 3,
    # "62714": 1,
    # "6397184": 1,
    "139073": 1,
    "123164": 3,
    "123329": 2,
    "139760": 2,
    "962": 1,
}

def main():
    
    for cid, multiplicity in cid_to_multiplicity.items():
        logging.info(f"Running LCU prep state benchmark for CID {cid}")
        res = LcuPrepStatesBenchmark._get_required_data_for_init(cid, geometry=None,  multiplicity=multiplicity, charge= 0)

        # Save the result as a list of dictionaries in JSON format

        output_data = {
            "cid": cid,
            "multiplicity": multiplicity,
            "result": res
        }

        output_file = f"output_{cid}.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

        logging.info(f"Data for CID {cid} saved to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()