import logging
from typing import Callable, List, Optional

import numpy as np
from rich.console import Console
from rich.table import Table

from .algorithms import StatePreparation
from .benchmark.states import BalancedHammingWeight
from .results import AVAILABLE_RESULT_ITEMS, StatePreparationResult

logger = logging.getLogger(__name__)

AVAILABLE_BENCHMARKS = {"balanced_hamming_weight": BalancedHammingWeight}


def color_generator():
    colors = ["blue", "green", "magenta", "blue", "cyan"]
    return iter(colors)


def run_state_preparations(
    state_vectors: List[np.ndarray],
    state_preparations: List[StatePreparation],
    result_items: Optional[List[str]] = None,
) -> None:
    if result_items:
        for item in result_items:
            if item not in AVAILABLE_RESULT_ITEMS:
                raise ValueError(f"Invalid Result Item {item}")
    else:
        result_items = AVAILABLE_RESULT_ITEMS

    table = Table(title="State Preparation Result")
    table.add_column("State Vector", justify="right", style="black", no_wrap=True)

    color_gen = color_generator()
    for state_preparation in state_preparations:
        color = next(color_gen)
        for item in result_items:
            table.add_column(
                f"{item}({state_preparation.name})",
                justify="right",
                style=color,
                no_wrap=True,
            )
    for idx, state_vector in enumerate(state_vectors):
        row_data = list()
        row_data.append(str(idx))
        for state_preparation in state_preparations:
            state_prep_result = state_preparation.run(state_vector)
            row_data += state_prep_result._export_to_row_data(result_items)
        table.add_row(*row_data)

    console = Console()
    console.print(table)


def run_benchmark(
    bechmark_id: str, num_qubit_range: str, qclib: bool = False, low_rank: bool = False
):
    """Runs state preparation for benchmark quantum states.

    Args:
        benchmark_id (str): The identifier for the benchmark state vector.
            Currently, only `balanced_hamming_weight` is supported.
        num_qubit_range (str): The range of qubit numbers for the benchmark state.
            Must be in the format `X-Y`, where `X` and `Y` are integers such that `X < Y`.
            For example, specifying `4-6` will run state preparation for quantum states
            with 4, 5, and 6 qubits.
    """
    raise NotImplementedError
    # if bechmark_id not in AVAILABLE_BENCHMARKS:
    #     raise ValueError(
    #         f"Invalid benchmakr {bechmark_id}. "
    #         f"Must be one of {list(AVAILABLE_BENCHMARKS.keys())}"
    #     )
    # def parse_qubit_range(qubit_range_str: str) -> Tuple[int, int]:
    #     split = qubit_range_str.split("-")
    #     return (int(split[0]), int(split[1]))


def run_random_state():
    raise NotImplementedError
