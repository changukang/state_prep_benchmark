import logging
from typing import List, Optional, Union

import numpy as np
from rich.console import Console
from rich.table import Table

from .algorithms import InvalidStatePreparationResult, StatePreparationBase
from .benchmark.states import BalancedHammingWeight
from .results import AVAILABLE_RESULT_ITEMS
from .statevector import StateVectorWithInfo

logger = logging.getLogger(__name__)

AVAILABLE_BENCHMARKS = {"balanced_hamming_weight": BalancedHammingWeight}


def color_generator():
    colors = ["blue", "green", "magenta", "blue", "cyan"]
    return iter(colors)


def run_state_preparations(
    state_vectors: Union[List[np.ndarray], List[StateVectorWithInfo]],
    state_preparations: List[StatePreparationBase],
    result_items: Optional[List[str]] = None,
) -> None:
    if result_items:
        for item in result_items:
            if item not in AVAILABLE_RESULT_ITEMS:
                raise ValueError(f"Invalid Result Item {item}")
    else:
        result_items = AVAILABLE_RESULT_ITEMS

    if not (
        all(isinstance(state_vector, np.ndarray) for state_vector in state_vectors)
        or all(
            isinstance(state_vector, StateVectorWithInfo)
            for state_vector in state_vectors
        )
    ):
        raise ValueError(
            "State vectors must be given in all np.ndarray or `StateVectorWithInfo` "
        )

    table = Table(title="State Preparation Result")
    table.add_column("Idx", justify="right", style="black", no_wrap=True)

    if isinstance(state_vectors[0], StateVectorWithInfo):
        sv_info_items = [
            state_vector.get_info_items() for state_vector in state_vectors
        ]
        if any(sv_info_items[0] != x for x in sv_info_items):
            raise ValueError("State vector info items must be all same")
        for sv_info_item in state_vectors[0].get_info_items():
            table.add_column(sv_info_item, justify="right", style="black", no_wrap=True)

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
        if isinstance(state_vector, StateVectorWithInfo):
            row_data += state_vector.get_data()

        for state_preparation in state_preparations:
            data_to_append = list()
            try:
                state_prep_result = state_preparation.run(
                    state_vector
                    if isinstance(state_vector, np.ndarray)
                    else state_vector.state_vector
                )
                data_to_append = state_prep_result._export_to_row_data(result_items)
            except InvalidStatePreparationResult:
                data_to_append = ["NA"] * len(result_items)
            assert data_to_append
            row_data += state_prep_result._export_to_row_data(result_items)
        table.add_row(*row_data)

    console = Console()
    console.print("\n", table)


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
