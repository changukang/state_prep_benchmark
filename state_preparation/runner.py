import logging
from typing import List, Union

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
    result_items: List[str] = AVAILABLE_RESULT_ITEMS,
) -> None:
    if result_items:
        for item in result_items:
            if item not in AVAILABLE_RESULT_ITEMS:
                raise ValueError(f"Invalid Result Item {item}")

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
            state_vector.get_info_items() for state_vector in state_vectors  # type: ignore
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
            except InvalidStatePreparationResult as e: 
                logger.warning(
                    f"State preparation failed for {state_preparation.name} by {e}"
                )
                data_to_append = ["N/A"] * len(result_items)
            assert data_to_append
            row_data += data_to_append
        table.add_row(*row_data)

    console = Console()
    console.print("\n", table)
