import logging
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from .algorithms import StatePreparationBase
from .results import AVAILABLE_RESULT_ITEMS, StatePreparationResultStatistics
from .state_samplers import get_random_sparse_state

logger = logging.getLogger(__name__)


def color_generator():
    colors = ["blue", "green", "magenta", "blue", "cyan"]
    return iter(colors)


def run_random_sparse_benchmark(
    num_qubit: int,
    sparsity: int,
    state_preparations: List[StatePreparationBase],
    num_sample: int = 10,
    seed: int = 2025,
    result_items: Optional[List[str]] = AVAILABLE_RESULT_ITEMS,
):

    table = Table(title="State Preparation Result in Statistics")
    table.add_column("Idx", justify="right", style="black", no_wrap=True)
    color_gen = color_generator()
    state_vectors = [
        get_random_sparse_state(num_qubit=num_qubit, sparsity=sparsity, seed=seed + i)
        for i in range(num_sample)
    ]

    for state_preparation in state_preparations:
        color = next(color_gen)
        for item in result_items:
            table.add_column(
                f"{item}(Avg.)({state_preparation.name})",
                justify="right",
                style=color,
                no_wrap=True,
            )

    row_data = list()
    row_data.append("1")

    for state_preparation in state_preparations:

        result_statistics = StatePreparationResultStatistics(
            id=f"In {state_preparation.name}",
            results=[
                state_preparation.run(state_vector) for state_vector in state_vectors
            ],
        )
        row_data += result_statistics._export_to_row_data(result_items)
    table.add_row(*row_data)
    console = Console()
    console.print("\n", table)
