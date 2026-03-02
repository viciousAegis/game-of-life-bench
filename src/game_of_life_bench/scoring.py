from dataclasses import dataclass

import numpy as np

from .life import LifeSimulator, SimulationResult


@dataclass
class EvaluationResult:
    score: int
    live_cells: int
    live_fraction: float
    max_live_cells: int
    simulation: SimulationResult


def validate_board(board: list[list[int]], rows: int, cols: int, max_live_fraction: float) -> np.ndarray:
    if len(board) != rows:
        raise ValueError(f"Expected {rows} rows, received {len(board)}.")

    normalized_rows: list[list[int]] = []
    for row_index, row in enumerate(board):
        if len(row) != cols:
            raise ValueError(f"Expected row {row_index} to have {cols} columns, received {len(row)}.")

        normalized_row: list[int] = []
        for value in row:
            if value not in (0, 1, False, True):
                raise ValueError("Board cells must be 0 or 1.")
            normalized_row.append(int(value))
        normalized_rows.append(normalized_row)

    board_array = np.asarray(normalized_rows, dtype=bool)
    live_cells = int(board_array.sum())
    max_live_cells = int(rows * cols * max_live_fraction)
    if live_cells > max_live_cells:
        raise ValueError(f"Board has {live_cells} live cells, exceeding cap of {max_live_cells}.")

    return board_array


def evaluate_board(
    board: list[list[int]],
    rows: int,
    cols: int,
    rule: str,
    topology: str,
    max_steps: int,
    max_live_fraction: float,
) -> EvaluationResult:
    board_array = validate_board(board, rows=rows, cols=cols, max_live_fraction=max_live_fraction)
    simulator = LifeSimulator(rows=rows, cols=cols, rule=rule, topology=topology)
    simulation = simulator.simulate(board_array, max_steps=max_steps)
    live_cells = int(board_array.sum())
    max_live_cells = int(rows * cols * max_live_fraction)
    score = simulation.first_repeat_step if simulation.first_repeat_step is not None else max_steps
    return EvaluationResult(
        score=score,
        live_cells=live_cells,
        live_fraction=live_cells / (rows * cols),
        max_live_cells=max_live_cells,
        simulation=simulation,
    )
