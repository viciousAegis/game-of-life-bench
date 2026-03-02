from dataclasses import dataclass

import numpy as np

from .rules import LifeRule, parse_rule


ArrayBool = np.ndarray


@dataclass
class SimulationResult:
    initial_board: list[list[int]]
    board_rows: int
    board_cols: int
    rule: str
    topology: str
    max_steps: int
    steps_completed: int
    first_repeat_step: int | None
    repeat_from_step: int | None
    period: int | None
    repeated: bool
    populations: list[int]
    activity: list[int]
    frames: list[list[list[int]]]


class LifeSimulator:
    def __init__(
        self,
        rows: int,
        cols: int,
        rule: str = "B3/S23",
        topology: str = "toroidal",
    ) -> None:
        if topology != "toroidal":
            raise ValueError(f"Unsupported topology '{topology}'. Only 'toroidal' is implemented.")

        self.rows = rows
        self.cols = cols
        self.topology = topology
        self.rule = parse_rule(rule) if isinstance(rule, str) else rule

    def simulate(self, initial_board: ArrayBool | list[list[int]], max_steps: int) -> SimulationResult:
        board = np.asarray(initial_board, dtype=bool)
        if board.shape != (self.rows, self.cols):
            raise ValueError(
                f"Board shape {board.shape} does not match configured grid {(self.rows, self.cols)}."
            )

        frames = [board.astype(np.uint8).tolist()]
        populations = [int(board.sum())]
        activity = [0]

        seen_at_step: dict[bytes, int] = {board.tobytes(): 0}
        current = board.copy()
        first_repeat_step: int | None = None
        repeat_from_step: int | None = None

        for step in range(1, max_steps + 1):
            next_board = self.step(current)
            frames.append(next_board.astype(np.uint8).tolist())
            populations.append(int(next_board.sum()))
            activity.append(int(np.count_nonzero(next_board != current)))

            state_key = next_board.tobytes()
            if state_key in seen_at_step:
                first_repeat_step = step
                repeat_from_step = seen_at_step[state_key]
                current = next_board
                break

            seen_at_step[state_key] = step
            current = next_board

        steps_completed = len(frames) - 1
        period = None
        repeated = first_repeat_step is not None and repeat_from_step is not None
        if repeated:
            period = first_repeat_step - repeat_from_step

        return SimulationResult(
            initial_board=frames[0],
            board_rows=self.rows,
            board_cols=self.cols,
            rule=self.rule.notation,
            topology=self.topology,
            max_steps=max_steps,
            steps_completed=steps_completed,
            first_repeat_step=first_repeat_step,
            repeat_from_step=repeat_from_step,
            period=period,
            repeated=repeated,
            populations=populations,
            activity=activity,
            frames=frames,
        )

    def step(self, board: ArrayBool) -> ArrayBool:
        neighbors = self._count_neighbors(board)
        born = np.isin(neighbors, list(self.rule.birth)) & ~board
        survive = np.isin(neighbors, list(self.rule.survive)) & board
        return born | survive

    def _count_neighbors(self, board: ArrayBool) -> np.ndarray:
        neighbors = np.zeros_like(board, dtype=np.uint8)
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                if row_shift == 0 and col_shift == 0:
                    continue
                neighbors += np.roll(board, shift=(row_shift, col_shift), axis=(0, 1))
        return neighbors
