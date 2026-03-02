from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable

from .config import Settings
from .models.openrouter import OpenRouterClient
from .scoring import EvaluationResult, evaluate_board


@dataclass
class BenchmarkTrialResult:
    seed: int
    score: int
    live_cells: int
    first_repeat_step: int | None
    period: int | None
    board: list[list[int]]
    raw_response: str
    response_metadata: dict[str, Any]
    run_id: str


@dataclass
class BenchmarkModelResult:
    model: str
    trials: int
    requested_trials: int
    failed_trials: int
    average_score: float
    median_score: float
    best_score: int
    worst_score: int
    best_seed: int
    best_board: list[list[int]]
    best_run_id: str
    submission_score: int
    submission_seed: int
    submission_board: list[list[int]]
    submission_run_id: str
    trial_results: list[BenchmarkTrialResult]


@dataclass
class BenchmarkResult:
    benchmark_id: str
    models: list[BenchmarkModelResult]
    grid_rows: int
    grid_cols: int
    rule: str
    topology: str
    max_steps: int
    max_live_fraction: float
    trials_per_model: int


class BenchmarkRunner:
    def __init__(
        self,
        settings: Settings,
        storage,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._progress_callback = progress_callback
        self._semaphore = asyncio.Semaphore(max(1, settings.benchmark_concurrency))

    async def run(
        self,
        models: list[str],
        trials_per_model: int,
        rule: str,
        topology: str,
        max_steps: int,
        max_live_fraction: float,
    ) -> BenchmarkResult:
        client = OpenRouterClient(self._settings)
        model_results: list[BenchmarkModelResult] = []
        for model in models:
            self._emit(f"starting model {model}")
            model_result = await self._run_model(
                client=client,
                model=model,
                trials_per_model=trials_per_model,
                rule=rule,
                topology=topology,
                max_steps=max_steps,
                max_live_fraction=max_live_fraction,
            )
            if model_result is None:
                self._emit(f"skipping model {model}: all {trials_per_model} trials failed")
            else:
                model_results.append(model_result)
            self._emit(f"finished model {model}")

        if not model_results:
            raise ValueError("All benchmark trials failed for all requested models.")

        model_results.sort(key=lambda item: (-item.submission_score, -item.average_score, item.model))
        payload = {
            "models": model_results,
            "grid_rows": self._settings.grid_rows,
            "grid_cols": self._settings.grid_cols,
            "rule": rule,
            "topology": topology,
            "max_steps": max_steps,
            "max_live_fraction": max_live_fraction,
            "trials_per_model": trials_per_model,
        }
        benchmark_id = self._storage.save_benchmark(payload)
        return BenchmarkResult(
            benchmark_id=benchmark_id,
            models=model_results,
            grid_rows=self._settings.grid_rows,
            grid_cols=self._settings.grid_cols,
            rule=rule,
            topology=topology,
            max_steps=max_steps,
            max_live_fraction=max_live_fraction,
            trials_per_model=trials_per_model,
        )

    async def _run_model(
        self,
        client: OpenRouterClient,
        model: str,
        trials_per_model: int,
        rule: str,
        topology: str,
        max_steps: int,
        max_live_fraction: float,
    ) -> BenchmarkModelResult | None:
        max_live_cells = int(self._settings.grid_rows * self._settings.grid_cols * max_live_fraction)
        tasks = [
            asyncio.create_task(
                self._run_trial(
                    client=client,
                    model=model,
                    seed=seed,
                    trials_per_model=trials_per_model,
                    max_live_cells=max_live_cells,
                    rule=rule,
                    topology=topology,
                    max_steps=max_steps,
                    max_live_fraction=max_live_fraction,
                )
            )
            for seed in range(trials_per_model)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        trial_results: list[BenchmarkTrialResult] = []
        failed_trials = 0
        for seed, result in enumerate(raw_results):
            if isinstance(result, Exception):
                failed_trials += 1
                self._emit(f"  seed {seed + 1}/{trials_per_model}: failed ({result})")
                continue
            trial_results.append(result)

        if not trial_results:
            return None

        trial_results.sort(key=lambda item: item.seed)

        scores = [trial.score for trial in trial_results]
        best_trial = max(trial_results, key=lambda item: (item.score, -item.seed))
        sorted_scores = sorted(scores)
        mid = len(sorted_scores) // 2
        median_score = (
            float(sorted_scores[mid])
            if len(sorted_scores) % 2 == 1
            else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        )
        return BenchmarkModelResult(
            model=model,
            trials=len(trial_results),
            requested_trials=trials_per_model,
            failed_trials=failed_trials,
            average_score=mean(scores),
            median_score=median_score,
            best_score=best_trial.score,
            worst_score=min(scores),
            best_seed=best_trial.seed,
            best_board=best_trial.board,
            best_run_id=best_trial.run_id,
            submission_score=best_trial.score,
            submission_seed=best_trial.seed,
            submission_board=best_trial.board,
            submission_run_id=best_trial.run_id,
            trial_results=trial_results,
        )

    async def _run_trial(
        self,
        client: OpenRouterClient,
        model: str,
        seed: int,
        trials_per_model: int,
        max_live_cells: int,
        rule: str,
        topology: str,
        max_steps: int,
        max_live_fraction: float,
    ) -> BenchmarkTrialResult:
        async with self._semaphore:
            self._emit(f"  seed {seed + 1}/{trials_per_model}: requesting board")
            model_run = await client.generate_board(
                model=model,
                rows=self._settings.grid_rows,
                cols=self._settings.grid_cols,
                max_live_cells=max_live_cells,
                rule=rule,
                topology=topology,
                seed=seed,
            )
            self._emit(f"  seed {seed + 1}/{trials_per_model}: evaluating board")
            evaluation = evaluate_board(
                board=model_run.board,
                rows=self._settings.grid_rows,
                cols=self._settings.grid_cols,
                rule=rule,
                topology=topology,
                max_steps=max_steps,
                max_live_fraction=max_live_fraction,
            )
            run_payload = {
                "source": "benchmark",
                "request": {
                    "model": model,
                    "rule": rule,
                    "topology": topology,
                    "max_steps": max_steps,
                    "max_live_fraction": max_live_fraction,
                    "seed": seed,
                },
                "model_run": {
                    "model": model_run.model,
                    "prompt": model_run.prompt,
                    "raw_response": model_run.raw_response,
                    "board": model_run.board,
                    "response_metadata": model_run.response_metadata,
                    "seed": seed,
                },
                "evaluation": evaluation,
            }
            run_id = self._storage.save_run(run_payload)
            self._emit(
                f"  seed {seed + 1}/{trials_per_model}: score={evaluation.score} run_id={run_id}"
            )
            return _trial_result(
                seed,
                evaluation,
                model_run.raw_response,
                model_run.response_metadata,
                run_id,
            )

    def _emit(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)


def _trial_result(
    seed: int,
    evaluation: EvaluationResult,
    raw_response: str,
    response_metadata: dict[str, Any],
    run_id: str,
) -> BenchmarkTrialResult:
    return BenchmarkTrialResult(
        seed=seed,
        score=evaluation.score,
        live_cells=evaluation.live_cells,
        first_repeat_step=evaluation.simulation.first_repeat_step,
        period=evaluation.simulation.period,
        board=evaluation.simulation.initial_board,
        raw_response=raw_response,
        response_metadata=response_metadata,
        run_id=run_id,
    )
