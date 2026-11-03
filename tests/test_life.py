import unittest
from types import SimpleNamespace

from game_of_life_bench.benchmark import BenchmarkRunner
from game_of_life_bench.main import _apply_runtime_overrides
from game_of_life_bench.main import settings as main_settings
from game_of_life_bench.models.openrouter import OpenRouterClient
from game_of_life_bench.models.openrouter import OpenRouterGenerationError
from game_of_life_bench.models.openrouter import (
    _build_prompt,
    _extract_board_from_reasoning,
    _extract_choice_debug,
    _extract_json_blob,
    _extract_response_metadata,
    _is_local_server_url,
)
from game_of_life_bench.scoring import evaluate_board, validate_board
from game_of_life_bench.leaderboard import build_leaderboard


class LifeEvaluationTests(unittest.TestCase):
    def test_validate_board_rejects_over_limit(self) -> None:
        board = [
            [1, 1],
            [1, 0],
        ]
        with self.assertRaises(ValueError):
            validate_board(board, rows=2, cols=2, max_live_fraction=0.5)

    def test_block_is_still_life_with_period_one(self) -> None:
        board = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        result = evaluate_board(
            board=board,
            rows=4,
            cols=4,
            rule="B3/S23",
            topology="toroidal",
            max_steps=10,
            max_live_fraction=1.0,
        )
        self.assertEqual(result.score, 1)
        self.assertEqual(result.simulation.period, 1)

    def test_blinker_has_period_two(self) -> None:
        board = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = evaluate_board(
            board=board,
            rows=5,
            cols=5,
            rule="B3/S23",
            topology="toroidal",
            max_steps=10,
            max_live_fraction=1.0,
        )
        self.assertEqual(result.score, 2)
        self.assertEqual(result.simulation.period, 2)

    def test_extract_json_blob_from_fenced_response(self) -> None:
        text = '```json\n{"board":[[0,1],[1,0]]}\n```'
        self.assertEqual(_extract_json_blob(text), '{"board":[[0,1],[1,0]]}')

    def test_generation_error_detail_contains_debug_fields(self) -> None:
        error = OpenRouterGenerationError(
            "Model response could not be parsed into a board.",
            model="test/model",
            prompt="prompt text",
            raw_response="not json",
            status_code=400,
            provider_error="bad request",
            response_metadata={"id": "generation-123"},
        )
        self.assertEqual(error.to_detail()["model"], "test/model")
        self.assertEqual(error.to_detail()["raw_response"], "not json")
        self.assertEqual(error.to_detail()["response_metadata"]["id"], "generation-123")

    def test_prompt_contains_example_board(self) -> None:
        prompt = _build_prompt(rows=4, cols=4, max_live_cells=8, rule="B3/S23", topology="toroidal")
        self.assertIn("Example response format:", prompt)
        self.assertIn('"board"', prompt)

    def test_extract_choice_debug_contains_message_payload(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "", "reasoning": "hidden"},
                }
            ]
        }
        debug = _extract_choice_debug(payload)
        self.assertIn('"finish_reason": "stop"', debug)
        self.assertIn('"reasoning": "hidden"', debug)

    def test_recover_last_exact_binary_board_from_reasoning(self) -> None:
        reasoning = (
            'An intermediate board was [[1, 1], [1, 1]]. '
            'Final answer: {"board": [[0, 1], [1, 0]]}'
        )
        board = _extract_board_from_reasoning(reasoning, rows=2, cols=2)
        self.assertEqual(board, [[0, 1], [1, 0]])

    def test_reasoning_recovery_rejects_wrong_shape(self) -> None:
        board = _extract_board_from_reasoning("[[0, 1, 0]]", rows=2, cols=2)
        self.assertIsNone(board)

    def test_extract_response_metadata_captures_reasoning_and_usage(self) -> None:
        payload = {
            "id": "resp_123",
            "model": "demo/model",
            "provider": "OpenRouter",
            "created": 1234567890,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"board":[[0,1],[1,0]]}',
                        "reasoning": "tested a few motifs",
                    },
                }
            ],
        }
        metadata = _extract_response_metadata(payload)
        self.assertEqual(metadata["usage"]["total_tokens"], 30)
        self.assertEqual(metadata["reasoning"]["reasoning_text"], "tested a few motifs")

    def test_build_leaderboard_picks_best_submission_per_model(self) -> None:
        leaderboard = build_leaderboard(
            [
                {
                    "benchmark_id": "bench-1",
                    "models": [
                        {
                            "model": "alpha",
                            "average_score": 6.0,
                            "submission_score": 8,
                            "best_score": 9,
                            "best_board": [[1]],
                            "best_seed": 2,
                            "best_run_id": "run-1",
                            "submission_board": [[1]],
                            "submission_seed": 2,
                            "submission_run_id": "run-1",
                            "median_score": 6.5,
                            "worst_score": 3,
                            "trial_results": [
                                {
                                    "seed": 2,
                                    "response_metadata": {
                                        "reasoning": {"reasoning_text": "visible"},
                                        "usage": {"total_tokens": 42, "completion_tokens": 12, "cost": 0.1},
                                    },
                                }
                            ],
                            "trials": 10,
                        }
                    ],
                },
                {
                    "benchmark_id": "bench-2",
                    "models": [
                        {
                            "model": "alpha",
                            "average_score": 7.5,
                            "submission_score": 7,
                            "best_score": 8,
                            "best_board": [[0]],
                            "best_seed": 1,
                            "best_run_id": "run-2",
                            "submission_board": [[0]],
                            "submission_seed": 1,
                            "submission_run_id": "run-2",
                            "median_score": 7.0,
                            "worst_score": 5,
                            "trial_results": [],
                            "trials": 10,
                        }
                    ],
                },
            ]
        )
        self.assertEqual(leaderboard[0]["model"], "alpha")
        self.assertEqual(leaderboard[0]["submission_score"], 8)
        self.assertEqual(leaderboard[0]["best_run_id"], "run-1")
        self.assertTrue(leaderboard[0]["visible_reasoning"])
        self.assertEqual(leaderboard[0]["submission_total_tokens"], 42)
        self.assertEqual(leaderboard[0]["total_cost"], 0.1)
        self.assertEqual(leaderboard[0]["avg_output_tokens"], 12.0)

    def test_apply_runtime_overrides_updates_openrouter_base_url(self) -> None:
        original_base_url = main_settings.openrouter_base_url
        try:
            override_url = "http://127.0.0.1:8001/v1"
            _apply_runtime_overrides(openrouter_base_url=override_url)
            self.assertEqual(main_settings.openrouter_base_url, override_url)
        finally:
            main_settings.openrouter_base_url = original_base_url

    def test_local_server_url_detection_accepts_loopback_hosts(self) -> None:
        self.assertTrue(_is_local_server_url("http://127.0.0.1:8000/v1"))
        self.assertTrue(_is_local_server_url("http://localhost:8000/v1"))
        self.assertFalse(_is_local_server_url("https://openrouter.ai/api/v1"))

    def test_openrouter_client_allows_missing_api_key_for_local_server(self) -> None:
        local_settings = SimpleNamespace(
            openrouter_api_key=None,
            openrouter_base_url="http://127.0.0.1:8000/v1",
        )
        client = OpenRouterClient(local_settings)
        self.assertEqual(client._settings.openrouter_base_url, "http://127.0.0.1:8000/v1")

    def test_openrouter_client_requires_api_key_for_non_local_server(self) -> None:
        remote_settings = SimpleNamespace(
            openrouter_api_key=None,
            openrouter_base_url="https://openrouter.ai/api/v1",
        )
        with self.assertRaises(ValueError):
            OpenRouterClient(remote_settings)


class BenchmarkRunnerFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_scores_board_recovered_from_reasoning(self) -> None:
        settings = SimpleNamespace(
            openrouter_api_key=None,
            openrouter_base_url="http://127.0.0.1:8000/v1",
            openrouter_timeout_seconds=1.0,
            openrouter_site_url=None,
            openrouter_site_name="test",
        )
        client = OpenRouterClient(settings)

        async def fake_request_board(**kwargs):
            return {
                "id": "generation-123",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "",
                            "reasoning": 'Final board: {"board": [[0, 1], [1, 0]]}',
                        },
                    }
                ],
            }

        client._request_board = fake_request_board  # type: ignore[method-assign]
        result = await client.generate_board(
            model="demo/model",
            rows=2,
            cols=2,
            max_live_cells=2,
            rule="B3/S23",
            topology="toroidal",
            seed=0,
        )

        self.assertEqual(result.board, [[0, 1], [1, 0]])
        self.assertEqual(result.response_metadata["board_recovered_from"], "reasoning")

    async def test_failed_trial_persists_provider_output_before_reraising(self) -> None:
        settings = SimpleNamespace(grid_rows=2, grid_cols=2, benchmark_concurrency=1)
        saved_payloads = []
        storage = SimpleNamespace(
            save_run=lambda payload: saved_payloads.append(payload) or "failed-run-id"
        )
        runner = BenchmarkRunner(settings, storage)

        class FailingClient:
            async def generate_board(self, **kwargs):
                raise OpenRouterGenerationError(
                    "output was reasoning-only",
                    model=kwargs["model"],
                    prompt="full prompt",
                    raw_response='{"finish_reason":"length","message":{"reasoning":"work"}}',
                    response_metadata={"id": "generation-123", "finish_reason": "length"},
                    response_payload={"choices": [{"message": {"reasoning": "work"}}]},
                )

        with self.assertRaises(OpenRouterGenerationError):
            await runner._run_trial(
                client=FailingClient(),
                model="demo/model",
                seed=0,
                trials_per_model=1,
                max_live_cells=2,
                rule="B3/S23",
                topology="toroidal",
                max_steps=10,
                max_live_fraction=0.5,
            )

        self.assertEqual(len(saved_payloads), 1)
        payload = saved_payloads[0]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["model_run"]["prompt"], "full prompt")
        self.assertIn("reasoning", payload["model_run"]["raw_response"])
        self.assertEqual(payload["model_run"]["response_metadata"]["id"], "generation-123")
        provider_response = payload["model_run"]["provider_response"]
        self.assertEqual(provider_response["choices"][0]["message"]["reasoning"], "work")

    async def test_partial_trial_failures_do_not_fail_model(self) -> None:
        settings = SimpleNamespace(
            grid_rows=2,
            grid_cols=2,
            benchmark_concurrency=2,
        )
        storage = SimpleNamespace(save_run=lambda payload: "run-id")
        runner = BenchmarkRunner(settings, storage)

        async def fake_run_trial(**kwargs):
            seed = kwargs["seed"]
            if seed == 1:
                raise OpenRouterGenerationError(
                    "bad output",
                    model="demo/model",
                    prompt="prompt",
                )
            return SimpleNamespace(
                seed=seed,
                score=seed + 2,
                live_cells=1,
                first_repeat_step=seed + 2,
                period=1,
                board=[[0, 1], [1, 0]],
                raw_response="{}",
                response_metadata={},
                run_id=f"run-{seed}",
            )

        runner._run_trial = fake_run_trial  # type: ignore[method-assign]
        result = await runner._run_model(
            client=object(),
            model="demo/model",
            trials_per_model=3,
            rule="B3/S23",
            topology="toroidal",
            max_steps=10,
            max_live_fraction=1.0,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.trials, 2)
        self.assertEqual(result.failed_trials, 1)
        self.assertEqual(result.requested_trials, 3)
        self.assertEqual(result.submission_score, 4)


if __name__ == "__main__":
    unittest.main()
