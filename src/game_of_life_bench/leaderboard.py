from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_leaderboard(benchmarks: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for benchmark in benchmarks:
        for model_entry in benchmark.get("models", []):
            by_model.setdefault(model_entry["model"], []).append(
                {
                    "benchmark_id": benchmark["benchmark_id"],
                    **model_entry,
                }
            )

    leaderboard: list[dict] = []
    for model, entries in by_model.items():
        for entry in entries:
            entry.setdefault("submission_score", entry.get("best_score", 0))
            entry.setdefault("median_score", entry.get("average_score", 0.0))
            entry.setdefault("worst_score", entry.get("best_score", 0))
            entry.setdefault("submission_board", entry.get("best_board", []))
            entry.setdefault("submission_seed", entry.get("best_seed", 0))
            entry.setdefault("submission_run_id", entry.get("best_run_id", ""))
            entry.setdefault("trial_results", [])
            entry.setdefault("submission_cost", None)
            entry.setdefault("total_cost", None)
            entry.setdefault("avg_output_tokens", None)

        entries.sort(key=lambda item: (-item["submission_score"], -item["average_score"], item["benchmark_id"]))
        best_entry = entries[0]
        submission_trial = find_trial_result(best_entry, best_entry["submission_seed"])
        submission_metadata = submission_trial.get("response_metadata", {}) if submission_trial else {}
        reasoning_text = (
            submission_metadata.get("reasoning", {}).get("reasoning_text")
            if isinstance(submission_metadata.get("reasoning"), dict)
            else None
        )
        usage = submission_metadata.get("usage", {}) if isinstance(submission_metadata.get("usage"), dict) else {}
        total_cost, avg_output_tokens = aggregate_trial_usage(best_entry)
        if total_cost is None:
            total_cost = best_entry.get("total_cost")
        if total_cost is None:
            total_cost = best_entry.get("submission_cost")
        leaderboard.append(
            {
                "model": model,
                "benchmarks_run": len(entries),
                "submission_score": best_entry["submission_score"],
                "best_average_score": best_entry["average_score"],
                "median_score": best_entry["median_score"],
                "floor_score": best_entry["worst_score"],
                "best_score": best_entry["best_score"],
                "best_board": best_entry["submission_board"],
                "best_seed": best_entry["submission_seed"],
                "best_run_id": best_entry["submission_run_id"],
                "best_benchmark_id": best_entry["benchmark_id"],
                "latest_average_score": entries[0]["average_score"],
                "trial_count": best_entry["trials"],
                "visible_reasoning": bool(reasoning_text),
                "submission_total_tokens": usage.get("total_tokens"),
                "total_cost": total_cost,
                "avg_output_tokens": avg_output_tokens,
            }
        )

    leaderboard.sort(key=lambda item: (-item["submission_score"], -item["best_average_score"], item["model"]))
    for index, entry in enumerate(leaderboard, start=1):
        entry["rank"] = index
    return leaderboard


def build_leaderboard_payload(benchmarks: list[dict]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_count": len(benchmarks),
        "leaderboard": build_leaderboard(benchmarks),
    }


def find_trial_result(model_entry: dict, seed: int) -> dict | None:
    for trial in model_entry.get("trial_results", []):
        if trial.get("seed") == seed:
            return trial
    return None


def aggregate_trial_usage(model_entry: dict) -> tuple[float | None, float | None]:
    costs: list[float] = []
    completion_tokens: list[float] = []
    for trial in model_entry.get("trial_results", []):
        response_metadata = trial.get("response_metadata", {})
        if not isinstance(response_metadata, dict):
            continue
        usage = response_metadata.get("usage", {})
        if not isinstance(usage, dict):
            continue
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            costs.append(float(cost))
        out_tokens = usage.get("completion_tokens")
        if isinstance(out_tokens, (int, float)):
            completion_tokens.append(float(out_tokens))

    total_cost = sum(costs) if costs else None
    avg_output_tokens = (sum(completion_tokens) / len(completion_tokens)) if completion_tokens else None
    return total_cost, avg_output_tokens
