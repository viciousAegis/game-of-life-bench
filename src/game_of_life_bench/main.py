import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import BenchmarkRunner
from .config import settings
from .leaderboard import build_leaderboard, build_leaderboard_payload
from .storage import RunStorage


def main() -> None:
    parser = argparse.ArgumentParser(prog="game-of-life-bench")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default=settings.app_host)
    serve_parser.add_argument("--port", type=int, default=settings.app_port)
    serve_parser.add_argument(
        "--server-url",
        dest="openrouter_base_url",
        default=None,
        help="Override the OpenRouter-compatible API base URL for this process.",
    )

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--models", nargs="+", required=True)
    benchmark_parser.add_argument("--trials", type=int, default=settings.benchmark_trials)
    benchmark_parser.add_argument("--rule", default=settings.rule)
    benchmark_parser.add_argument("--topology", default=settings.topology)
    benchmark_parser.add_argument("--max-steps", type=int, default=settings.max_steps)
    benchmark_parser.add_argument("--max-live-fraction", type=float, default=settings.max_live_fraction)
    benchmark_parser.add_argument("--concurrency", type=int, default=settings.benchmark_concurrency)
    benchmark_parser.add_argument(
        "--server-url",
        dest="openrouter_base_url",
        default=None,
        help="Override the OpenRouter-compatible API base URL for this process.",
    )

    leaderboard_parser = subparsers.add_parser("leaderboard")
    leaderboard_parser.add_argument("--json", action="store_true")
    leaderboard_parser.add_argument("--out", type=Path, default=Path("gameoflifebench/leaderboard.json"))

    args = parser.parse_args()
    if args.command in (None, "serve"):
        _apply_runtime_overrides(openrouter_base_url=getattr(args, "openrouter_base_url", None))
        _serve(getattr(args, "host", settings.app_host), getattr(args, "port", settings.app_port))
        return

    if args.command == "benchmark":
        _apply_runtime_overrides(openrouter_base_url=args.openrouter_base_url)
        asyncio.run(
            _run_benchmark(
                models=args.models,
                trials=args.trials,
                rule=args.rule,
                topology=args.topology,
                max_steps=args.max_steps,
                max_live_fraction=args.max_live_fraction,
                concurrency=args.concurrency,
            )
        )
        return

    if args.command == "leaderboard":
        _print_leaderboard(as_json=args.json, out_path=args.out)
        return


def _apply_runtime_overrides(*, openrouter_base_url: str | None) -> None:
    if openrouter_base_url:
        settings.openrouter_base_url = openrouter_base_url


def _serve(host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Web dependencies are not installed. Run `uv sync --extra web`.") from exc

    uvicorn.run(
        "game_of_life_bench.web.app:app",
        host=host,
        port=port,
        reload=False,
        factory=False,
    )


async def _run_benchmark(
    models: list[str],
    trials: int,
    rule: str,
    topology: str,
    max_steps: int,
    max_live_fraction: float,
    concurrency: int,
) -> None:
    settings.benchmark_concurrency = max(1, concurrency)
    storage = RunStorage(settings.runs_dir, settings.benchmarks_dir)
    print(f"[{_timestamp()}] benchmark run started", flush=True)
    print(f"[{_timestamp()}] runs directory: {storage.root.resolve()}", flush=True)
    print(f"[{_timestamp()}] benchmarks directory: {storage.benchmarks_root.resolve()}", flush=True)
    print(
        f"[{_timestamp()}] models={models} trials={trials} concurrency={settings.benchmark_concurrency} rule={rule} topology={topology} max_steps={max_steps}",
        flush=True,
    )
    runner = BenchmarkRunner(settings, storage, progress_callback=_print_progress)
    result = await runner.run(
        models=models,
        trials_per_model=trials,
        rule=rule,
        topology=topology,
        max_steps=max_steps,
        max_live_fraction=max_live_fraction,
    )
    summary = {
        "benchmark_id": result.benchmark_id,
        "trials_per_model": result.trials_per_model,
        "models": [
            {
                "model": model.model,
                "submission_score": model.submission_score,
                "average_score": model.average_score,
                "submission_seed": model.submission_seed,
                "submission_run_id": model.submission_run_id,
            }
            for model in result.models
        ],
    }
    print(f"[{_timestamp()}] benchmark run complete: {result.benchmark_id}", flush=True)
    print(json.dumps(summary, indent=2))


def _print_progress(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _print_leaderboard(as_json: bool, out_path: Path | None) -> None:
    storage = RunStorage(settings.runs_dir, settings.benchmarks_dir)
    benchmarks = storage.load_benchmarks()
    payload = build_leaderboard_payload(benchmarks)
    leaderboard = payload["leaderboard"]

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote leaderboard snapshot to {out_path.resolve()}")

    if as_json:
        print(json.dumps(payload, indent=2))
        return

    if not leaderboard:
        print("No benchmark results found.")
        return

    print(f"Benchmarks loaded: {len(benchmarks)}")
    print("")
    header = f"{'rk':<4}{'model':<40}{'submission':>12}{'avg':>10}{'trials':>8}{'batches':>10}"
    print(header)
    print("-" * len(header))
    for entry in leaderboard:
        model = entry["model"]
        if len(model) > 38:
            model = model[:37] + "…"
        print(
            f"{entry['rank']:<4}{model:<40}{entry['submission_score']:>12}"
            f"{entry['best_average_score']:>10.2f}{entry['trial_count']:>8}{entry['benchmarks_run']:>10}"
        )
