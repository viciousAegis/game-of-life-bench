import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class RunStorage:
    def __init__(self, root: Path, benchmarks_root: Path | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.benchmarks_root = benchmarks_root or (self.root.parent / "benchmarks")
        self.benchmarks_root.mkdir(parents=True, exist_ok=True)

    def save_run(self, payload: dict) -> str:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        target = self.root / f"{run_id}.json"
        with target.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2)
        return run_id

    def save_benchmark(self, payload: dict) -> str:
        benchmark_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        target = self.benchmarks_root / f"{benchmark_id}.json"
        with target.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2)
        return benchmark_id

    def load_benchmarks(self) -> list[dict]:
        benchmarks: list[dict] = []
        for path in sorted(self.benchmarks_root.glob("*.json"), reverse=True):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["benchmark_id"] = path.stem
            benchmarks.append(payload)
        return benchmarks


def _to_jsonable(value):
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value
