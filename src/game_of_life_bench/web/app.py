from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..config import settings
from ..leaderboard import build_leaderboard
from ..models.openrouter import OpenRouterClient, OpenRouterGenerationError
from ..scoring import EvaluationResult, evaluate_board
from ..storage import RunStorage


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Game of Life Bench")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

storage = RunStorage(settings.runs_dir, settings.benchmarks_dir)

class ManualRunRequest(BaseModel):
    board: list[list[int]]
    max_steps: int = Field(default=settings.max_steps, ge=1)
    rule: str = settings.rule
    topology: str = settings.topology
    max_live_fraction: float = Field(default=settings.max_live_fraction, gt=0.0, le=1.0)


class ModelRunRequest(BaseModel):
    model: str = settings.openrouter_model
    max_steps: int = Field(default=settings.max_steps, ge=1)
    rule: str = settings.rule
    topology: str = settings.topology
    max_live_fraction: float = Field(default=settings.max_live_fraction, gt=0.0, le=1.0)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "rows": settings.grid_rows,
            "cols": settings.grid_cols,
            "default_rule": settings.rule,
            "default_max_steps": settings.max_steps,
            "default_model": settings.openrouter_model,
            "model_options": settings.openrouter_model_options,
            "max_live_cells": int(settings.grid_rows * settings.grid_cols * settings.max_live_fraction),
            "max_live_fraction": settings.max_live_fraction,
        },
    )


@app.get("/benchmark", response_class=HTMLResponse)
async def benchmark_page(request: Request) -> HTMLResponse:
    benchmarks = storage.load_benchmarks()
    leaderboard = build_leaderboard(benchmarks)
    return templates.TemplateResponse(
        "benchmark.html",
        {
            "request": request,
            "rows": settings.grid_rows,
            "cols": settings.grid_cols,
            "leaderboard": leaderboard,
            "benchmarks": benchmarks,
            "rule": settings.rule,
            "topology": settings.topology,
            "max_steps": settings.max_steps,
            "trials_per_model": settings.benchmark_trials,
        },
    )

@app.get("/api/config")
async def get_config() -> dict:
    return {
        "rows": settings.grid_rows,
        "cols": settings.grid_cols,
        "rule": settings.rule,
        "topology": settings.topology,
        "max_steps": settings.max_steps,
        "max_live_fraction": settings.max_live_fraction,
        "max_live_cells": int(settings.grid_rows * settings.grid_cols * settings.max_live_fraction),
        "default_model": settings.openrouter_model,
        "model_options": list(settings.openrouter_model_options),
        "benchmark_trials": settings.benchmark_trials,
    }


@app.get("/api/leaderboard")
async def get_leaderboard() -> dict:
    benchmarks = storage.load_benchmarks()
    leaderboard = build_leaderboard(benchmarks)
    return {"leaderboard": leaderboard, "benchmarks": benchmarks}


@app.post("/api/evaluate/manual")
async def evaluate_manual(request: ManualRunRequest) -> dict:
    try:
        evaluation = evaluate_board(
            board=request.board,
            rows=settings.grid_rows,
            cols=settings.grid_cols,
            rule=request.rule,
            topology=request.topology,
            max_steps=request.max_steps,
            max_live_fraction=request.max_live_fraction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _save_payload("manual", evaluation=evaluation, request=request.model_dump())


@app.post("/api/evaluate/openrouter")
async def evaluate_openrouter(request: ModelRunRequest) -> dict:
    try:
        client = OpenRouterClient(settings)
        max_live_cells = int(settings.grid_rows * settings.grid_cols * request.max_live_fraction)
        model_run = await client.generate_board(
            model=request.model,
            rows=settings.grid_rows,
            cols=settings.grid_cols,
            max_live_cells=max_live_cells,
            rule=request.rule,
            topology=request.topology,
        )
        evaluation = evaluate_board(
            board=model_run.board,
            rows=settings.grid_rows,
            cols=settings.grid_cols,
            rule=request.rule,
            topology=request.topology,
            max_steps=request.max_steps,
            max_live_fraction=request.max_live_fraction,
        )
    except OpenRouterGenerationError as exc:
        raise HTTPException(status_code=502, detail=exc.to_detail()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _save_payload(
        "openrouter",
        evaluation=evaluation,
        request=request.model_dump(),
        model_run={
            "model": model_run.model,
            "prompt": model_run.prompt,
            "raw_response": model_run.raw_response,
            "board": model_run.board,
            "response_metadata": model_run.response_metadata,
        },
    )
def _save_payload(source: str, evaluation: EvaluationResult, request: dict, model_run: dict | None = None) -> dict:
    payload = {
        "source": source,
        "request": request,
        "model_run": model_run,
        "evaluation": evaluation,
    }
    run_id = storage.save_run(payload)
    return {"run_id": run_id, **payload}
