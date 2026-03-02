from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "127.0.0.1"
    app_port: int = 8000

    grid_rows: int = 8
    grid_cols: int = 8
    topology: str = "toroidal"
    rule: str = "B3/S23"
    max_steps: int = 1000
    max_live_fraction: float = 0.5
    benchmark_trials: int = 10
    benchmark_concurrency: int = 5

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_model_options: tuple[str, ...] = (
        "google/gemma-3-27b-it",
        "qwen/qwen3-30b-a3b",
        "deepseek/deepseek-r1-distill-qwen-32b",
    )
    openrouter_timeout_seconds: float = 60.0
    openrouter_site_url: str | None = None
    openrouter_site_name: str = "game-of-life-bench"

    runs_dir: Path = Field(default_factory=lambda: Path("runs"))
    benchmarks_dir: Path = Field(default_factory=lambda: Path("benchmarks"))


settings = Settings()
