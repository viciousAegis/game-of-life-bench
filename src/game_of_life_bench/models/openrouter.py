import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import Settings


@dataclass
class ModelRunResult:
    model: str
    prompt: str
    raw_response: str
    board: list[list[int]]
    response_metadata: dict[str, Any]


class OpenRouterGenerationError(Exception):
    def __init__(
        self,
        message: str,
        *,
        model: str,
        prompt: str,
        raw_response: str | None = None,
        status_code: int | None = None,
        provider_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.prompt = prompt
        self.raw_response = raw_response
        self.status_code = status_code
        self.provider_error = provider_error

    def to_detail(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "model": self.model,
            "status_code": self.status_code,
            "provider_error": self.provider_error,
            "raw_response": self.raw_response,
            "prompt": self.prompt,
        }


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if not settings.openrouter_api_key and not _is_local_server_url(settings.openrouter_base_url):
            raise ValueError("OPENROUTER_API_KEY is not configured.")

    async def generate_board(
        self,
        model: str,
        rows: int,
        cols: int,
        max_live_cells: int,
        rule: str,
        topology: str,
        seed: int | None = None,
    ) -> ModelRunResult:
        prompt = _build_prompt(
            rows=rows,
            cols=cols,
            max_live_cells=max_live_cells,
            rule=rule,
            topology=topology,
            seed=seed,
        )
        headers = {
            "Content-Type": "application/json",
        }
        if self._settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {self._settings.openrouter_api_key}"
        if self._settings.openrouter_site_url:
            headers["HTTP-Referer"] = self._settings.openrouter_site_url
        if self._settings.openrouter_site_name:
            headers["X-Title"] = self._settings.openrouter_site_name

        async with httpx.AsyncClient(
            base_url=self._settings.openrouter_base_url,
            timeout=self._settings.openrouter_timeout_seconds,
        ) as client:
            try:
                data = await self._request_board(
                    client=client,
                    headers=headers,
                    model=model,
                    prompt=prompt,
                    rows=rows,
                    cols=cols,
                )
            except httpx.HTTPError as exc:
                raise OpenRouterGenerationError(
                    "OpenRouter request failed.",
                    model=model,
                    prompt=prompt,
                    status_code=getattr(exc.response, "status_code", None),
                    provider_error=_extract_http_error_text(exc),
                ) from exc

        choice = data["choices"][0]
        message = choice["message"]["content"]
        raw_response = _message_to_text(message)
        response_metadata = _extract_response_metadata(data)
        try:
            board = _extract_board(raw_response)
        except ValueError as exc:
            choice_debug = _extract_choice_debug(data)
            raise OpenRouterGenerationError(
                f"Model response could not be parsed into a board: {exc}",
                model=model,
                prompt=prompt,
                raw_response=raw_response or choice_debug,
            ) from exc
        return ModelRunResult(
            model=model,
            prompt=prompt,
            raw_response=raw_response,
            board=board,
            response_metadata=response_metadata,
        )

    async def _request_board(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        model: str,
        prompt: str,
        rows: int,
        cols: int,
    ) -> dict[str, Any]:
        payload_with_schema = _build_payload(
            model=model,
            prompt=prompt,
            rows=rows,
            cols=cols,
            use_schema=True,
        )
        response = await client.post("/chat/completions", headers=headers, json=payload_with_schema)
        if response.status_code < 400:
            return response.json()

        payload_without_schema = _build_payload(
            model=model,
            prompt=prompt,
            rows=rows,
            cols=cols,
            use_schema=False,
        )
        fallback_response = await client.post("/chat/completions", headers=headers, json=payload_without_schema)
        fallback_response.raise_for_status()
        return fallback_response.json()


def _build_prompt(
    rows: int,
    cols: int,
    max_live_cells: int,
    rule: str,
    topology: str,
    seed: int | None = None,
) -> str:
    example_board = _build_example_board(rows=rows, cols=cols)
    seed_line = f"Trial seed: {seed}\n" if seed is not None else ""
    return (
        f"Create an initial {rows}x{cols} binary grid for a cellular automaton.\n"
        "Rule system: Conway's Game of Life.\n"
        "A live cell survives only if it has 2 or 3 live neighbors.\n"
        "A dead cell becomes alive only if it has exactly 3 live neighbors.\n"
        f"Rule notation reference: {rule}\n"
        f"Topology: {topology}\n"
        f"{seed_line}"
        f"Goal: maximize the number of steps before the first repeated global state.\n"
        f"Constraint: at most {max_live_cells} live cells total.\n"
        "You may reason internally as much as needed, but do not include any reasoning in the output.\n"
        "Return JSON with one field named 'board'. "
        "The board must be an array of rows containing only 0 or 1.\n"
        "Example response format:\n"
        f"{json.dumps({'board': example_board})}\n"
        "Do not include any explanation."
    )


def _build_payload(model: str, prompt: str, rows: int, cols: int, use_schema: bool) -> dict[str, Any]:
    system_message = (
        "You design high-scoring cellular automata initial states. "
        "Return only a compact JSON object with a single field named 'board'."
    )
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    }
    if use_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "game_of_life_bench_board",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "board": {
                            "type": "array",
                            "minItems": rows,
                            "maxItems": rows,
                            "items": {
                                "type": "array",
                                "minItems": cols,
                                "maxItems": cols,
                                "items": {"type": "integer", "enum": [0, 1]},
                            },
                        }
                    },
                    "required": ["board"],
                    "additionalProperties": False,
                },
            },
        }
    return payload


def _is_local_server_url(base_url: str) -> bool:
    hostname = urlparse(base_url).hostname
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return json.dumps(message)


def _extract_board(text: str) -> list[list[int]]:
    parsed = json.loads(_extract_json_blob(text))
    board = parsed["board"]
    if not isinstance(board, list):
        raise ValueError("Model response is missing a valid 'board' array.")
    return board


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    raise ValueError("Model response did not contain a JSON object.")


def _extract_http_error_text(exc: httpx.HTTPError) -> str | None:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    try:
        data = response.json()
    except ValueError:
        body = response.text.strip()
        return body or str(exc)

    if isinstance(data, dict):
        if "error" in data:
            error_value = data["error"]
            if isinstance(error_value, dict):
                message = error_value.get("message")
                code = error_value.get("code")
                if message and code:
                    return f"{code}: {message}"
                if message:
                    return str(message)
            return str(error_value)
        if "message" in data:
            return str(data["message"])
    return json.dumps(data)


def _build_example_board(rows: int, cols: int) -> list[list[int]]:
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    if rows >= 3 and cols >= 3:
        mid_row = rows // 2
        mid_col = cols // 2
        board[mid_row][mid_col - 1 : mid_col + 2] = [1, 1, 1]
    return board


def _extract_choice_debug(data: dict[str, Any]) -> str:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return json.dumps(data)

    debug_payload = {
        "finish_reason": choice.get("finish_reason"),
        "message": choice.get("message"),
    }
    return json.dumps(debug_payload, indent=2, default=str)


def _extract_response_metadata(data: dict[str, Any]) -> dict[str, Any]:
    choice = {}
    if isinstance(data.get("choices"), list) and data["choices"]:
        first_choice = data["choices"][0]
        if isinstance(first_choice, dict):
            choice = first_choice

    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}

    metadata = {
        "id": data.get("id"),
        "model": data.get("model"),
        "provider": data.get("provider"),
        "created": data.get("created"),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "reasoning": {
            "reasoning": message.get("reasoning"),
            "reasoning_details": message.get("reasoning_details"),
            "reasoning_text": _extract_reasoning_text(message),
        },
        "message": {
            "role": message.get("role"),
            "refusal": message.get("refusal"),
            "annotations": message.get("annotations"),
        },
    }
    return metadata


def _extract_reasoning_text(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    details = message.get("reasoning_details")
    if not isinstance(details, list):
        return None

    parts: list[str] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts) if parts else None
