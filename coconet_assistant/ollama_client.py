"""Model-agnostic Ollama HTTP client (any local Llama / Qwen / similar tag)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5"
# Ollama's own default context (2-4k tokens) silently truncates the system prompt
# once tool results arrive, so the window is always set explicitly.
DEFAULT_NUM_CTX = 16384


class OllamaNotAvailable(RuntimeError):
    """Ollama is not running, the host is unreachable, or the model is missing."""


def default_ollama_host() -> str:
    return os.environ.get("COCONET_OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def default_ollama_model() -> str:
    return os.environ.get("COCONET_ASSISTANT_MODEL", DEFAULT_MODEL)


def default_num_ctx() -> int:
    value = os.environ.get("COCONET_OLLAMA_NUM_CTX", "").strip()
    return int(value) if value.isdigit() and int(value) > 0 else DEFAULT_NUM_CTX


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "The assistant extra is required for the Ollama client. "
            "Install with: pip install 'coconet-python[assistant]'"
        ) from exc
    return httpx


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    raw: dict[str, Any]


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        name = function.get("name") or raw.get("name")
        arguments = function.get("arguments", raw.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        if name:
            calls.append({"id": raw.get("id") or name, "name": name, "arguments": arguments})
    return calls


class OllamaClient:
    """Thin wrapper around Ollama ``/api/chat`` with tool calling."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,
        num_ctx: int | None = None,
    ) -> None:
        self.host = (host or default_ollama_host()).rstrip("/")
        self.model = model or default_ollama_model()
        self.timeout = timeout
        self.num_ctx = num_ctx or default_num_ctx()

    def check(self) -> dict[str, Any]:
        """Return tags if Ollama is up; raise ``OllamaNotAvailable`` otherwise."""
        httpx = _import_httpx()
        parsed = urlparse(self.host)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise OllamaNotAvailable(
                f"Invalid Ollama host {self.host!r}. Expected http://127.0.0.1:11434"
            )
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=min(self.timeout, 5.0))
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaNotAvailable(
                f"Ollama is not running at {self.host}. Start it with `ollama serve`, "
                f"then pull a model (for example `ollama pull {self.model}`)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaNotAvailable(
                f"Timed out reaching Ollama at {self.host}. Is `ollama serve` running?"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaNotAvailable(f"Ollama at {self.host} returned an error: {exc}") from exc

        payload = response.json()
        names = [item.get("name", "") for item in payload.get("models") or []]
        if names and not any(
            self.model == name or name.startswith(f"{self.model}:") for name in names
        ):
            listed = ", ".join(names) or "(none)"
            raise OllamaNotAvailable(
                f"Ollama is running but model {self.model!r} is not pulled. "
                f"Installed models: {listed}. Run: ollama pull {self.model}"
            )
        return {"host": self.host, "model": self.model, "models": names}

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        httpx = _import_httpx()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Deterministic translation: the same question gives the same scenario.
            "options": {"num_ctx": self.num_ctx, "temperature": 0, "seed": 0},
        }
        if tools:
            body["tools"] = tools
        try:
            response = httpx.post(
                f"{self.host}/api/chat",
                json=body,
                timeout=self.timeout,
            )
        except httpx.ConnectError as exc:
            raise OllamaNotAvailable(
                f"Ollama is not running at {self.host}. Start it with `ollama serve` "
                f"and pull a model (`ollama pull {self.model}`)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaNotAvailable(
                f"Timed out calling Ollama model {self.model} at {self.host}."
            ) from exc

        if response.status_code == 404:
            raise OllamaNotAvailable(
                f"Ollama model {self.model!r} was not found. Run: ollama pull {self.model}"
            )
        if response.status_code >= 400:
            raise OllamaNotAvailable(
                f"Ollama chat failed ({response.status_code}): {response.text[:400]}"
            )

        payload = response.json()
        message = payload.get("message") or {}
        content = message.get("content") or ""
        tool_calls = _parse_tool_calls(message)
        if tools and not tool_calls and not content:
            raise OllamaNotAvailable(
                f"Model {self.model} returned an empty reply with no tool calls. "
                "Use a tool-capable local model such as qwen2.5 or llama3.1."
            )
        return LLMResponse(content=content, tool_calls=tool_calls, raw=payload)
