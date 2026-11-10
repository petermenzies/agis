"""Model backends. Each one turns chat messages plus a JSON schema into a JSON string."""

from __future__ import annotations

from typing import Protocol

import httpx
import ollama

from agis.errors import LLMError

Message = dict[str, str]
NUM_CTX = 8192


class Backend(Protocol):
    model: str

    def generate_json(self, messages: list[Message], schema: dict) -> str: ...

    def ready(self) -> bool:
        """True when the model is in memory with enough context to plan."""
        ...


class OllamaBackend:
    def __init__(self, model: str, host: str | None = None, timeout: float = 300):
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host, timeout=timeout)

    def generate_json(self, messages: list[Message], schema: dict) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=schema,
                options={"temperature": 0, "num_ctx": NUM_CTX},
                keep_alive="10m",
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMError(
                    f"model {self.model!r} isn't installed in Ollama. "
                    f"Run `agis setup` or `ollama pull {self.model}`."
                ) from exc
            raise LLMError(f"Ollama error: {exc.error}") from exc
        except (httpx.ConnectError, ConnectionError) as exc:
            raise LLMError(
                "can't reach Ollama. Start it with `ollama serve` or open the Ollama app, "
                "then try again (`agis doctor` can help)."
            ) from exc
        return response.message.content or ""

    def ready(self) -> bool:
        try:
            response = httpx.get(f"{_host(self.host)}/api/ps", timeout=2)
            response.raise_for_status()
            models = response.json().get("models") or []
        except (httpx.HTTPError, ValueError):
            return False
        return model_ready(models, self.model, NUM_CTX)

    def installed_models(self) -> list[str]:
        try:
            return [m.model for m in self.client.list().models]
        except (httpx.ConnectError, ConnectionError) as exc:
            raise LLMError("can't reach Ollama") from exc


def make_backend(model: str, host: str | None = None) -> Backend:
    return OllamaBackend(model, host)


def model_ready(models: list[dict], name: str, num_ctx: int) -> bool:
    """True when one running model is `name` and its context can hold a plan."""
    for item in models:
        found = item.get("model") or item.get("name") or ""
        if not _same_model(name, found):
            continue
        context = item.get("context_length")
        return context is None or context >= num_ctx
    return False


def _same_model(wanted: str, found: str) -> bool:
    wanted, found = wanted.casefold(), found.casefold()
    return wanted == found or (":" not in wanted and found == f"{wanted}:latest")


def _host(host: str | None) -> str:
    return (host or "http://127.0.0.1:11434").rstrip("/")
