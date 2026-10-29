"""Model backends. Each one turns chat messages plus a JSON schema into a JSON string."""

from __future__ import annotations

from typing import Protocol

import httpx
import ollama

from agis.errors import LLMError

Message = dict[str, str]


class Backend(Protocol):
    model: str

    def generate_json(self, messages: list[Message], schema: dict) -> str: ...


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
                options={"temperature": 0, "num_ctx": 8192},
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

    def installed_models(self) -> list[str]:
        try:
            return [m.model for m in self.client.list().models]
        except (httpx.ConnectError, ConnectionError) as exc:
            raise LLMError("can't reach Ollama") from exc


def make_backend(model: str, host: str | None = None) -> Backend:
    return OllamaBackend(model, host)
