"""Deterministic adapter for a local llama.cpp HTTP server.

The model is loaded by ``llama-server``; this module deliberately owns only the
inference transport.  Keeping that boundary means callers cannot accidentally
drop the seed or the JSON-Schema grammar on one extraction path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from optiflow_ctrm.runtime.base import (
    ChatMessage,
    InferenceConnectionError,
    InferenceError,
    InferenceProtocolError,
    InferenceResult,
)


# Qwen3-4B-Instruct-2507 is the best small-model candidate for this comparison:
# four dense billion parameters at Q4_K_M fit a local llama.cpp deployment while
# retaining an instruction-tuned checkpoint suited to schema-constrained
# extraction.  It is from the same Qwen3 release family as the big candidate, so
# the evaluation exposes the capacity trade-off without a model-family change.
# llama.cpp needs the converted GGUF artifact pinned below; a benchmark must
# additionally record its resolved commit and checksum.
SOURCE_MODEL_REPOSITORY = "Qwen/Qwen3-4B-Instruct-2507"
GGUF_MODEL_REPOSITORY = "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF"
GGUF_MODEL_FILE = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
GGUF_MODEL_REVISION = "ae44f08e1392f39c0e474af10c3ff8355c8b6688"
GGUF_MODEL_SHA256 = "2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e"

Transport = Callable[[str, bytes, dict[str, str], float], bytes]

LlamaCppError = InferenceError


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    """Pinned request settings for reproducible extraction.

    ``model`` is a llama.cpp server model alias, not a path and not permission
    to download weights.  The server should be started separately with the GGUF
    file named above.
    """

    base_url: str = "http://127.0.0.1:8080"
    model: str = SOURCE_MODEL_REPOSITORY
    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 240.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("base_url must point to a local llama.cpp server")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class LlamaCppClient:
    """Small synchronous client for ``/v1/chat/completions``."""

    def __init__(
        self,
        config: LlamaCppConfig | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config or LlamaCppConfig()
        self._transport = transport or _post_json

    def generate_json(
        self,
        messages: Sequence[ChatMessage],
        json_schema: Mapping[str, Any],
    ) -> InferenceResult:
        """Generate one JSON value constrained by ``json_schema``.

        This checks transport/protocol validity, but leaves validation against
        the canonical Pydantic model to the extractor layer.
        """

        clean_messages = _validate_messages(messages)
        if not isinstance(json_schema, Mapping) or not json_schema:
            raise ValueError("json_schema must be a non-empty mapping")

        payload = {
            "model": self.config.model,
            "messages": clean_messages,
            # llama.cpp's native OpenAI-compatible shape uses json_object plus
            # a sibling schema. Using type=json_schema with a sibling schema is
            # silently ignored by some releases, resulting in unconstrained JSON.
            "response_format": {"type": "json_object", "schema": dict(json_schema)},
            "seed": self.config.seed,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        endpoint = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        raw = self._transport(
            endpoint,
            encoded,
            {"Content-Type": "application/json", "Accept": "application/json"},
            self.config.timeout_seconds,
        )
        response = _decode_response(raw)
        text, finish_reason = _extract_content(response)
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InferenceProtocolError("model content was not valid JSON") from exc

        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
        rate = timings.get("predicted_per_second")
        return InferenceResult(
            value=value,
            raw_text=text,
            model=response.get("model") if isinstance(response.get("model"), str) else None,
            finish_reason=finish_reason,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
            tokens_per_second=float(rate) if isinstance(rate, (int, float)) else None,
            timings=dict(timings),
        )


def _validate_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("messages must not be empty")
    result: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("each message needs a valid role and string content")
        result.append({"role": role, "content": content})
    return result


def _decode_response(raw: bytes) -> Mapping[str, Any]:
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceProtocolError("llama.cpp returned an invalid JSON response") from exc
    if not isinstance(response, Mapping):
        raise InferenceProtocolError("llama.cpp response must be a JSON object")
    return response


def _extract_content(response: Mapping[str, Any]) -> tuple[str, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise InferenceProtocolError("llama.cpp response has no completion choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise InferenceProtocolError("llama.cpp response has no string message content")
    finish = choice.get("finish_reason")
    return message["content"], finish if isinstance(finish, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _post_json(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is local-only
            return response.read()
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise InferenceProtocolError(f"llama.cpp returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise InferenceConnectionError(f"could not reach local llama.cpp server: {exc}") from exc
