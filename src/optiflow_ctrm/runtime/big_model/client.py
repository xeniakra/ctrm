"""vLLM client for the Qwen3-30B-A3B-Instruct-2507 FP8 checkpoint.

The checkpoint is distributed as FP8 safetensors rather than GGUF, so it is
served by vLLM and accessed through vLLM's OpenAI-compatible HTTP API.  This
module owns transport and structured decoding only; extraction and validation
remain model-independent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from optiflow_ctrm.runtime.base import (
    ChatMessage,
    InferenceConnectionError,
    InferenceProtocolError,
    InferenceResult,
)


# Qwen3-30B-A3B-Instruct-2507 is the best big-model candidate for this
# comparison: its 30B MoE capacity tests a much larger parameter pool while
# activating only about 3B parameters per token.  The official FP8 checkpoint
# cuts weight memory relative to BF16, and matching the small candidate's Qwen3
# release and instruction-tuned family isolates the capacity trade-off.
MODEL_REPOSITORY = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_REVISION = "e87d00b8a62c184ac41942416fc9e861a84ad71b"
MODEL_QUANTIZATION = "FP8"

Transport = Callable[[str, bytes, dict[str, str], float], bytes]


@dataclass(frozen=True, slots=True)
class Qwen30BVllmConfig:
    """Pinned, deterministic generation settings for a local vLLM server."""

    base_url: str = "http://127.0.0.1:8000"
    model: str = MODEL_REPOSITORY
    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 240.0
    api_key: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("base_url must point to a local vLLM server")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class Qwen30BVllmClient:
    """Synchronous schema-constrained client implementing ``InferenceRuntime``."""

    def __init__(
        self,
        config: Qwen30BVllmConfig | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config or Qwen30BVllmConfig(api_key=os.getenv("VLLM_API_KEY"))
        self._transport = transport or _post_json

    def generate_json(
        self,
        messages: Sequence[ChatMessage],
        json_schema: Mapping[str, Any],
    ) -> InferenceResult:
        clean_messages = _validate_messages(messages)
        if not isinstance(json_schema, Mapping) or not json_schema:
            raise ValueError("json_schema must be a non-empty mapping")

        payload = {
            "model": self.config.model,
            "messages": clean_messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "optiflow_ctrm_extraction",
                    "schema": dict(json_schema),
                },
            },
            "seed": self.config.seed,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        endpoint = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        raw = self._transport(
            endpoint,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers,
            self.config.timeout_seconds,
        )
        response = _decode_response(raw)
        text, finish_reason = _extract_content(response)
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InferenceProtocolError("vLLM model content was not valid JSON") from exc

        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        return InferenceResult(
            value=value,
            raw_text=text,
            model=response.get("model") if isinstance(response.get("model"), str) else None,
            finish_reason=finish_reason,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
            tokens_per_second=None,
            timings={},
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
        raise InferenceProtocolError("vLLM returned an invalid JSON response") from exc
    if not isinstance(response, Mapping):
        raise InferenceProtocolError("vLLM response must be a JSON object")
    return response


def _extract_content(response: Mapping[str, Any]) -> tuple[str, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise InferenceProtocolError("vLLM response has no completion choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise InferenceProtocolError("vLLM response has no string message content")
    finish = choice.get("finish_reason")
    return message["content"], finish if isinstance(finish, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _post_json(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local-only URL
            return response.read()
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise InferenceProtocolError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise InferenceConnectionError(f"could not reach local vLLM server: {exc}") from exc
