"""Model-independent contract for schema-constrained local inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


ChatMessage = Mapping[str, str]


class InferenceError(RuntimeError):
    """Base class for safe, user-presentable inference failures."""


class InferenceConnectionError(InferenceError):
    """The configured local inference runtime could not be reached."""


class InferenceProtocolError(InferenceError):
    """The inference runtime returned a malformed or unsuccessful response."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Parsed structured output plus implementation-neutral measurements."""

    value: Any
    raw_text: str
    model: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    tokens_per_second: float | None
    timings: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class InferenceRuntime(Protocol):
    """Generate a JSON value constrained by the supplied JSON Schema."""

    def generate_json(
        self,
        messages: Sequence[ChatMessage],
        json_schema: Mapping[str, Any],
    ) -> InferenceResult: ...
