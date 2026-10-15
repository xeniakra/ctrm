"""Model-independent entry points for self-hosted inference runtimes."""

from optiflow_ctrm.runtime.base import (
    ChatMessage,
    InferenceConnectionError,
    InferenceError,
    InferenceProtocolError,
    InferenceResult,
    InferenceRuntime,
)

__all__ = [
    "ChatMessage",
    "InferenceConnectionError",
    "InferenceError",
    "InferenceProtocolError",
    "InferenceResult",
    "InferenceRuntime",
    "create_default_runtime",
]


def create_default_runtime() -> InferenceRuntime:
    """Create the configured production runtime, currently Qwen via llama.cpp."""

    from optiflow_ctrm.runtime.small_model import LlamaCppClient

    return LlamaCppClient()
