import json

import pytest

from optiflow_ctrm.runtime import InferenceRuntime, create_default_runtime
from optiflow_ctrm.runtime.small_model import LlamaCppClient, LlamaCppConfig
from optiflow_ctrm.runtime.base import InferenceProtocolError


def test_llama_cpp_client_implements_inference_runtime() -> None:
    assert isinstance(LlamaCppClient(), InferenceRuntime)


def test_qwen_llama_cpp_client_remains_the_default_runtime() -> None:
    assert isinstance(create_default_runtime(), LlamaCppClient)


def test_generate_json_sends_deterministic_schema_constrained_request() -> None:
    captured = {}

    def transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        captured.update(url=url, payload=json.loads(body), headers=headers, timeout=timeout)
        return json.dumps(
            {
                "model": "qwen-local",
                "choices": [{"message": {"content": '{"buyer":"Acme"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
                "timings": {"predicted_per_second": 22.5},
            }
        ).encode()

    client = LlamaCppClient(LlamaCppConfig(base_url="http://127.0.0.1:8080/"), transport)
    result = client.generate_json(
        [{"role": "user", "content": "extract"}],
        {"type": "object", "properties": {"buyer": {"type": "string"}}},
    )

    assert result.value == {"buyer": "Acme"}
    assert result.tokens_per_second == 22.5
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert captured["payload"]["seed"] == 42
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["response_format"] == {
        "type": "json_object",
        "schema": {"type": "object", "properties": {"buyer": {"type": "string"}}},
    }


def test_generate_json_rejects_non_json_model_content() -> None:
    def transport(*_args: object) -> bytes:
        return b'{"choices":[{"message":{"content":"not json"}}]}'

    client = LlamaCppClient(transport=transport)
    with pytest.raises(InferenceProtocolError, match="valid JSON"):
        client.generate_json(
            [{"role": "user", "content": "extract"}],
            {"type": "object"},
        )


def test_config_rejects_non_local_endpoint_by_default() -> None:
    with pytest.raises(ValueError, match="local llama.cpp"):
        LlamaCppConfig(base_url="https://example.com")
