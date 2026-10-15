import json

import pytest

from optiflow_ctrm.runtime import InferenceRuntime
from optiflow_ctrm.runtime.big_model import (
    MODEL_REPOSITORY,
    Qwen30BVllmClient,
    Qwen30BVllmConfig,
)
from optiflow_ctrm.runtime.base import InferenceProtocolError


def test_vllm_client_implements_inference_runtime() -> None:
    assert isinstance(Qwen30BVllmClient(), InferenceRuntime)


def test_vllm_client_sends_native_json_schema_response_format() -> None:
    captured = {}

    def transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        captured.update(url=url, payload=json.loads(body), headers=headers, timeout=timeout)
        return json.dumps(
            {
                "model": MODEL_REPOSITORY,
                "choices": [
                    {"message": {"content": '{"buyer":"Acme"}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            }
        ).encode()

    config = Qwen30BVllmConfig(
        base_url="http://127.0.0.1:8000/",
        api_key="local-test-key",
    )
    client = Qwen30BVllmClient(config, transport)
    schema = {"type": "object", "properties": {"buyer": {"type": "string"}}}

    result = client.generate_json([{"role": "user", "content": "extract"}], schema)

    assert result.value == {"buyer": "Acme"}
    assert result.prompt_tokens == 11
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer local-test-key"
    assert captured["payload"]["model"] == MODEL_REPOSITORY
    assert captured["payload"]["seed"] == 42
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "optiflow_ctrm_extraction", "schema": schema},
    }


def test_vllm_client_rejects_non_json_content() -> None:
    def transport(*_args: object) -> bytes:
        return b'{"choices":[{"message":{"content":"not json"}}]}'

    with pytest.raises(InferenceProtocolError, match="valid JSON"):
        Qwen30BVllmClient(transport=transport).generate_json(
            [{"role": "user", "content": "extract"}],
            {"type": "object"},
        )


def test_vllm_config_rejects_remote_endpoint() -> None:
    with pytest.raises(ValueError, match="local vLLM"):
        Qwen30BVllmConfig(base_url="https://example.com")
