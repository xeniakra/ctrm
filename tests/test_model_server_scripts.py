import hashlib
import json
from pathlib import Path
import signal

import pytest

from scripts import run_local_eval, start_llama_server
from scripts.model_cache import ModelArtifact, ensure_cached
from scripts.start_llama_server import build_server_command, stop_existing_llama_server
from optiflow_ctrm.runtime.small_model import GGUF_MODEL_REVISION, GGUF_MODEL_SHA256
from scripts.model_cache import QWEN_GGUF


def test_model_server_runners_default_to_8k_context() -> None:
    assert run_local_eval._parse_args([]).ctx_size == 8192
    assert start_llama_server._parse_args([]).ctx_size == 8192


def test_cached_model_is_reused_without_downloading(tmp_path: Path) -> None:
    content = b"small fake gguf"
    artifact = ModelArtifact(
        repository="example/model",
        revision="abc123",
        filename="model.gguf",
        sha256=hashlib.sha256(content).hexdigest(),
    )
    destination = artifact.cache_path(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(content)

    called = False

    def download(_url: str, _partial: Path) -> None:
        nonlocal called
        called = True

    assert ensure_cached(artifact, tmp_path, download=download) == destination
    assert called is False


def test_server_command_uses_cached_weight_and_fixed_local_settings(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    command = build_server_command(
        binary="/usr/local/bin/llama-server",
        model_path=model,
        host="127.0.0.1",
        port=8080,
        context_size=32768,
        gpu_layers=-1,
        threads=8,
    )

    assert command == [
        "/usr/local/bin/llama-server",
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--ctx-size",
        "32768",
        "--n-gpu-layers",
        "-1",
        "--threads",
        "8",
        "--seed",
        "42",
        "--metrics",
    ]


def test_vllm_command_uses_pinned_fp8_model_and_persistent_cache(tmp_path: Path) -> None:
    command = run_local_eval.build_vllm_server_command(
        binary="/usr/local/bin/vllm",
        cache_dir=tmp_path,
        host="127.0.0.1",
        port=8000,
        context_size=32768,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
    )

    assert command[:3] == [
        "/usr/local/bin/vllm",
        "serve",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    ]
    assert command[command.index("--download-dir") + 1] == str(tmp_path)
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--revision" in command


def test_runtime_and_download_scripts_pin_the_same_artifact() -> None:
    assert QWEN_GGUF.revision == GGUF_MODEL_REVISION
    assert QWEN_GGUF.sha256 == GGUF_MODEL_SHA256


def test_process_tree_rss_sums_server_and_descendants(monkeypatch) -> None:
    class Result:
        stdout = "100 1 1000\n101 100 500\n102 101 250\n999 1 9000\n"

    monkeypatch.setattr(run_local_eval.subprocess, "run", lambda *_args, **_kwargs: Result())

    process_ids, rss_bytes = run_local_eval._process_tree_rss(100)

    assert process_ids == {100, 101, 102}
    assert rss_bytes == 1750 * 1024


def test_resource_metrics_are_added_to_report(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    report_path.write_text(json.dumps({"summary": {"cases": 1}}))
    metrics = run_local_eval.ResourceMetrics(
        sample_interval_seconds=0.25,
        samples=4,
        peak_ram_bytes=512 * 1024 * 1024,
        peak_vram_bytes=256 * 1024 * 1024,
        vram_source="nvidia-smi process accounting",
    )

    assert run_local_eval.add_resource_metrics(report_path, metrics) is True
    report = json.loads(report_path.read_text())
    assert report["summary"]["peak_ram_mib"] == 512.0
    assert report["summary"]["peak_vram_mib"] == 256.0
    assert report["resources"]["samples"] == 4


def test_combined_runner_stops_server_after_evaluation(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    calls = {}

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = FakeProcess()
    monkeypatch.setattr(run_local_eval, "ensure_cached", lambda **_kwargs: model)
    monkeypatch.setattr(run_local_eval, "find_server_binary", lambda _value: "/bin/llama-server")
    monkeypatch.setattr(run_local_eval, "server_version", lambda _value: "llama.cpp test")
    monkeypatch.setattr(run_local_eval, "stop_existing_llama_server", lambda _port: [])
    monkeypatch.setattr(run_local_eval, "wait_until_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_local_eval.subprocess, "Popen", lambda *_args, **_kwargs: process)

    def fake_eval(arguments):
        calls["eval_arguments"] = arguments
        return 0

    monkeypatch.setattr(run_local_eval, "evaluation_main", fake_eval)
    output = tmp_path / "evaluation.json"
    log = tmp_path / "server.log"

    assert run_local_eval.main(["--output", str(output), "--server-log", str(log)]) == 0
    assert process.terminated is True
    assert calls["eval_arguments"][:2] == ["--base-url", "http://127.0.0.1:8080"]
    assert str(output) in calls["eval_arguments"]


def test_combined_runner_can_start_big_model_vllm(tmp_path: Path, monkeypatch) -> None:
    calls = {}

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(run_local_eval, "find_vllm_binary", lambda _value: "/bin/vllm")
    monkeypatch.setattr(run_local_eval, "server_version", lambda _value: "vLLM test")
    monkeypatch.setattr(run_local_eval, "stop_existing_model_server", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(run_local_eval, "wait_until_ready", lambda *_args, **_kwargs: None)

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["environment"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(run_local_eval.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        run_local_eval,
        "evaluation_main",
        lambda arguments: calls.update(eval_arguments=arguments) or 0,
    )

    output = tmp_path / "evaluation.json"
    log = tmp_path / "server.log"
    assert (
        run_local_eval.main(
            [
                "--runtime",
                "vllm",
                "--offline",
                "--cache-dir",
                str(tmp_path / "models"),
                "--output",
                str(output),
                "--server-log",
                str(log),
            ]
        )
        == 0
    )

    assert calls["command"][:3] == [
        "/bin/vllm",
        "serve",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    ]
    assert calls["environment"]["HF_HUB_OFFLINE"] == "1"
    assert calls["eval_arguments"][:4] == [
        "--base-url",
        "http://127.0.0.1:8000",
        "--runtime",
        "vllm",
    ]


def test_existing_llama_listener_is_terminated_before_restart(monkeypatch) -> None:
    listeners = iter([[1234], [], []])
    signals = []
    monkeypatch.setattr(
        "scripts.start_llama_server._listener_pids", lambda _port: next(listeners)
    )
    monkeypatch.setattr(
        "scripts.start_llama_server._process_command", lambda _pid: "/opt/homebrew/bin/llama-server"
    )
    monkeypatch.setattr("scripts.start_llama_server.os.kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr("scripts.start_llama_server.time.sleep", lambda _seconds: None)

    assert stop_existing_llama_server(8080) == [1234]
    assert signals == [(1234, signal.SIGTERM)]


def test_non_llama_listener_is_never_killed(monkeypatch) -> None:
    signals = []
    monkeypatch.setattr("scripts.start_llama_server._listener_pids", lambda _port: [4321])
    monkeypatch.setattr("scripts.start_llama_server._process_command", lambda _pid: "postgres")
    monkeypatch.setattr("scripts.start_llama_server.os.kill", lambda pid, sig: signals.append((pid, sig)))

    with pytest.raises(RuntimeError, match="not llama-server"):
        stop_existing_llama_server(8080)
    assert signals == []
