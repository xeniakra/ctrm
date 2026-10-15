"""Start a pinned local model server, run evaluation, and stop the server."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = str(REPOSITORY_ROOT / "src")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from optiflow_ctrm.runtime.big_model import (  # noqa: E402
    MODEL_REPOSITORY as VLLM_MODEL_REPOSITORY,
    MODEL_REVISION as VLLM_MODEL_REVISION,
)

try:
    from scripts.model_cache import DEFAULT_CACHE_DIR, QWEN_GGUF, ensure_cached
    from scripts.start_llama_server import (
        build_server_command,
        find_server_binary,
        server_version,
        stop_existing_llama_server,
        stop_existing_model_server,
    )
except ModuleNotFoundError:  # supports: python scripts/run_local_eval.py
    from model_cache import DEFAULT_CACHE_DIR, QWEN_GGUF, ensure_cached
    from start_llama_server import (
        build_server_command,
        find_server_binary,
        server_version,
        stop_existing_llama_server,
        stop_existing_model_server,
    )


def find_vllm_binary(requested: str | None) -> str:
    """Resolve the vLLM CLI without importing its large Python package."""

    candidate = requested or "vllm"
    path = Path(candidate).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    raise FileNotFoundError(
        "vllm was not found on PATH; install the optional alternative runtime in this "
        "environment with `python -m pip install -e '.[alternative]'` on a GPU host"
    )


def configure_vllm_cache(
    cache_dir: Path, environment: dict[str, str], *, offline: bool
) -> None:
    """Use one cache for metadata and weights, reusing older split metadata."""
    default_home = Path(environment.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"
    hf_home = Path(environment.get("HF_HOME", str(default_home))).expanduser()
    old_cache = Path(environment.get(
        "HF_HUB_CACHE", environment.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    )).expanduser()
    snapshot = (
        Path("models--" + VLLM_MODEL_REPOSITORY.replace("/", "--"))
        / "snapshots" / VLLM_MODEL_REVISION
    )
    if old_cache.resolve() != cache_dir.resolve():
        source = old_cache / snapshot
        destination = cache_dir / snapshot
        # Only metadata from the exact pinned revision; never copy large weights
        # or replace files already present in the chosen cache.
        for item in source.glob("*"):
            if item.suffix not in {".json", ".txt", ".jinja"} or not item.is_file():
                continue
            target = destination / item.name
            if not target.exists():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    environment["HF_HUB_CACHE"] = str(cache_dir)
    environment["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"


def build_vllm_server_command(
    *,
    binary: str,
    cache_dir: Path,
    host: str,
    port: int,
    context_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> list[str]:
    """Build the pinned vLLM command with a persistent Hugging Face cache."""

    return [
        binary,
        "serve",
        VLLM_MODEL_REPOSITORY,
        "--revision",
        VLLM_MODEL_REVISION,
        "--host",
        host,
        "--port",
        str(port),
        "--download-dir",
        str(cache_dir),
        "--max-model-len",
        str(context_size),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        "42",
        "--generation-config",
        "vllm",
        "--structured-outputs-config",
        json.dumps({"backend": "xgrammar", "disable_any_whitespace": True}),
    ]


def evaluation_main(arguments: Sequence[str]) -> int:
    """Import lazily so this script also works without an editable install."""

    source_dir = str(REPOSITORY_ROOT / "src")
    repository_dir = str(REPOSITORY_ROOT)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    if repository_dir not in sys.path:
        sys.path.insert(0, repository_dir)
    from evals.run import main

    return main(arguments)


def server_is_ready(base_url: str, timeout_seconds: float = 1.0) -> bool:
    try:
        with urlopen(f"{base_url}/health", timeout=timeout_seconds) as response:  # noqa: S310
            return response.status == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def wait_until_ready(
    process: subprocess.Popen,
    base_url: str,
    startup_timeout: float,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"model server exited with code {return_code} before becoming ready; "
                f"see {log_path}"
            )
        if server_is_ready(base_url):
            return
        time.sleep(0.25)
    raise TimeoutError(
        f"model server did not become ready within {startup_timeout:g}s; see {log_path}"
    )


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@dataclass
class ResourceMetrics:
    """Peak resources observed for the managed model-server process tree."""

    sample_interval_seconds: float
    samples: int = 0
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    vram_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": "model_server_process_tree",
            "measurement_window": "server_start_through_evaluation",
            "sample_interval_seconds": self.sample_interval_seconds,
            "samples": self.samples,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_ram_mib": _bytes_to_mib(self.peak_ram_bytes),
            "ram_metric": "sum_of_process_tree_rss",
            "peak_vram_bytes": self.peak_vram_bytes,
            "peak_vram_mib": _bytes_to_mib(self.peak_vram_bytes),
            "vram_source": self.vram_source,
        }


class ResourceMonitor:
    """Sample process-tree RSS and NVIDIA VRAM without extra dependencies."""

    def __init__(self, root_pid: int | None, interval_seconds: float = 0.25) -> None:
        self.root_pid = root_pid
        self.interval_seconds = interval_seconds
        self.metrics = ResourceMetrics(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvidia_smi = shutil.which("nvidia-smi")

    def start(self) -> None:
        if self.root_pid is None:
            return
        self._sample()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> ResourceMetrics:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
            self._sample()
        return self.metrics

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        if self.root_pid is None:
            return
        process_ids, rss_bytes = _process_tree_rss(self.root_pid)
        if rss_bytes is not None:
            self.metrics.samples += 1
            self.metrics.peak_ram_bytes = max(self.metrics.peak_ram_bytes or 0, rss_bytes)
        if self._nvidia_smi and process_ids:
            vram_bytes = _nvidia_process_vram(process_ids, self._nvidia_smi)
            if vram_bytes is not None:
                self.metrics.peak_vram_bytes = max(
                    self.metrics.peak_vram_bytes or 0, vram_bytes
                )
                self.metrics.vram_source = "nvidia-smi process accounting"


def _process_tree_rss(root_pid: int) -> tuple[set[int], int | None]:
    """Return all current descendants and their summed RSS in bytes."""

    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return set(), None

    rows: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, parent_pid, rss_kib = map(int, parts)
        except ValueError:
            continue
        rows[pid] = (parent_pid, rss_kib)

    process_ids = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _rss_kib) in rows.items():
            if parent_pid in process_ids and pid not in process_ids:
                process_ids.add(pid)
                changed = True
    observed = [rows[pid][1] for pid in process_ids if pid in rows]
    if not observed:
        return process_ids, None
    return process_ids, sum(observed) * 1024


def _nvidia_process_vram(process_ids: set[int], binary: str) -> int | None:
    """Return summed VRAM for the process tree, or None when unavailable."""

    try:
        result = subprocess.run(
            [
                binary,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    total_mib = 0
    found = False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            used_mib = int(parts[1])
        except ValueError:
            continue
        if pid in process_ids:
            total_mib += used_mib
            found = True
    return total_mib * 1024 * 1024 if found else None


def add_resource_metrics(report_path: Path, metrics: ResourceMetrics) -> bool:
    """Attach process measurements to a completed evaluation report."""

    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    resources = metrics.to_dict()
    report["resources"] = resources
    summary = report.setdefault("summary", {})
    summary["peak_ram_bytes"] = resources["peak_ram_bytes"]
    summary["peak_ram_mib"] = resources["peak_ram_mib"]
    summary["peak_vram_bytes"] = resources["peak_vram_bytes"]
    summary["peak_vram_mib"] = resources["peak_vram_mib"]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


def _bytes_to_mib(value: int | None) -> float | None:
    return value / (1024 * 1024) if value is not None else None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = REPOSITORY_ROOT / "evals" / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        choices=("llama-cpp", "vllm"),
        default="llama-cpp",
        help="server to start for evaluation (default: llama-cpp)",
    )
    parser.add_argument("--binary", help="path or command name for llama-server or vllm")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--offline", action="store_true", help="never download missing weights")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--replace-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop an existing selected-runtime listener on this port (default: enabled)",
    )
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument(
        "--resource-sample-interval",
        type=float,
        default=0.25,
        help="seconds between model-server RAM/VRAM samples (default: 0.25)",
    )
    parser.add_argument("--case-dir", type=Path, default=REPOSITORY_ROOT / "evals" / "cases")
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--server-log",
        type=Path,
    )
    parser.set_defaults(timestamp=timestamp, result_dir=result_dir)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    port = args.port or (8000 if args.runtime == "vllm" else 8080)
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.ctx_size < 1 or args.startup_timeout <= 0:
        raise ValueError("ctx-size and startup-timeout must be positive")
    if args.resource_sample_interval <= 0:
        raise ValueError("resource-sample-interval must be positive")
    if args.threads is not None and args.threads < 1:
        raise ValueError("threads must be positive")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")

    base_url = f"http://127.0.0.1:{port}"
    process_environment = os.environ.copy()
    if args.runtime == "vllm":
        binary = find_vllm_binary(args.binary)
        persistent_cache = (args.cache_dir / "huggingface").expanduser().resolve()
        persistent_cache.mkdir(parents=True, exist_ok=True)
        command = build_vllm_server_command(
            binary=binary,
            cache_dir=persistent_cache,
            host="127.0.0.1",
            port=port,
            context_size=args.ctx_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        replaced = (
            stop_existing_model_server(
                port, {Path(binary).name}, server_label="vLLM server"
            )
            if args.replace_existing
            else []
        )
        configure_vllm_cache(persistent_cache, process_environment, offline=args.offline)
        model_cache_description = str(persistent_cache)
        runtime_label = "vLLM"
        output_name = f"{args.timestamp}_qwen3-30b-a3b-fp8.json"
        log_name = f"{args.timestamp}_vllm-server.log"
    else:
        binary = find_server_binary(args.binary)
        model_path = ensure_cached(cache_dir=args.cache_dir, offline=args.offline)
        command = build_server_command(
            binary=binary,
            model_path=model_path,
            host="127.0.0.1",
            port=port,
            context_size=args.ctx_size,
            gpu_layers=args.gpu_layers,
            threads=args.threads,
        )
        replaced = stop_existing_llama_server(port) if args.replace_existing else []
        model_cache_description = str(model_path)
        runtime_label = "llama.cpp"
        output_name = f"{args.timestamp}_qwen3-4b-q4_k_m.json"
        log_name = f"{args.timestamp}_llama-server.log"

    runtime_version = server_version(binary)
    if replaced:
        print(f"Stopped existing model-server PID(s): {', '.join(map(str, replaced))}")
    if not args.replace_existing and server_is_ready(base_url):
        raise RuntimeError(
            f"a server is already responding at {base_url}; refusing to manage an existing process"
        )

    args.output = args.output or args.result_dir / output_name
    args.server_log = args.server_log or args.result_dir / log_name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using persistent model cache: {model_cache_description}")
    print(f"Starting {runtime_label} {runtime_version}")
    print(f"Server log: {args.server_log}")

    with args.server_log.open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_environment,
        )
        resource_monitor = ResourceMonitor(
            getattr(process, "pid", None), args.resource_sample_interval
        )
        resource_monitor.start()
        try:
            wait_until_ready(process, base_url, args.startup_timeout, args.server_log)
            print(f"Server ready at {base_url}; running {len(list(args.case_dir.glob('*.json')))} cases")
            evaluation_arguments = [
                "--base-url",
                base_url,
                "--runtime",
                args.runtime,
                "--case-dir",
                str(args.case_dir),
                "--output",
                str(args.output),
                "--runtime-version",
                runtime_version,
                "--server-command",
                json.dumps(command),
            ]
            if args.runtime == "vllm":
                evaluation_arguments.extend(
                    [
                        "--model",
                        VLLM_MODEL_REPOSITORY,
                        "--model-revision",
                        VLLM_MODEL_REVISION,
                    ]
                )
            else:
                evaluation_arguments.extend(
                    [
                        "--model-revision",
                        QWEN_GGUF.revision,
                        "--gguf-sha256",
                        QWEN_GGUF.sha256,
                    ]
                )
            return evaluation_main(evaluation_arguments)
        finally:
            resource_metrics = resource_monitor.stop()
            if add_resource_metrics(args.output, resource_metrics):
                ram = resource_metrics.to_dict()["peak_ram_mib"]
                vram = resource_metrics.to_dict()["peak_vram_mib"]
                ram_text = f"{ram:.1f} MiB" if isinstance(ram, float) else "unavailable"
                vram_text = f"{vram:.1f} MiB" if isinstance(vram, float) else "unavailable"
                print(f"Peak model-server RAM: {ram_text}; peak VRAM: {vram_text}")
            stop_server(process)
            if args.output.is_file():
                print(f"Stopped {runtime_label}; evaluation report: {args.output}")
            else:
                print(f"Stopped {runtime_label}; no evaluation report was produced; see {args.server_log}")


if __name__ == "__main__":
    sys.exit(main())
