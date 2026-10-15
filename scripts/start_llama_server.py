"""Verify cached Qwen weights and start the local llama.cpp HTTP server."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

try:
    from scripts.model_cache import DEFAULT_CACHE_DIR, QWEN_GGUF, ensure_cached
except ModuleNotFoundError:  # supports: python scripts/start_llama_server.py
    from model_cache import DEFAULT_CACHE_DIR, QWEN_GGUF, ensure_cached


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def build_server_command(
    *,
    binary: str,
    model_path: Path,
    host: str,
    port: int,
    context_size: int,
    gpu_layers: int,
    threads: int | None,
) -> list[str]:
    command = [
        binary,
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--n-gpu-layers",
        str(gpu_layers),
    ]
    if threads is not None:
        command.extend(["--threads", str(threads)])
    command.extend(["--seed", "42", "--metrics"])
    return command


def find_server_binary(requested: str | None) -> str:
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        resolved = shutil.which(requested)
        if resolved:
            return resolved
        raise FileNotFoundError(f"llama.cpp server binary is not executable: {requested}")
    resolved = shutil.which("llama-server")
    if not resolved:
        raise FileNotFoundError(
            "llama-server was not found on PATH; install llama.cpp or pass --binary"
        )
    return resolved


def server_version(binary: str) -> str:
    try:
        result = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({exc})"
    output = (result.stdout or result.stderr).strip()
    return output or f"unavailable (exit {result.returncode})"


def _listener_pids(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("lsof is required to replace an existing server safely") from exc
    return sorted({int(line) for line in result.stdout.splitlines() if line.strip().isdigit()})


def _process_command(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not inspect listener process {pid}") from exc
    return result.stdout.strip()


def stop_existing_model_server(
    port: int,
    executable_names: set[str],
    *,
    server_label: str,
    timeout_seconds: float = 15.0,
) -> list[int]:
    """Stop only listeners whose command contains an approved server executable."""

    pids = _listener_pids(port)
    if not pids:
        return []
    commands = {pid: _process_command(pid) for pid in pids}
    unsafe = {
        pid: command
        for pid, command in commands.items()
        if not any(Path(token).name in executable_names for token in shlex.split(command))
    }
    if unsafe:
        details = ", ".join(f"PID {pid}: {command or '<unknown>'}" for pid, command in unsafe.items())
        raise RuntimeError(
            f"listener on port {port} is not {server_label}; refusing to kill {details}"
        )

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError(f"permission denied stopping {server_label} PID {pid}") from exc

    deadline = time.monotonic() + timeout_seconds
    remaining = pids
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        listening = set(_listener_pids(port))
        remaining = [pid for pid in pids if pid in listening]
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError(f"permission denied killing {server_label} PID {pid}") from exc
    if _listener_pids(port):
        raise RuntimeError(f"port {port} remained occupied after stopping {server_label}")
    return pids


def stop_existing_llama_server(port: int, timeout_seconds: float = 15.0) -> list[int]:
    """Stop only verified llama-server listeners on ``port`` and return their PIDs."""

    return stop_existing_model_server(
        port,
        {"llama-server"},
        server_label="llama-server",
        timeout_seconds=timeout_seconds,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", help="path or command name for llama-server")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--offline", action="store_true", help="never download missing weights")
    parser.add_argument("--host", default="127.0.0.1", choices=sorted(LOOPBACK_HOSTS))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--replace-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop an existing llama-server listener on this port (default: enabled)",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="verify/download the model and print the launch command without starting it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.ctx_size < 1:
        raise ValueError("ctx-size must be positive")
    if args.threads is not None and args.threads < 1:
        raise ValueError("threads must be positive")

    model_path = ensure_cached(QWEN_GGUF, args.cache_dir, offline=args.offline)
    binary = find_server_binary(args.binary)
    replaced = stop_existing_llama_server(args.port) if args.replace_existing else []
    if replaced:
        print(f"Stopped existing llama-server PID(s): {', '.join(map(str, replaced))}")
    command = build_server_command(
        binary=binary,
        model_path=model_path,
        host=args.host,
        port=args.port,
        context_size=args.ctx_size,
        gpu_layers=args.gpu_layers,
        threads=args.threads,
    )
    print(f"Model cache: {model_path}")
    print(f"Model revision: {QWEN_GGUF.revision}")
    print(f"Model SHA-256: {QWEN_GGUF.sha256}")
    print(f"llama.cpp version: {server_version(binary)}")
    print(f"Server command: {shlex.join(command)}", flush=True)
    if args.print_command:
        return 0
    os.execv(binary, command)
    return 0  # unreachable; keeps the function's return contract explicit


if __name__ == "__main__":
    sys.exit(main())
