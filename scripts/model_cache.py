"""Download and verify the pinned GGUF artifact into a persistent cache."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / "models"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    repository: str
    revision: str
    filename: str
    sha256: str

    @property
    def url(self) -> str:
        filename = quote(self.filename, safe="")
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{filename}"

    def cache_path(self, cache_dir: Path) -> Path:
        owner, name = self.repository.split("/", 1)
        return cache_dir / owner / name / self.revision / self.filename


QWEN_GGUF = ModelArtifact(
    repository="bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    revision="ae44f08e1392f39c0e474af10c3ff8355c8b6688",
    filename="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    sha256="2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e",
)

Download = Callable[[str, Path], None]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cached(
    artifact: ModelArtifact = QWEN_GGUF,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    offline: bool = False,
    download: Download | None = None,
) -> Path:
    """Return a verified cache path, downloading only when it is absent."""

    destination = artifact.cache_path(cache_dir.expanduser().resolve())
    if destination.is_file():
        actual = sha256_file(destination)
        if actual != artifact.sha256:
            raise RuntimeError(
                f"cached model checksum mismatch at {destination}: expected "
                f"{artifact.sha256}, got {actual}; move the corrupt file aside and retry"
            )
        return destination
    if offline:
        raise FileNotFoundError(f"verified model is not cached at {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    (download or _download_resumable)(artifact.url, partial)
    actual = sha256_file(partial)
    if actual != artifact.sha256:
        raise RuntimeError(
            f"downloaded model checksum mismatch at {partial}: expected "
            f"{artifact.sha256}, got {actual}; partial file retained for inspection"
        )
    partial.replace(destination)
    _write_metadata(destination, artifact)
    return destination


def _download_resumable(url: str, partial: Path) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "optiflow-ctrm-model-cache/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned HTTPS host
            status = getattr(response, "status", 200)
            mode = "ab" if offset and status == 206 else "wb"
            if offset and status != 206:
                print("Server ignored resume request; restarting download.", file=sys.stderr)
            with partial.open(mode) as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"model download failed; rerun to resume from {partial}: {exc}"
        ) from exc


def _write_metadata(model_path: Path, artifact: ModelArtifact) -> None:
    metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {**asdict(artifact), "download_url": artifact.url, "local_path": str(model_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--offline", action="store_true", help="verify cache without downloading")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = ensure_cached(cache_dir=args.cache_dir, offline=args.offline)
    print(path)
    print(f"sha256={QWEN_GGUF.sha256}")
    print(f"revision={QWEN_GGUF.revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
